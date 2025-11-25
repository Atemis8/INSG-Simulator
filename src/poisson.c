#include "../headers/poisson.h"
#include "../headers/utils.h"
#include "../headers/config.h"

#ifdef USE_MPI
#include <mpi.h>
#endif

/* Will transform the scalar field into something usable by the solver (remove useless points) */
void computeRHS(double *rhs, PetscInt rowStart, PetscInt rowEnd, ScalarField *f, int mode) {
    int nx_in = f->nx - 2;

    for (PetscInt row = rowStart; row < rowEnd; ++row) {
        int y = row / nx_in;
        int x = row % nx_in;

        rhs[row - rowStart] = get_scal(f, x + 1, y + 1);
    }

    if (rowStart == 0 && mode == M_BOUNDARY) rhs[0] = 0.0;
}

/*To call at each time step after computation of U_star. This function solves the poisson equation*/
/*and copies the solution of the equation into your vector Phi*/
/*More than probably, you should need to add arguments to the prototype ... */
/*Modification to do :*/
/*    - Change the call to computeRHS as you have to modify its prototype too*/
/*    - Copy solution of the equation into your vector PHI*/
void poisson_solver(Poisson_data *data, ScalarField *i, ScalarField *o) {
    int          its;
    PetscInt     rowStart, rowEnd;
    PetscScalar *rhs, *sol;

    KSP sles = data->sles;
    Vec b    = data->b;
    Vec x    = data->x;
    
    /* Fill the right-hand-side vector : b */
    VecGetOwnershipRange(b, &rowStart, &rowEnd);
    VecGetArray(b, &rhs);
    computeRHS(rhs, rowStart, rowEnd, i, data->mode);
    VecRestoreArray(b, &rhs);

    if (data->mode == M_PERIODIC) {
        MatNullSpace nullsp;
        MatGetNullSpace(data->A, &nullsp);
        MatNullSpaceRemove(nullsp, data->b);
    }

    /*Solve the linear system of equations */
    KSPSolve(sles, b, x);
    KSPGetIterationNumber(sles, &its);

    if (data->mode == M_PERIODIC) {
        PetscScalar sum_phi;
        VecSum(x, &sum_phi);
        VecShift(x, -sum_phi / ((o->nx-2) * (o->ny-2)));
    }

    // Get local portion of solution
    VecGetOwnershipRange(x, &rowStart, &rowEnd);
    VecGetArray(x, &sol);

    int nx = o->nx;
    int ny = o->ny;

    // Each rank fills in its portion of the solution
    for(PetscInt idx = rowStart; idx < rowEnd; ++idx) {
        int y = idx / (nx-2);
        int x = idx % (nx-2);
        set_scal(o, x + 1, y + 1, sol[idx - rowStart]);
    }

    VecRestoreArray(x, &sol);
    
#ifdef USE_MPI
    // Gather the complete solution to all ranks
    // This is necessary because your ScalarField is replicated on all ranks
    PetscScalar *global_sol;

    /* scatter from distributed x to sequential x_global */
    VecScatterBegin(data->scatter_ctx, x, data->x_global, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(data->scatter_ctx, x, data->x_global, INSERT_VALUES, SCATTER_FORWARD);

    /* access the gathered vector */
    VecGetArray(data->x_global, &global_sol);

    for(int x = 1; x < nx-1; ++x) {
        for(int y = 1; y < ny-1; ++y) {
            int idx = ((y-1) * (nx-2)) + (x-1);
            set_scal(o, x, y, global_sol[idx]);
        }
    }

    VecRestoreArray(data->x_global, &global_sol);
#endif

    // Set boundary values
    switch(data->mode) {
    case M_PERIODIC:
        set_scal(o, 0, 0, get_scal(o, nx-2, ny-2));
        set_scal(o, nx-1, 0, get_scal(o, 1, ny-2));
        set_scal(o, 0, ny-1, get_scal(o, nx-2, 1));
        set_scal(o, nx-1, ny-1, get_scal(o, 1, 1));

        // Top and Bottom rows
        for(int x = 1; x < nx-1; ++x) {
            set_scal(o, x, 0, get_scal(o, x, ny-2));
            set_scal(o, x, ny-1, get_scal(o, x, 1));
        }

        // Left and Right columns
        for(int y = 1; y < ny-1; ++y) {
            set_scal(o, 0, y, get_scal(o, nx-2, y));
            set_scal(o, nx-1, y, get_scal(o, 1, y));
        }
        break;
        
    case M_BOUNDARY:

        // First the 4 cornes which are equal to the diagonal term
        set_scal(o, 0, 0, get_scal(o, 1, 1));
        set_scal(o, nx-1, 0, get_scal(o, nx-2, 1));
        set_scal(o, 0, ny-1, get_scal(o, 1, ny-2));
        set_scal(o, nx-1, ny-1, get_scal(o, nx-2, ny-2));

        // Now do top and bottom rows
        for(int x = 1; x < nx-1; ++x) {
            set_scal(o, x, 0, get_scal(o, x, 1));
            set_scal(o, x, ny-1, get_scal(o, x, ny-2));
        }

        for(int y = 1; y < ny-1; ++y) {
            set_scal(o, 0, y, get_scal(o, 1, y));
            set_scal(o, nx-1, y, get_scal(o, nx-2, y));
        }
        break;
    }
}

/*This function is called only once during the simulation, i.e. in initialize_poisson_solver.*/
/*In its current state, it inserts unity on the main diagonal.*/
/*More than probably, you should need to add arguments to the prototype ... .*/
/*Modification to do in this function : */
/*   -Insert the correct factor in matrix A*/
void computeLaplacianMatrix(Mat A, int rowStart, int rowEnd, ScalarField *f, int flagtype) {
    // Consider a N x M grid, a point (i, j) on the grid 

    int r;
    int nx = f->nx-2;
    int ny = f->ny-2;

    switch (flagtype) {
    case M_PERIODIC:
        for (r = rowStart; r < rowEnd; r++) {
            int y = r / nx;
            int x = r % nx;
            
            // Calculate indices for neighbors, with periodic wrapping
            int left = (x == 0) ? r + nx - 1 : r - 1;
            int right = (x == nx - 1) ? r - nx + 1 : r + 1;
            int up = (y == 0) ? r + nx * (ny - 1) : r - nx;
            int down = (y == ny - 1) ? r - nx * (ny - 1) : r + nx;
            
            // Set matrix values
            MatSetValue(A, r, r, -4.0, INSERT_VALUES);
            MatSetValue(A, r, left, 1.0, INSERT_VALUES);
            MatSetValue(A, r, right, 1.0, INSERT_VALUES);
            MatSetValue(A, r, up, 1.0, INSERT_VALUES);
            MatSetValue(A, r, down, 1.0, INSERT_VALUES);
        }
        break;
    
    case M_BOUNDARY:
        for (r = rowStart; r < rowEnd; r++) {
            if (r == 0) {
                MatSetValue(A, r, r, 1.0, INSERT_VALUES);
            } else if (r < nx - 1) {
                MatSetValue(A, r, r, -3.0, INSERT_VALUES);
                MatSetValue(A, r, r - 1, 1.0, INSERT_VALUES);
                MatSetValue(A, r, r + 1, 1.0, INSERT_VALUES);
                MatSetValue(A, r, r + nx, 1.0, INSERT_VALUES);
            } else if (r == nx - 1) {
                MatSetValue(A, r, r, -2.0, INSERT_VALUES);
                MatSetValue(A, r, r - 1, 1.0, INSERT_VALUES);
                MatSetValue(A, r, r + nx, 1.0, INSERT_VALUES);
            } else if (r >= nx && r < nx * (ny - 1)) {
                if(r % nx == 0) {
                    MatSetValue(A, r, r, -3.0, INSERT_VALUES);
                    MatSetValue(A, r, r + 1, 1.0, INSERT_VALUES);
                    MatSetValue(A, r, r - nx, 1.0, INSERT_VALUES);
                    MatSetValue(A, r, r + nx, 1.0, INSERT_VALUES);
                } else if(r % nx == nx - 1) {
                    MatSetValue(A, r, r, -3.0, INSERT_VALUES);
                    MatSetValue(A, r, r - 1, 1.0, INSERT_VALUES);
                    MatSetValue(A, r, r - nx, 1.0, INSERT_VALUES);
                    MatSetValue(A, r, r + nx, 1.0, INSERT_VALUES);
                } else {
                    MatSetValue(A, r, r, -4.0, INSERT_VALUES);
                    MatSetValue(A, r, r + 1, 1.0, INSERT_VALUES);
                    MatSetValue(A, r, r - 1, 1.0, INSERT_VALUES);
                    MatSetValue(A, r, r + nx, 1.0, INSERT_VALUES);
                    MatSetValue(A, r, r - nx, 1.0, INSERT_VALUES);
                }
            } else if (r == nx * (ny - 1)) {
                MatSetValue(A, r, r, -2.0, INSERT_VALUES);
                MatSetValue(A, r, r + 1, 1.0, INSERT_VALUES);
                MatSetValue(A, r, r - nx, 1.0, INSERT_VALUES);
            } else if (r < nx * ny - 1) {
                MatSetValue(A, r, r, -3.0, INSERT_VALUES);
                MatSetValue(A, r, r - 1, 1.0, INSERT_VALUES);
                MatSetValue(A, r, r + 1, 1.0, INSERT_VALUES);
                MatSetValue(A, r, r - nx, 1.0, INSERT_VALUES);
            } else if (r == nx * ny - 1) {
                MatSetValue(A, r, r, -2.0, INSERT_VALUES);
                MatSetValue(A, r, r - 1, 1.0, INSERT_VALUES);
                MatSetValue(A, r, r - nx, 1.0, INSERT_VALUES);
            }
        }
        break;
    default:
        mprintf("Error: Unknown flagtype in computeLaplacianMatrix\n");
        break;
    }
}

/*To call during the initialization of your solver, before the begin of the time loop*/
/*Maybe you should need to add an argument to specify the number of unknows*/
/*Modification to do in this function :*/
/*   -Specify the number of unknows*/
/*   -Specify the number of non-zero diagonals in the sparse matrix*/
PetscErrorCode initialize_poisson_solver(Poisson_data *data, ScalarField* o, int mode) {
    PetscInt       rowStart, rowEnd;
    PetscErrorCode ierr;

    int nphi = (o->nx-2) * (o->ny-2);
    data->mode = mode;

    /* Create the right-hand-side vector : b */
    ierr = VecCreate(PETSC_COMM_WORLD, &(data->b)); CHKERRQ(ierr);
    ierr = VecSetSizes(data->b, PETSC_DECIDE, nphi); CHKERRQ(ierr);
    ierr = VecSetFromOptions(data->b); CHKERRQ(ierr);

    /* Create the solution vector : x */
    ierr = VecCreate(PETSC_COMM_WORLD, &(data->x)); CHKERRQ(ierr);
    ierr = VecSetSizes(data->x, PETSC_DECIDE, nphi); CHKERRQ(ierr);
    ierr = VecSetFromOptions(data->x); CHKERRQ(ierr);

#ifdef USE_MPI
    /* Create global vector for gathering results */
    ierr = VecCreateSeq(PETSC_COMM_SELF, nphi, &(data->x_global)); CHKERRQ(ierr);
    
    /* Create scatter context for gathering */
    ierr = VecScatterCreateToAll(data->x, &(data->scatter_ctx), &(data->x_global)); CHKERRQ(ierr);
#endif

    /* Create and assemble the Laplacian matrix : A  */
    ierr = MatCreate(PETSC_COMM_WORLD, &(data->A)); CHKERRQ(ierr);
    ierr = MatSetSizes(data->A, PETSC_DECIDE, PETSC_DECIDE, nphi, nphi); CHKERRQ(ierr);
    ierr = MatSetFromOptions(data->A); CHKERRQ(ierr);
    ierr = MatSetUp(data->A); CHKERRQ(ierr);
    
    ierr = MatMPIAIJSetPreallocation(data->A, 5, NULL, 5, NULL); CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(data->A, 5, NULL); CHKERRQ(ierr);
    
    ierr = MatGetOwnershipRange(data->A, &rowStart, &rowEnd); CHKERRQ(ierr);

    computeLaplacianMatrix(data->A, rowStart, rowEnd, o, mode);
    
    ierr = MatAssemblyBegin(data->A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(data->A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    if (mode == M_PERIODIC) {
        MatNullSpace nullspace;
        ierr = MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, NULL, &nullspace); CHKERRQ(ierr);
        ierr = MatSetNullSpace(data->A, nullspace); CHKERRQ(ierr);
        ierr = MatNullSpaceDestroy(&nullspace); CHKERRQ(ierr);
    }

    /* Create the Krylov context */
    ierr = KSPCreate(PETSC_COMM_WORLD, &(data->sles)); CHKERRQ(ierr);
    ierr = KSPSetOperators(data->sles, data->A, data->A); CHKERRQ(ierr);
    ierr = KSPSetType(data->sles, KSPGMRES); CHKERRQ(ierr);
    
    PC prec;
    ierr = KSPGetPC(data->sles, &prec); CHKERRQ(ierr);
    
#ifdef USE_MPI
    ierr = PCSetType(prec, PCASM); CHKERRQ(ierr);
    ierr = KSPSetTolerances(data->sles, 1.e-12, 1e-12, PETSC_DEFAULT, 1000); CHKERRQ(ierr);
#else
    ierr = PCSetType(prec, PCLU); CHKERRQ(ierr);
    ierr = KSPSetTolerances(data->sles, 1.e-12, 1e-12, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRQ(ierr);
#endif
    
    ierr = KSPSetReusePreconditioner(data->sles, PETSC_TRUE); CHKERRQ(ierr);
    ierr = KSPSetUseFischerGuess(data->sles, 1, 4); CHKERRQ(ierr);
    ierr = KSPGMRESSetPreAllocateVectors(data->sles); CHKERRQ(ierr);

    PetscPrintf(PETSC_COMM_WORLD, "Assembly of Matrix and Vectors is done\n");

    return ierr;
}

/*To call after the simulation to free the vectors needed for Poisson equation*/
/*Modification to do : nothing */
void free_poisson_solver(Poisson_data *data) {
    MatDestroy(&(data->A));
    VecDestroy(&(data->b));
    VecDestroy(&(data->x));
    KSPDestroy(&(data->sles));
#ifdef USE_MPI
    VecDestroy(&(data->x_global));
    VecScatterDestroy(&(data->scatter_ctx));
#endif
}