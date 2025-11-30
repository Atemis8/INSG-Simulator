#include "../headers/poisson.h"
#include "../headers/utils.h"
#include "../headers/config.h"
#include "../headers/finite_diff.h"

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <petscdmda.h>

void computeRHS_DMDA(Vec b, DM da, ScalarField *f, Mode mode, MPIDomain *domain) {
    PetscScalar **array;
    PetscInt i, j, xs, ys, xm, ym;

    DMDAVecGetArray(da, b, &array);
    DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL);

    int gw = domain->ghost_width;

    /* Fill local portion of RHS: loop over DMDA global indices (xs..xs+xm-1) */
    for (j = ys; j < ys + ym; j++) {
        for (i = xs; i < xs + xm; i++) {
            /* Map DMDA (global) index (i,j) -> local ScalarField index (with gw) */
            int local_x = (int)(i - xs) + gw;
            int local_y = (int)(j - ys) + gw;


            array[j][i] = get_scal(f, local_x, local_y);
        }
    }

    DMDAVecRestoreArray(da, b, &array);
}


void extractSolution_DMDA(Vec x, DM da, ScalarField *o, MPIDomain *domain) {
    PetscScalar **array;
    PetscInt i, j, xs, ys, xm, ym;

    DMDAVecGetArray(da, x, &array);
    DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL);

    int gw = domain->ghost_width;

    /* Extract local portion of solution */
    for (j = ys; j < ys + ym; j++) {
        for (i = xs; i < xs + xm; i++) {
            /* Map DMDA global index to local field index */
            int local_x = (int)(i - xs) + gw;
            int local_y = (int)(j - ys) + gw;

            set_scal(o, local_x, local_y, array[j][i]);
        }
    }

    DMDAVecRestoreArray(da, x, &array);

}

/*
 * computeLaplacianMatrix_DMDA: Assemble Laplacian matrix using DMDA stencil
 * Handles both periodic and boundary conditions
 */
void computeLaplacianMatrix_DMDA(Mat A, DM da, ScalarField *f, int flagtype) {
    PetscInt i, j, xs, ys, xm, ym, nx, ny;
    MatStencil row, col[5];
    PetscScalar v[5];
    PetscInt ncols;
    
    // Get local portion of grid
    DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL);
    DMDAGetInfo(da, NULL, &nx, &ny, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
    
    switch (flagtype) {
    case M_PERIODIC:
        // Periodic boundaries: standard 5-point stencil everywhere
        for (j = ys; j < ys + ym; j++) {
            for (i = xs; i < xs + xm; i++) {
                row.i = i; row.j = j;
                ncols = 0;
                
                // Center: -4
                col[ncols].i = i; col[ncols].j = j; v[ncols++] = -4.0;
                
                // West: +1
                col[ncols].i = i - 1; col[ncols].j = j; v[ncols++] = 1.0;
                
                // East: +1
                col[ncols].i = i + 1; col[ncols].j = j; v[ncols++] = 1.0;
                
                // South: +1
                col[ncols].i = i; col[ncols].j = j - 1; v[ncols++] = 1.0;
                
                // North: +1
                col[ncols].i = i; col[ncols].j = j + 1; v[ncols++] = 1.0;
                
                MatSetValuesStencil(A, 1, &row, ncols, col, v, INSERT_VALUES);
            }
        }
        break;
    default:
        mprintf("Error: Unknown flagtype in computeLaplacianMatrix_DMDA\n");
        break;
    }
}

/*
 * poisson_solver: Solve Poisson equation and update solution field
 * Handles both periodic and boundary conditions
 */
void poisson_solver(Poisson_data *data, ScalarField *i, ScalarField *o) {
    int its;
    
    /* Fill right-hand-side vector */
    computeRHS_DMDA(data->b, data->da, i, data->mode, data->domain);
    
    /* For periodic: remove null space (constant) */
    if (data->mode == M_PERIODIC) {
        MatNullSpace nullsp;
        MatGetNullSpace(data->A, &nullsp);
        MatNullSpaceRemove(nullsp, data->b);
    }
    
    /* Solve the linear system */
    KSPSolve(data->sles, data->b, data->x);
    KSPGetIterationNumber(data->sles, &its);
    
    /* For periodic: shift solution to have zero mean */
    if (data->mode == M_PERIODIC) {
        PetscScalar sum_phi;
        VecSum(data->x, &sum_phi);
        PetscInt nx, ny;
        DMDAGetInfo(data->da, NULL, &nx, &ny, NULL, NULL, NULL, NULL, 
                   NULL, NULL, NULL, NULL, NULL, NULL);
        VecShift(data->x, -sum_phi / (nx * ny));
    }
    
    /* Extract solution to ScalarField */
    extractSolution_DMDA(data->x, data->da, o, data->domain);
    apply_periodic_bc(o);
}

/*
 * initialize_poisson_solver: Set up DMDA, matrix, vectors, and solver
 * Called once during simulation initialization
 */
PetscErrorCode init_poisson_solver(MPIDomain *domain, Poisson_data *data, ScalarField* o, int mode) {
    PetscErrorCode ierr;

    data->mode = mode;
    data->domain = domain;

    /* Create DMDA for structured grid management */
    DM da;
    DMBoundaryType bx = (mode == M_PERIODIC) ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_NONE;
    DMBoundaryType by = (mode == M_PERIODIC) ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_NONE;
    
    // Grid size excludes ghost cells (interior points only)
    PetscInt grid_nx = domain->global_nx;
    PetscInt grid_ny = domain->global_ny;
    
    ierr = DMDACreate2d(PETSC_COMM_WORLD,
                       bx, by,
                       DMDA_STENCIL_STAR,
                       grid_nx, grid_ny,     // Global interior grid size
                       domain->dims[0], domain->dims[1],
                       1,                     // 1 DOF per node (scalar)
                       1,                     // Stencil width = 1
                       NULL, NULL,
                       &da); CHKERRQ(ierr);
    ierr = DMSetFromOptions(da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);
    
    data->da = da;

    /* Create vectors from DMDA (automatically distributed) */
    ierr = DMCreateGlobalVector(da, &(data->b)); CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(da, &(data->x)); CHKERRQ(ierr);

    /* Create matrix from DMDA (automatically sets up parallel structure) */
    ierr = DMCreateMatrix(da, &(data->A)); CHKERRQ(ierr);
    
    /* Assemble Laplacian matrix */
    computeLaplacianMatrix_DMDA(data->A, da, o, mode);
    
    ierr = MatAssemblyBegin(data->A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(data->A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    /* For periodic: set null space to handle singular system */
    if (mode == M_PERIODIC) {
        MatNullSpace nullspace;
        ierr = MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, NULL, &nullspace); CHKERRQ(ierr);
        ierr = MatSetNullSpace(data->A, nullspace); CHKERRQ(ierr);
        ierr = MatNullSpaceDestroy(&nullspace); CHKERRQ(ierr);
    }

    /* Create Krylov solver */
    ierr = KSPCreate(PETSC_COMM_WORLD, &(data->sles)); CHKERRQ(ierr);
    ierr = KSPSetOperators(data->sles, data->A, data->A); CHKERRQ(ierr);
    ierr = KSPSetType(data->sles, KSPCG); CHKERRQ(ierr);
    
    /* Set preconditioner */
    PC prec;
    ierr = KSPGetPC(data->sles, &prec); CHKERRQ(ierr);
    
    int size;
    MPI_Comm_size(PETSC_COMM_WORLD, &size);
    
    if (size == 1) {
        // Sequential: use direct LU solver
        ierr = PCSetType(prec, PCLU); CHKERRQ(ierr);
    } else {
        // Parallel: use block Jacobi with LU on each block
        ierr = PCSetType(prec, PCBJACOBI); CHKERRQ(ierr);
    }
    
    /* Set solver tolerances and options */
    ierr = KSPSetTolerances(data->sles, 1.e-12, 1e-12, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRQ(ierr);
    ierr = KSPSetReusePreconditioner(data->sles, PETSC_TRUE); CHKERRQ(ierr);
    ierr = KSPSetUseFischerGuess(data->sles, 1, 4); CHKERRQ(ierr);
    
    /* Allow command-line options to override settings */
    ierr = KSPSetFromOptions(data->sles); CHKERRQ(ierr);

    PetscPrintf(PETSC_COMM_WORLD, "Assembly of Matrix and Vectors is done\n");

    return ierr;
}

/*
 * free_poisson_solver: Clean up PETSc objects
 * Called at end of simulation
 */
void free_poisson_solver(Poisson_data *data) {
    DMDestroy(&(data->da));
    MatDestroy(&(data->A));
    VecDestroy(&(data->b));
    VecDestroy(&(data->x));
    KSPDestroy(&(data->sles));
}