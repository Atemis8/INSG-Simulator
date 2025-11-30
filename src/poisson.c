#include "../headers/poisson.h"
#include "../headers/utils.h"
#include "../headers/config.h"

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
    
    // Fill local portion of RHS
    for (j = ys; j < ys + ym; j++) {
        for (i = xs; i < xs + xm; i++) {
            // Convert DMDA global indices to local ScalarField indices
            // DMDA uses 0-based global indexing
            // ScalarField uses local indexing with ghosts: [0...gw-1][gw...gw+nx-1][gw+nx...tnx-1]
            
            int local_x = (i - domain->start_x) + gw;  // Map to local interior + ghost offset
            int local_y = (j - domain->start_y) + gw;
            
            array[j][i] = get_scal(f, local_x, local_y);
        }
    }
    
    // Special handling for boundary mode: pin first point to zero
    if (xs == 0 && ys == 0 && mode == M_BOUNDARY) {
        array[0][0] = 0.0;
    }
    
    DMDAVecRestoreArray(da, b, &array);
}

void extractSolution_DMDA(Vec x, DM da, ScalarField *o, MPIDomain *domain) {
    PetscScalar **array;
    PetscInt i, j, xs, ys, xm, ym;
    
    DMDAVecGetArray(da, x, &array);
    DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL);
    
    int gw = domain->ghost_width;
    
    // Extract local portion of solution
    for (j = ys; j < ys + ym; j++) {
        for (i = xs; i < xs + xm; i++) {
            // Convert DMDA global indices to local ScalarField indices
            int local_x = (i - domain->start_x) + gw;
            int local_y = (j - domain->start_y) + gw;
            
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
        
    case M_BOUNDARY:
        // Boundary conditions: Neumann on most boundaries, Dirichlet at one point
        for (j = ys; j < ys + ym; j++) {
            for (i = xs; i < xs + xm; i++) {
                row.i = i; row.j = j;
                ncols = 0;
                
                // Special case: bottom-left corner (Dirichlet point to fix constant)
                if (i == 0 && j == 0) {
                    col[ncols].i = i; col[ncols].j = j; v[ncols++] = 1.0;
                }
                // Bottom-right corner
                else if (i == nx - 1 && j == 0) {
                    col[ncols].i = i; col[ncols].j = j; v[ncols++] = -2.0;
                    col[ncols].i = i - 1; col[ncols].j = j; v[ncols++] = 1.0;
                    col[ncols].i = i; col[ncols].j = j + 1; v[ncols++] = 1.0;
                }
                // Top-left corner
                else if (i == 0 && j == ny - 1) {
                    col[ncols].i = i; col[ncols].j = j; v[ncols++] = -2.0;
                    col[ncols].i = i + 1; col[ncols].j = j; v[ncols++] = 1.0;
                    col[ncols].i = i; col[ncols].j = j - 1; v[ncols++] = 1.0;
                }
                // Top-right corner
                else if (i == nx - 1 && j == ny - 1) {
                    col[ncols].i = i; col[ncols].j = j; v[ncols++] = -2.0;
                    col[ncols].i = i - 1; col[ncols].j = j; v[ncols++] = 1.0;
                    col[ncols].i = i; col[ncols].j = j - 1; v[ncols++] = 1.0;
                }
                // Bottom edge (not corners)
                else if (j == 0) {
                    col[ncols].i = i; col[ncols].j = j; v[ncols++] = -3.0;
                    col[ncols].i = i - 1; col[ncols].j = j; v[ncols++] = 1.0;
                    col[ncols].i = i + 1; col[ncols].j = j; v[ncols++] = 1.0;
                    col[ncols].i = i; col[ncols].j = j + 1; v[ncols++] = 1.0;
                }
                // Top edge (not corners)
                else if (j == ny - 1) {
                    col[ncols].i = i; col[ncols].j = j; v[ncols++] = -3.0;
                    col[ncols].i = i - 1; col[ncols].j = j; v[ncols++] = 1.0;
                    col[ncols].i = i + 1; col[ncols].j = j; v[ncols++] = 1.0;
                    col[ncols].i = i; col[ncols].j = j - 1; v[ncols++] = 1.0;
                }
                // Left edge (not corners)
                else if (i == 0) {
                    col[ncols].i = i; col[ncols].j = j; v[ncols++] = -3.0;
                    col[ncols].i = i + 1; col[ncols].j = j; v[ncols++] = 1.0;
                    col[ncols].i = i; col[ncols].j = j - 1; v[ncols++] = 1.0;
                    col[ncols].i = i; col[ncols].j = j + 1; v[ncols++] = 1.0;
                }
                // Right edge (not corners)
                else if (i == nx - 1) {
                    col[ncols].i = i; col[ncols].j = j; v[ncols++] = -3.0;
                    col[ncols].i = i - 1; col[ncols].j = j; v[ncols++] = 1.0;
                    col[ncols].i = i; col[ncols].j = j - 1; v[ncols++] = 1.0;
                    col[ncols].i = i; col[ncols].j = j + 1; v[ncols++] = 1.0;
                }
                // Interior points
                else {
                    col[ncols].i = i; col[ncols].j = j; v[ncols++] = -4.0;
                    col[ncols].i = i - 1; col[ncols].j = j; v[ncols++] = 1.0;
                    col[ncols].i = i + 1; col[ncols].j = j; v[ncols++] = 1.0;
                    col[ncols].i = i; col[ncols].j = j - 1; v[ncols++] = 1.0;
                    col[ncols].i = i; col[ncols].j = j + 1; v[ncols++] = 1.0;
                }
                
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
    
    /* Set ghost cell values based on boundary conditions */

     PetscInt nx = data->domain->tnx;  // total including ghosts
    PetscInt ny = data->domain->tny;
    int gw = data->domain->ghost_width;

    switch (data->mode) {

    case M_BOUNDARY:
        /* Neumann BCs: ghost = interior */

        /* Corners */
        set_scal(o, gw-1,     gw-1,     get_scal(o, gw,     gw));
        set_scal(o, nx-gw,    gw-1,     get_scal(o, nx-gw-1,gw));
        set_scal(o, gw-1,     ny-gw,    get_scal(o, gw,     ny-gw-1));
        set_scal(o, nx-gw,    ny-gw,    get_scal(o, nx-gw-1,ny-gw-1));

        /* Bottom and top boundaries */
        for (int x = gw; x < nx-gw; x++) {
            set_scal(o, x, gw-1,     get_scal(o, x, gw));
            set_scal(o, x, ny-gw,    get_scal(o, x, ny-gw-1));
        }

        /* Left and right boundaries */
        for (int y = gw; y < ny-gw; y++) {
            set_scal(o, gw-1,    y, get_scal(o, gw,      y));
            set_scal(o, nx-gw,   y, get_scal(o, nx-gw-1, y));
        }

        break;


    case M_PERIODIC:
        /* Periodic BCs: wrap ghost cells */

        /* Left/right periodic */
        for (int y = gw; y < ny-gw; y++) {
            // left ghosts
            for (int k = 0; k < gw; k++)
                set_scal(o, k, y, get_scal(o, nx-2*gw + k, y));

            // right ghosts
            for (int k = 0; k < gw; k++)
                set_scal(o, nx-gw + k, y, get_scal(o, gw + k, y));
        }

        /* Top/bottom periodic */
        for (int x = 0; x < nx; x++) {
            for (int k = 0; k < gw; k++)
                set_scal(o, x, k,          get_scal(o, x, ny-2*gw + k));
            for (int k = 0; k < gw; k++)
                set_scal(o, x, ny-gw + k,  get_scal(o, x, gw + k));
        }

        break;
    }
    
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