#include "../headers/poisson.h"
#include "../headers/utils.h"
#include "../headers/config.h"
#include "../headers/finite_diff.h"

#ifdef USE_MPI
#include <mpi.h>
#endif
#include <assert.h>
#include <petscdmda.h>

void computeRHS_DMDA(Vec b, DM da, ScalarField *f, Mode mode, MPIDomain *domain) {
    PetscScalar **array;
    PetscInt i, j, xs, ys, xm, ym;
    
    DMDAVecGetArray(da, b, &array);
    DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL);
    
    int gw = domain->ghost_width;
    
    // Verify we're accessing the right global range
    assert(xs == domain->start_x && ys == domain->start_y);
    assert(xm == domain->nx && ym == domain->ny);
    
    for (j = ys; j < ys + ym; j++) {
        for (i = xs; i < xs + xm; i++) {
            // Map global DMDA indices to local ScalarField indices
            // i ∈ [start_x, start_x+nx) → local_x ∈ [gw, gw+nx)
            int local_x = (i - xs) + gw; 
            int local_y = (j - ys) + gw;
            
            // Safety check (remove after debugging)
            assert(local_x >= gw && local_x < gw + domain->nx);
            assert(local_y >= gw && local_y < gw + domain->ny);
            
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

    assert(xs == domain->start_x && ys == domain->start_y);
    assert(xm == domain->nx && ym == domain->ny);
    
    int gw = domain->ghost_width;
    
    for (j = ys; j < ys + ym; j++) {
        for (i = xs; i < xs + xm; i++) {
            // Map global DMDA indices to local ScalarField indices
            int local_x = (i - xs) + gw;  // CORRECTED
            int local_y = (j - ys) + gw;  // CORRECTED
            
            // Safety check
            assert(local_x >= gw && local_x < gw + domain->nx);
            assert(local_y >= gw && local_y < gw + domain->ny);
            
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
                col[ncols].i = i - 1; col[ncols].j = j; v[ncols++] = 1.0;
                col[ncols].i = i + 1; col[ncols].j = j; v[ncols++] = 1.0;
                col[ncols].i = i; col[ncols].j = j - 1; v[ncols++] = 1.0;
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
    // Add diagnostics
    PetscInt its;
    PetscReal rnorm;
    KSPGetIterationNumber(data->sles, &its);
    KSPGetResidualNorm(data->sles, &rnorm);
    
    KSPConvergedReason reason;
    KSPGetConvergedReason(data->sles, &reason);
    
    PetscPrintf(PETSC_COMM_WORLD, " KSP: its=%d, rnorm=%.2e, ", its, rnorm);
    
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
#ifndef USE_MPI
    apply_periodic_bc(o);
#endif
}

/*
 * initialize_poisson_solver: Set up DMDA, matrix, vectors, and solver
 * Called once during simulation initialization
 */
PetscErrorCode init_poisson_solver(MPIDomain *domain, Poisson_data *data, ScalarField* o, int mode) {
    PetscErrorCode ierr;

    data->mode = mode;
    data->domain = domain;

    DM da;
    DMBoundaryType bx = (mode == M_PERIODIC) ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_NONE;
    DMBoundaryType by = (mode == M_PERIODIC) ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_NONE;
    
    PetscInt grid_nx = domain->global_nx;
    PetscInt grid_ny = domain->global_ny;
    
    // Build arrays describing how many points each process has in each direction
    PetscInt *lx = (PetscInt*)malloc(domain->dims[0] * sizeof(PetscInt));
    PetscInt *ly = (PetscInt*)malloc(domain->dims[1] * sizeof(PetscInt));
    
    int base_nx = grid_nx / domain->dims[0];
    int extra_nx = grid_nx % domain->dims[0];
    int base_ny = grid_ny / domain->dims[1];
    int extra_ny = grid_ny % domain->dims[1];
    
    // X-direction distribution (must match your MPIDomain calculation!)
    for (int i = 0; i < domain->dims[0]; i++)
        lx[i] = base_nx + (i < extra_nx ? 1 : 0);
    
    // Y-direction distribution (must match your MPIDomain calculation!)
    for (int j = 0; j < domain->dims[1]; j++)
        ly[j] = base_ny + (j < extra_ny ? 1 : 0);
    
    // Verify the sum is correct
    PetscInt sum_x = 0, sum_y = 0;
    for (int i = 0; i < domain->dims[0]; i++) sum_x += lx[i];
    for (int j = 0; j < domain->dims[1]; j++) sum_y += ly[j];
    assert(sum_x == grid_nx);
    assert(sum_y == grid_ny);

    #ifdef USE_MPI
        MPI_Comm petsc_comm = domain->cart_comm;
    #else
        MPI_Comm petsc_comm = PETSC_COMM_WORLD;
    #endif
        
    // Create DMDA with YOUR decomposition
    ierr = DMDACreate2d(petsc_comm,
                       bx, by,
                       DMDA_STENCIL_STAR,
                       grid_nx, grid_ny,
                       domain->dims[0], domain->dims[1],
                       1,    // 1 DOF per node
                       1,    // Stencil width = 1
                       lx, ly,  // ← YOUR decomposition, not NULL!
                       &da); CHKERRQ(ierr);
    
    free(lx);
    free(ly);
    
    ierr = DMSetFromOptions(da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);
    
    // ============ Verify DMDA matches MPIDomain ============
    PetscInt xs, ys, xm, ym;
    DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL);

    PetscSynchronizedPrintf(PETSC_COMM_WORLD,
        "Rank %d: cart_coords=(%d,%d) start=(%d,%d) nx=%d,ny=%d | DMDA: xs=%d,ys=%d,xm=%d,ym=%d\n",
        domain->rank, domain->coords[0], domain->coords[1],
        domain->start_x, domain->start_y, domain->nx, domain->ny,
        (int)xs, (int)ys, (int)xm, (int)ym);
    PetscSynchronizedFlush(PETSC_COMM_WORLD, STDOUT_FILENO);
    
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
        ierr = PCSetType(prec, PCGAMG); CHKERRQ(ierr);

        /* AMG controls */
        ierr = PCGAMGSetNSmooths(prec, 1); CHKERRQ(ierr);
        ierr = PCGAMGSetType(prec, PCGAMGAGG); CHKERRQ(ierr);
        ierr = PCGAMGSetNlevels(prec, 4); CHKERRQ(ierr);
        ierr = PCGAMGSetAggressiveLevels(prec, 1); CHKERRQ(ierr);
        ierr = PCGAMGSetThresholdScale(prec, 1.0); CHKERRQ(ierr); // 1.0 means no extra scaling
        {
            PetscReal thr = 0.04;
            ierr = PCGAMGSetThreshold(prec, &thr, 1); CHKERRQ(ierr);
        }
    }

    
    /* Set solver tolerances and options */
    ierr = KSPSetTolerances(data->sles, 1e-8, 1e-8, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRQ(ierr);
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