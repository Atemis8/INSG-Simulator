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
    // Since MPIDomain now matches DMDA, this is simple

    assert(xm == domain->nx && ym == domain->ny);
    assert(xs == domain->start_x && ys == domain->start_y);
    assert(xm < f->nx && ym < f->ny);

    for (j = ys; j < ys + ym; j++) {
        for (i = xs; i < xs + xm; i++) {
            int local_x = (i - xs) + gw;
            int local_y = (j - ys) + gw;

            assert(xs <= i && i <= xs+xm-1);
            assert(ys <= j && j <= ys+ym-1);

            array[j][i] = get_scal(f, local_x, local_y);
        }
    }
    
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
    
    for (j = ys; j < ys + ym; j++) {
        for (i = xs; i < xs + xm; i++) {
            int local_x = (i - xs) + gw;
            int local_y = (j - ys) + gw;
             
            set_scal(o, local_x, local_y, array[j][i]);
        }
    }
    DMDAVecRestoreArray(da, x, &array);
}

/*
 * computeLaplacianMatrix_DMDA: Assemble Laplacian matrix using DMDA stencil
 * Handles both periodic and boundary conditions
 */
void computeLaplacianMatrix_DMDA(Mat A, DM da, int flagtype) {
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
// In poisson.c
PetscErrorCode init_poisson_solver(MPIDomain *domain, Poisson_data *data, Mode mode) {
    PetscErrorCode ierr;

    data->mode = mode;
    data->domain = domain;

    DM da;
    DMBoundaryType bx = (mode == M_PERIODIC) ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_NONE;
    DMBoundaryType by = (mode == M_PERIODIC) ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_NONE;
    
    PetscInt grid_nx = domain->global_nx;
    PetscInt grid_ny = domain->global_ny;
    
#ifdef USE_MPI
    MPI_Comm petsc_comm = domain->cart_comm;
#else
    MPI_Comm petsc_comm = PETSC_COMM_WORLD;
#endif
    
    ierr = DMDACreate2d(petsc_comm,
                       bx, by,
                       DMDA_STENCIL_STAR,
                       grid_nx, grid_ny,
                       domain->dims[0], domain->dims[1],
                       1, 1,
                       NULL, NULL,
                       &da); CHKERRQ(ierr);
    
    ierr = DMSetFromOptions(da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);
    
    // ============ Update MPIDomain to match DMDA ============
    PetscInt xs, ys, xm, ym;
    DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL);
    
#ifdef USE_MPI
    update_domain_from_dmda(domain, xs, ys, xm, ym);
#else
    // Sequential mode - just update the values directly
    domain->start_x = xs;
    domain->start_y = ys;
    domain->nx = xm;
    domain->ny = ym;
    domain->tnx = xm + 2 * domain->ghost_width;
    domain->tny = ym + 2 * domain->ghost_width;
#endif
    printf("rank : %d, startx : %d, starty : %d, nx : %d, ny : %d, xs : %d, ys : %d, xm : %d, ym : %d\n",
           mpi_rank(), domain->start_x, domain->start_y, domain->nx, domain->ny,
           (int)xs, (int)ys, (int)xm, (int)ym);
    data->da = da;

    /* Create vectors from DMDA */
    ierr = DMCreateGlobalVector(da, &(data->b)); CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(da, &(data->x)); CHKERRQ(ierr);

    /* Create matrix from DMDA */
    ierr = DMCreateMatrix(da, &(data->A)); CHKERRQ(ierr);
    
    /* Assemble Laplacian matrix */
    computeLaplacianMatrix_DMDA(data->A, da, mode);
    
    ierr = MatAssemblyBegin(data->A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(data->A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    /* For periodic: set null space */
    if (mode == M_PERIODIC) {
        MatNullSpace nullspace;
        ierr = MatNullSpaceCreate(petsc_comm, PETSC_TRUE, 0, NULL, &nullspace); CHKERRQ(ierr);
        ierr = MatSetNullSpace(data->A, nullspace); CHKERRQ(ierr);
        ierr = MatNullSpaceDestroy(&nullspace); CHKERRQ(ierr);
    }

    /* Create Krylov solver */
    ierr = KSPCreate(petsc_comm, &(data->sles)); CHKERRQ(ierr);
    ierr = KSPSetOperators(data->sles, data->A, data->A); CHKERRQ(ierr);
    ierr = KSPSetType(data->sles, KSPCG); CHKERRQ(ierr);
    
    /* Set preconditioner */
    PC prec;
    ierr = KSPGetPC(data->sles, &prec); CHKERRQ(ierr);
    
    int size;
    MPI_Comm_size(petsc_comm, &size);
    
    if (size == 1) {
        ierr = PCSetType(prec, PCLU); CHKERRQ(ierr);
    } else {
        ierr = PCSetType(prec, PCGAMG); CHKERRQ(ierr);
        ierr = PCGAMGSetNSmooths(prec, 1); CHKERRQ(ierr);
        ierr = PCGAMGSetType(prec, PCGAMGAGG); CHKERRQ(ierr);
        ierr = PCGAMGSetNlevels(prec, 4); CHKERRQ(ierr);
        ierr = PCGAMGSetAggressiveLevels(prec, 1); CHKERRQ(ierr);
        ierr = PCGAMGSetThresholdScale(prec, 1.0); CHKERRQ(ierr);
        PetscReal thr = 0.04;
        ierr = PCGAMGSetThreshold(prec, &thr, 1); CHKERRQ(ierr);
    }
    
    ierr = KSPSetTolerances(data->sles, 1e-8, 1e-8, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRQ(ierr);
    ierr = KSPSetReusePreconditioner(data->sles, PETSC_TRUE); CHKERRQ(ierr);
    ierr = KSPSetUseFischerGuess(data->sles, 1, 4); CHKERRQ(ierr);
    ierr = KSPSetFromOptions(data->sles); CHKERRQ(ierr);

    PetscPrintf(petsc_comm, "Assembly of Matrix and Vectors is done\n");

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