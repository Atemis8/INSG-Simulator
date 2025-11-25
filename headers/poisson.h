#pragma once

#include "mesh.h"
#include <petscksp.h>

typedef struct {
    Mat A;
    Vec b;
    Vec x;
    KSP sles;
    int mode;
#ifdef USE_MPI
    Vec x_global;           // Global vector for gathering solution
    VecScatter scatter_ctx;  // Scatter context for MPI communication
#endif
} Poisson_data;

void poisson_solver(Poisson_data *data, ScalarField *i, ScalarField *o);
PetscErrorCode initialize_poisson_solver(Poisson_data *data, ScalarField* o, int mode);
void free_poisson_solver(Poisson_data *data);
