#pragma once

#include "mesh.h"
#include <petscksp.h>

typedef struct {
    Mat A;
    Vec b;
    Vec x;
    KSP sles;
    int mode;
} Poisson_data;

void poisson_solver(Poisson_data *data, ScalarField *i, ScalarField *o);
PetscErrorCode initialize_poisson_solver(Poisson_data *data, ScalarField* o, int mode);
void free_poisson_solver(Poisson_data *data);
