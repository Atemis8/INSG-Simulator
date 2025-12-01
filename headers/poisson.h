#pragma once

#include "mesh.h"
#include "mpi_domain.h"
#include <petscksp.h>
#include <petscdmda.h>  // Add this for DMDA support

typedef struct {
    MPIDomain *domain;
    Mat A;
    Vec b;
    Vec x;
    KSP sles;
    DM da;      // Add this for DMDA
    int mode;
} Poisson_data;

void poisson_solver(Poisson_data *data, ScalarField *i, ScalarField *o);
PetscErrorCode init_poisson_solver(MPIDomain *domain, Poisson_data *data, double h, Mode mode);
void free_poisson_solver(Poisson_data *data);