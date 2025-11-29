#pragma once
#include "../headers/poisson.h"
#include "../headers/penalization.h"
#include "../headers/mpi_domain.h"
/*
Config file containing the structure for the simulation paramters and flag
*/

#include <math.h>

typedef struct
{
    double dt;
    double CFL; //pas sur a voir de comment on definit u_mesh
    double dTau;

} DataSim;

typedef struct SimulationParams {
    MPIDomain domain;
    int num_epsiodes; // Number of simulation step
    int dump_period; // Period at which dumps are created
    int mode; // Either PERIODIC or BOUNDARY
    double Re; // Reynolds number
    double nu; // Viscosity
    double dt; // Time step 
    double dtau; // Penalization step
    double h; // Spacial discretization
    FishData body;
} SimulationParams;

typedef struct Simulation {
    SimulationParams *params;
    MACMesh mesh;
    VectorField Hn;
    VectorField Hnm1;
    VectorField vstar;
    VectorField vsn1;
    VectorField buffer;
    ScalarField phi;
    Poisson_data pdata;
    double t;
    int mode;
} Simulation;

void init_simulation(Simulation *sim, SimulationParams *params, int argc, char *argv[]);
void default_params(SimulationParams *params, double N, double Re, double Lstar, double Hstar, Mode mode);
void free_simulation(Simulation* sim);