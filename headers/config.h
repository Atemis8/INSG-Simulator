#pragma once
#include "../headers/poisson.h"
#include "../headers/penalization.h"
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

enum Mode {
    M_PERIODIC,
    M_BOUNDARY
};

typedef struct SimulationParams {
    int ghost_nb;
    int num_epsiodes;
    int dump_period;
    int mode;
    double Re;
    double nu;
    double dt;
    double dtau;
    double h;
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

Simulation init_simulation(SimulationParams *params);
SimulationParams default_params(double N, double Re, double Lstar, double Hstar,int mode);
void free_simulation(Simulation* sim);