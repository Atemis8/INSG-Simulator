#pragma once
#include <math.h>
#include "mesh.h"
#include "penalization.h"
#include "config.h"

#define PI acos(-1.0)
#define mod(x, N) (x % N + N) % N
#define min(a, b) (a) < (b) ? (a) : (b)
#define max(a, b) (a) > (b) ? (a) : (b)
// #define uint unsigned int

typedef struct PostProcessor {
    const char *dir;
    ScalarField w;
    Simulation *m;
} PostProcessor;

PostProcessor initialize_postprocessor(Simulation *m, const char* dir_path);

// Save fields
SimulationParams load_params(const char *dir);
Simulation load_simulation(SimulationParams *params, const char *dir, int *ep_o);
int initialize_dump(const char* dir_path);
void mprintf(const char *fmt, ...);
void save_fieldtxt(ScalarField *f, const char* file);
void save_field(ScalarField *f, const char* file);
void dump_params(SimulationParams *params, PostProcessor *p);
void dump_mesh(int ep, PostProcessor *post);
void dump_fish_data(FishData *body, PostProcessor *p);
void dump_mesh_data(SimulationParams *sim, PostProcessor *p);
double mod_d(double x, double N);