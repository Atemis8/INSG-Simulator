#pragma once
#include "mesh.h"


#define MEM_SIZE 200
struct FishData;
typedef struct FishController {
    double speed_buffer[MEM_SIZE];
    double avg_speed;
    double target_speed;
    double integral;
    double period;

    int buf_idx;
    void (*up) (struct FishData*);
} FishController;

typedef struct FishData {
    double yfish;
    double xfish;
    double ufish;
    double vfish;
    double Lfish;
    
    int nx;
    int ny;
    double L;
    double H;
    double h;
    double dt;
    double area;
    double xforce;
    double yforce;
    VectorField mask;
    FishController *cont;
} FishData;

void no_control(FishData* dat);
void pid_control(FishData* dat);

FishController* create_controller(double target, double start, void (*up) (FishData*));
FishData initialize_body(double L, double H, double h, double dt, double Lfish, int nx, int ny,int mode);
void compute_forces(FishData* data, VectorField* integ, int mode);
void compute_speed_mask(FishData* fish, VectorField* out, double time);
void compute_vorticity_mask(FishData* fish, ScalarField* vort_mask, double time);

void free_body(FishData* fish);



