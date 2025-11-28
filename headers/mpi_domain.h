#pragma once

#ifdef USE_MPI
#include <mpi.h>
#endif
#include <stdbool.h>

#ifndef MPI_PROC_NULL
#define MPI_PROC_NULL -2
#endif

#include "../headers/mesh.h"

typedef struct {

#ifdef USE_MPI
    MPI_Comm cart_comm;     // Cartesian communicator
    MPI_Datatype x_slice;   // For north-south communication
    MPI_Datatype y_slice;   // For east-west communication
    MPI_Datatype corner;    // For coner communications
#endif    
    int rank;                // Process rank in cart_comm
    int size;                // Total number of processes
    int dims[2];             // Grid dimensions [nx_procs, ny_procs]
    int coords[2];           // Process coordinates in grid [ix, iy]
    
    // Neighbor ranks (MPI_PROC_NULL if no neighbor)
    int north, south, east, west;
    int nw, sw, ne, se;
    
    // Global domain info
    int global_nx, global_ny;
    
    // Local domain info (including ghost cells)
    int nx, ny; // Size without counting ghost points
    int tnx, tny; // Total size including ghost points
    int ghost_width;
    
    // Starting indices in global domain
    int start_x, start_y;
} MPIDomain;


// All finite difference operators need synchronization before being called
int mpi_rank();
void init_mpi(int argc, char *argv[]);
void synchronize_cells(double *field, MPIDomain *domain);
void synchronize_field(ScalarField *field, MPIDomain *domain);
void synchronize_vecfield(VectorField *field, MPIDomain *domain);
void print_field(double *field, MPIDomain *domain, const char *label);
MPIDomain init_domain(int global_nx, int global_ny, int gw);

