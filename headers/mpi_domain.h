#pragma once

#include <mpi.h>
#include <stdbool.h>

typedef struct {
    MPI_Comm cart_comm;      // Cartesian communicator
    int rank;                // Process rank in cart_comm
    int size;                // Total number of processes
    int dims[2];             // Grid dimensions [nx_procs, ny_procs]
    int coords[2];           // Process coordinates in grid [ix, iy]
    
    // Neighbor ranks (MPI_PROC_NULL if no neighbor)
    int north, south, east, west;
    
    // Global domain info
    int global_nx, global_ny;
    
    // Local domain info (including ghost cells)
    int nx, ny; // Size without counting ghost points
    int tnx, tny; // Total size including ghost points
    int ghost_width;
    
    // Starting indices in global domain
    int start_x, start_y;
    
    // MPI datatypes for communication
    MPI_Datatype x_slice;  // For north-south communication
    MPI_Datatype y_slice;  // For east-west communication
} MPIDomain;

int mpi_rank();
void synchronize_cells(double *field, MPIDomain *domain);
void print_field(double *field, MPIDomain *domain, const char *label);
MPIDomain init_mpi(int argc, char *argv[], int global_nx, int global_ny);