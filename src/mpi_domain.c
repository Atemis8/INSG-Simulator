#include "../headers/mpi_domain.h"
#include "../headers/utils.h"

int rank = 0;
int size = 1;
int mpi_rank() {return rank;}
int mpi_size() {return size;}
// MPI Helper functions
#ifdef USE_MPI

void init_mpi(int argc, char *argv[]) {
    int mpi_initialized;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
}

int MPI_Cart_shift_nd(
    MPI_Comm cart_comm,
    const int *offset,   // e.g. {-1, +1} or {+1, -1, +1}
    int ndims,
    int *source,
    int *dest
) {
    int *dims    = malloc(ndims * sizeof(int));
    int *periods = malloc(ndims * sizeof(int));
    int *coords  = malloc(ndims * sizeof(int));
    int *csrc    = malloc(ndims * sizeof(int));
    int *cdst    = malloc(ndims * sizeof(int));

    // Query topology information
    MPI_Cart_get(cart_comm, ndims, dims, periods, coords);

    // ----- Compute source = coords - offset -----
    for (int d = 0; d < ndims; d++) {
        csrc[d] = coords[d] - offset[d];

        if (csrc[d] < 0 || csrc[d] >= dims[d]) {
            if (periods[d]) csrc[d] = (csrc[d] + dims[d]) % dims[d];
            else {
                *source = MPI_PROC_NULL;
                goto compute_dest;
            }
        }
    }

    MPI_Cart_rank(cart_comm, csrc, source);

compute_dest:

    // ----- Compute dest = coords + offset -----
    for (int d = 0; d < ndims; d++) {
        cdst[d] = coords[d] + offset[d];

        if (cdst[d] < 0 || cdst[d] >= dims[d]) {
            if (periods[d]) cdst[d] = (cdst[d] + dims[d]) % dims[d];
            else {
                *dest = MPI_PROC_NULL;
                goto finish;
            }
        }
    }

    MPI_Cart_rank(cart_comm, cdst, dest);

finish:
    free(dims);
    free(periods);
    free(coords);
    free(csrc);
    free(cdst);
    return MPI_SUCCESS;
}

void print_field(double *field, MPIDomain *domain, const char *label) {
    printf("\n=== Rank %d (%d,%d) - %s ===\n", 
           domain->rank, domain->coords[0], domain->coords[1], label);
    
    // Print from top to bottom (y from ny+2*gw-1 down to 0)
    for (int y = domain->tny - 1; y >= 0; y--) {
        printf("y=%2d: ", y);
        for (int x = 0; x < domain->tnx; x++) {
            double val = field[xytok(x, y, domain->tny)];
            if (val < 0) printf(" .  "); 
            else printf("%3.0f ", val);
        }
        
        // Add markers for regions
        if (y == domain->tny - 1) printf(" <- North ghost");
        else if (y == domain->ghost_width + domain->ny - 1) printf(" <- North interior boundary");
        else if (y == domain->ghost_width) printf(" <- South interior boundary");
        else if (y == 0) printf(" <- South ghost");
        printf("\n");
    }
    
    // X-axis labels
    printf("      ");
    for (int x = 0; x < domain->tnx; x++) printf("x%-2d ", x);
    printf("\n");
    
    printf("      ");
    for (int x = 0; x < domain->tnx; x++) {
        if (x == 0) printf("W   ");
        else if (x == domain->ghost_width) printf("|   ");
        else if (x == domain->ghost_width + domain->nx) printf("|   ");
        else if (x == domain->tnx - 1) printf("E   ");
        else printf("    ");
    }
    printf("\n");
    printf("      W=West ghost, E=East ghost, | = interior boundaries\n");
    fflush(stdout);
}

/*
Note : the x-direction is the horizontal axis and y direction is the vertical axis, 
which means we are using standart carthesian coordiantes. However the memory layout 
is a bit different since the index since it is #define xytok(x, y, ny) ((x) * (ny) + (y))
which means the indexing is "column" major
*/
MPIDomain init_domain(int global_nx, int global_ny, int gw, Mode mode) {
    MPIDomain domain;
    domain.ghost_width = gw;
    domain.global_nx = global_nx;
    domain.global_ny = global_ny;
    
    MPI_Comm_size(MPI_COMM_WORLD, &domain.size);
    domain.rank = mpi_rank();
    
    // Create optimal 2D decomposition
    domain.dims[0] = 0; 
    domain.dims[1] = 0;
    
    MPI_Dims_create(domain.size, 2, domain.dims);
    mprintf("MPI Grid: %d x %d processes\n", domain.dims[0], domain.dims[1]);
    
    // Create Cartesian communicator
    int periods[2] = {(mode == M_PERIODIC) ? 1 : 0, (mode == M_PERIODIC) ? 1 : 0};
    MPI_Cart_create(MPI_COMM_WORLD, 2, domain.dims, periods, 1, &domain.cart_comm);

    // Get rank and coordinates in Cartesian grid
    MPI_Comm_rank(domain.cart_comm, &domain.rank);
    MPI_Comm_size(domain.cart_comm, &domain.size);
    MPI_Cart_coords(domain.cart_comm, domain.rank, 2, domain.coords);
    
    // Find neighbor ranks
    MPI_Cart_shift(domain.cart_comm, 0, 1, &domain.west, &domain.east);
    MPI_Cart_shift(domain.cart_comm, 1, 1, &domain.south, &domain.north);

    // Find corner ranks
    int offset1[2] = {+1, -1};
    int offset2[2] = {+1, +1};
    MPI_Cart_shift_nd(domain.cart_comm, offset1, 2, &domain.nw, &domain.se);
    MPI_Cart_shift_nd(domain.cart_comm, offset2, 2, &domain.sw, &domain.ne);
    
    // Calculate local domain sizes
    int base_nx = global_nx / domain.dims[0];
    int extra_nx = global_nx % domain.dims[0];
    int base_ny = global_ny / domain.dims[1];
    int extra_ny = global_ny % domain.dims[1];
    
    // Local size (without ghost cells)
    domain.nx = base_nx + (domain.coords[0] < extra_nx ? 1 : 0);
    domain.ny = base_ny + (domain.coords[1] < extra_ny ? 1 : 0);
    
    // Calculate starting indices
    domain.start_x = domain.coords[0] * base_nx + (domain.coords[0] < extra_nx ? domain.coords[0] : extra_nx);
    domain.start_y = domain.coords[1] * base_ny + (domain.coords[1] < extra_ny ? domain.coords[1] : extra_ny);
    
    // Total size including ghost cells
    domain.tnx = domain.nx + 2 * domain.ghost_width;
    domain.tny = domain.ny + 2 * domain.ghost_width;

    // East-West exchange: vertical strips (contiguous in memory, y varies)
    // gw strips, each of height tny
    MPI_Type_contiguous(domain.tny * domain.ghost_width, MPI_DOUBLE, &domain.y_slice);
    MPI_Type_commit(&domain.y_slice);
    
    // North-South exchange: horizontal strips (non-contiguous, x varies)
    // gw elements per strip, tnx strips total, stride tny between elements
    MPI_Type_vector(domain.tnx, domain.ghost_width, domain.tny, MPI_DOUBLE, &domain.x_slice);
    MPI_Type_commit(&domain.x_slice);

    // Corner exchange, the last blocks have a size gw x gw
    MPI_Type_vector(domain.ghost_width, domain.ghost_width, domain.tny, MPI_DOUBLE, &domain.corner);
    MPI_Type_commit(&domain.corner);

    return domain;
}

void update_domain_from_dmda(MPIDomain *domain, int start_x, int start_y, int nx, int ny) {
    // Free old MPI types
    MPI_Type_free(&domain->y_slice);
    MPI_Type_free(&domain->x_slice);
    MPI_Type_free(&domain->corner);
    
    // Update decomposition
    domain->start_x = start_x;
    domain->start_y = start_y;
    domain->nx = nx;
    domain->ny = ny;
    domain->tnx = nx + 2 * domain->ghost_width;
    domain->tny = ny + 2 * domain->ghost_width;
    
    // Recreate MPI datatypes with correct sizes
    MPI_Type_contiguous(domain->tny * domain->ghost_width, MPI_DOUBLE, &domain->y_slice);
    MPI_Type_commit(&domain->y_slice);
    
    MPI_Type_vector(domain->tnx, domain->ghost_width, domain->tny, MPI_DOUBLE, &domain->x_slice);
    MPI_Type_commit(&domain->x_slice);

    MPI_Type_vector(domain->ghost_width, domain->ghost_width, domain->tny, MPI_DOUBLE, &domain->corner);
    MPI_Type_commit(&domain->corner);
}

void synchronize_cells(double *field, MPIDomain *domain) {
    int tny = domain->tny;
    int nx = domain->nx;
    int ny = domain->ny;
    int gw = domain->ghost_width;

    MPI_Request requests[8];
    MPI_Status statuses[8];
    int req_count = 0;
    
    if (domain->east != MPI_PROC_NULL) {
        MPI_Irecv(&field[xytok(nx+gw, 0, tny)], 1, domain->y_slice, domain->east, 0, domain->cart_comm, &requests[req_count++]);
        MPI_Isend(&field[xytok(nx, 0, tny)], 1, domain->y_slice, domain->east, 1, domain->cart_comm, &requests[req_count++]);
    }  
    if (domain->west != MPI_PROC_NULL) { 
        MPI_Irecv(&field[xytok(0, 0, tny)], 1, domain->y_slice, domain->west, 1, domain->cart_comm, &requests[req_count++]);
        MPI_Isend(&field[xytok(gw, 0, tny)], 1, domain->y_slice, domain->west, 0, domain->cart_comm, &requests[req_count++]);
    }
    
    if (domain->north != MPI_PROC_NULL) {
        MPI_Irecv(&field[xytok(0, ny+gw, tny)], 1, domain->x_slice, domain->north, 2, domain->cart_comm, &requests[req_count++]);
        MPI_Isend(&field[xytok(0, ny, tny)], 1, domain->x_slice, domain->north, 3, domain->cart_comm, &requests[req_count++]);
    }  
    if (domain->south != MPI_PROC_NULL) {
        MPI_Irecv(&field[xytok(0, 0, tny)], 1, domain->x_slice, domain->south, 3, domain->cart_comm, &requests[req_count++]);
        MPI_Isend(&field[xytok(0, gw, tny)], 1, domain->x_slice, domain->south, 2, domain->cart_comm, &requests[req_count++]);
    }
    
    MPI_Waitall(req_count, requests, statuses);
    
    req_count = 0;
    if (domain->ne != MPI_PROC_NULL) {
        MPI_Irecv(&field[xytok(nx+gw, ny+gw, tny)], 1, domain->corner, domain->ne, 0, domain->cart_comm, &requests[req_count++]);
        MPI_Isend(&field[xytok(nx, ny, tny)], 1, domain->corner, domain->ne, 1, domain->cart_comm, &requests[req_count++]);
    }
    
    if (domain->sw != MPI_PROC_NULL) { 
        MPI_Irecv(&field[xytok(0, 0, tny)], 1, domain->corner, domain->sw, 1, domain->cart_comm, &requests[req_count++]);
        MPI_Isend(&field[xytok(gw, gw, tny)], 1, domain->corner, domain->sw, 0, domain->cart_comm, &requests[req_count++]);
    }
    
    if (domain->nw != MPI_PROC_NULL) {
        MPI_Irecv(&field[xytok(0, ny+gw, tny)], 1, domain->corner, domain->nw, 2, domain->cart_comm, &requests[req_count++]);
        MPI_Isend(&field[xytok(gw, ny, tny)], 1, domain->corner, domain->nw, 3, domain->cart_comm, &requests[req_count++]);
    }
    
    if (domain->se != MPI_PROC_NULL) {
        MPI_Irecv(&field[xytok(nx+gw, 0, tny)], 1, domain->corner, domain->se, 3, domain->cart_comm, &requests[req_count++]);
        MPI_Isend(&field[xytok(nx, gw, tny)], 1, domain->corner, domain->se, 2, domain->cart_comm, &requests[req_count++]);
    }
    
    MPI_Waitall(req_count, requests, statuses);
}

#else

void init_mpi(int argc, char *argv[]) {}
void synchronize_cells(double *field, MPIDomain *domain) {}
void print_field(double *field, MPIDomain *domain, const char *label) {}

MPIDomain init_domain(int global_nx, int global_ny, int gw, Mode mode) { 
    MPIDomain domain;
    domain.rank = mpi_rank();
    domain.size = 1;
    domain.ghost_width = gw;

    domain.dims[0] = 1;
    domain.dims[1] = 1;

    domain.coords[0] = 0;
    domain.coords[1] = 0;
    
    domain.north = MPI_PROC_NULL;
    domain.south = MPI_PROC_NULL;
    domain.east = MPI_PROC_NULL;
    domain.west = MPI_PROC_NULL;

    domain.nw = MPI_PROC_NULL;
    domain.ne = MPI_PROC_NULL;
    domain.sw = MPI_PROC_NULL;
    domain.se = MPI_PROC_NULL;
    
    domain.global_nx = global_nx;
    domain.global_ny = global_ny;

    int base_nx = global_nx / domain.dims[0];
    int extra_nx = global_nx % domain.dims[0];
    int base_ny = global_ny / domain.dims[1];
    int extra_ny = global_ny % domain.dims[1];
    
    domain.nx = base_nx + (domain.coords[0] < extra_nx ? 1 : 0);
    domain.ny = base_ny + (domain.coords[1] < extra_ny ? 1 : 0);
    
    domain.start_x = domain.coords[0] * base_nx + (domain.coords[0] < extra_nx ? domain.coords[0] : extra_nx);
    domain.start_y = domain.coords[1] * base_ny + (domain.coords[1] < extra_ny ? domain.coords[1] : extra_ny);
    
    domain.tnx = domain.nx + 2 * domain.ghost_width;
    domain.tny = domain.ny + 2 * domain.ghost_width;

    return domain; 
}

#endif

void synchronize_field(ScalarField *field, MPIDomain *domain) {
    synchronize_cells(field->v, domain);
}

void synchronize_vecfield(VectorField *field, MPIDomain *domain) {
    synchronize_field(&field->u, domain);
    synchronize_field(&field->v, domain);
}

#ifndef USE_MPI
void update_domain_from_dmda(MPIDomain *domain, int start_x, int start_y, int nx, int ny) {}
#endif
/*

P   u   P   u   P   u   P   u   P   u

v   w---v---w---v---w---v---w   v   -
    |                       |
P   u   P   u   P   u   P   u   P   u
    |                       |   
v   w   v   w   v   w   v   w   v   -
    |                       |
P   u   P   u   P   u   P   u   P   u
    |                       |
v   w   v   w   v   w   v   w   v   -
    |                       |
P   u   P   u   P   u   P   u   P   u
    |                       |
v   w---v---w---v---w---v---w   v   -

P   u   P   u   P   u   P   u   P   u

Note that when doing a synchronization, the values on the borders are never sent
*/
