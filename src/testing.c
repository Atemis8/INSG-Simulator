#include "../headers/poisson.h"
#include "../headers/mesh.h"
#include "../headers/utils.h"
#include "../headers/config.h"
#include "../headers/finite_diff.h"
#include "../headers/penalization.h"
#include "../headers/testing.h"
#include "../headers/mpi_domain.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include <sys/types.h>
#include <stdint.h>
#include <string.h>
#include <sys/stat.h>
#include <string.h>
#include <dirent.h>
#include <sys/types.h>

double cosx_cosy(double x, double y) {
    return -2 * cos(x) * cos(y);
}

double sin_cos(double x, double y) {
    return sin(x) * cos(y);
}

double cos_sin(double x, double y) {
    return cos(x) * sin(y);
}

void test_grad_term(double Lx, double Ly, double h) {
    if (fmod(Lx, h) != 0 || fmod(Ly, h) != 0) assert(0);
    int nx = Lx / h;
    int ny = Ly / h;

    ScalarField P = allocate_field(nx, ny);
    VectorField num = allocate_vecfield(nx, ny);

    for(int x = 0; x < nx; ++x)
        for(int y = 0; y < ny; ++y) {
            set_scal(&P, x, y, sin_cos(x * h, y * h));
        }
    grad_field(&num, &P, h, set_scal);

    double erru_rms = 0.0;
    double errv_rms = 0.0;
    double maxu = 0.0;
    double maxv = 0.0;
    for(int x = 0; x < nx-1; ++x) 
        for(int y = 0; y < ny-1; ++y) {
            double erru = get_scal(&num.u, x, y) - cos((x + 0.5) * h) * cos(y * h);
            if(maxu < fabs(erru)) maxu = fabs(erru);
            erru_rms += erru * erru;

            double errv = get_scal(&num.v, x, y) + sin(x * h) * sin((y + 0.5) * h);
            if(maxv < fabs(errv)) maxv = fabs(errv);
            errv_rms += errv * errv;
        }

    mprintf("Pressure -> erru : (%.3e, %.3e); errv : (%.3e, %.3e)\n", 
        sqrt(erru_rms) / ((nx - 1) * (ny - 1)), maxu, 
        sqrt(errv_rms) / ((nx - 1) * (ny - 1)), maxv);
    
    free_field(&P);
    free_vecfield(&num);
}

void test_viscosity(double Lx, double Ly, double h) {
    if (fmod(Lx, h) != 0 || fmod(Ly, h) != 0) assert(0);
    int nx = Lx / h;
    int ny = Ly / h;
    MACMesh mesh = allocate_mesh(nx, ny, h);
    mesh.uv.u.type = -1;
    mesh.uv.v.type = -1;
    for(int x = 0; x < nx; ++x) 
        for(int y = 0; y < ny; ++y) {
            set_scal(&mesh.uv.u, x, y, sin_cos((x + 0.5) * h, y * h));
            set_scal(&mesh.uv.v, x, y, cos_sin(x * h, (y + 0.5) * h));
        }
    mesh.uv.u.type = 0;
    mesh.uv.v.type = 1;
    VectorField num = vecfield_like(&mesh.uv);
    viscosity_term(&mesh, &num, set_scal);

    double erru_rms = 0.0;
    double errv_rms = 0.0;
    double maxu = 0.0;
    double maxv = 0.0;
    for(int x = 1; x < nx - 2; ++x) 
        for(int y = 1; y < ny - 2; ++y) {
            double erru = get_scal(&num.u, x, y) + 2 * sin_cos((x + 0.5) * h, y * h);
            if(maxu < fabs(erru)) maxu = fabs(erru);
            erru_rms += erru * erru;

            double errv = get_scal(&num.v, x, y) + 2 * cos_sin(x * h, (y + 0.5) * h);
            if(maxv < fabs(errv)) maxv = fabs(errv);
            errv_rms += errv * errv;
        }
    mprintf("Viscosity -> erru : (%.3e, %.3e), errv : (%.3e, %.3e)\n", 
        sqrt(erru_rms) / ((nx - 2) * (ny - 2)), maxu,
        sqrt(errv_rms) / ((nx - 2) * (ny - 2)), maxv);

    free_mesh(&mesh);
    free_vecfield(&num);
}

void test_convective(double Lx, double Ly, double h) {
    if (fmod(Lx, h) != 0 || fmod(Ly, h) != 0) assert(0);
    int nx = Lx / h;
    int ny = Ly / h;
    MACMesh mesh = allocate_mesh(nx, ny, h);
    mesh.uv.u.type = -1;
    mesh.uv.v.type = -1;
    for(int x = 0; x < nx; ++x) 
        for(int y = 0; y < ny; ++y) {
            set_scal(&mesh.uv.u, x, y, sin_cos((x + 0.5) * h, y * h));
            set_scal(&mesh.uv.v, x, y, -cos_sin(x * h, (y + 0.5) * h));
        }
    
    mesh.uv.u.type = 0;
    mesh.uv.v.type = 1;
    VectorField num = vecfield_like(&mesh.uv);

    divergence_form(&mesh, &num, set_scal);
    double erru_rms = 0.0;
    double errv_rms = 0.0;
    double maxu = 0.0;
    double maxv = 0.0;
    for(int x = 1; x < nx - 2; ++x) 
        for(int y = 1; y < ny - 2; ++y) {
            double erru = get_scal(&num.u, x, y) - (sin(2 * (x + 0.5) * h) * (cos(y * h) * cos(y * h) - 0.5 * cos(2 * y * h)));
            if(maxu < fabs(erru)) maxu = fabs(erru);
            erru_rms += erru * erru;

            double errv = get_scal(&num.v, x, y) - (-cos((y + 0.5) * h) * sin((y + 0.5) * h) * cos(2 * x * h)
            + cos(x * h) * cos(x * h) * sin(2 * (y + 0.5) * h));
            if(maxv < fabs(errv)) maxv = fabs(errv);
            errv_rms += errv * errv;
        }

    mprintf("Convective -> erru : (%.3e, %.3e), errv : (%.3e, %.3e)\n", 
        sqrt(erru_rms) / ((nx - 1) * (ny - 1)), maxu,
        sqrt(errv_rms) / ((nx - 1) * (ny - 1)), maxv);

    free_mesh(&mesh);
    free_vecfield(&num);
}

void test_divergence(double Lx, double Ly, double h) {
    if (fmod(Lx, h) != 0 || fmod(Ly, h) != 0) assert(0);
    int nx = Lx / h;
    int ny = Ly / h;
    VectorField v = allocate_vecfield(nx, ny);
    for(int x = 0; x < nx; ++x) 
        for(int y = 0; y < ny; ++y) {
            set_scal(&v.u, x, y, sin_cos((x + 0.5) * h, y * h));
            set_scal(&v.v, x, y, cos_sin(x * h, (y + 0.5) * h));
        }

    ScalarField f = allocate_field(nx, ny);
    divergence(&f, &v, h, set_scal);

    double err_rms = 0.0;
    double maxerr = 0.0;
    for(int x = 1; x < nx-1; ++x) 
        for(int y = 1; y < ny-1; ++y) {
            double err = get_scal(&f, x, y) - 2 * cos(x * h) * cos(y * h);
            if(maxerr < fabs(err)) maxerr = fabs(err);
            err_rms += err * err;
        }

    mprintf("Divergence -> err : (%.3e, %.3e)\n", 
        sqrt(err_rms / ((nx - 1) * (ny - 1))), maxerr);

    free_field(&f);
    free_vecfield(&v);

}

void test_vectorfield_integration(double Lx, double Ly, double h) {
    int nx = Lx / h;
    int ny = Ly / h;
    MACMesh mesh = allocate_mesh(nx, ny, h);

    nx += 2; ny += 2;

    for (int x = 0; x < nx; ++x)
        for (int y = 0; y < ny; ++y) {
            set_scal(&mesh.uv.u, x, y, (x * h) * (x * h) + (y * h) * (y * h));
            set_scal(&mesh.uv.v, x, y, 2 * (x * h) * (y * h));
        }

    // Fill the points that shouldnt be used with garbadge
    for (int x = 0; x < nx-1; ++x) set_scal(&mesh.uv.v, x, ny-1, 10000.0);
    for (int y = 0; y < ny-1; ++y) set_scal(&mesh.uv.u, nx-1, y, 10000.0);

    double Iu = 0.0;
    for (int x = 0; x < nx - 2; ++x)
        for (int y = 1; y < ny - 1; ++y) {
            double u = (get_scal(&mesh.uv.u, x, y) + get_scal(&mesh.uv.u, x + 1, y)) / 2.0;
            Iu += u * h * h;
        }

    double Iv = 0.0;
    for (int x = 1; x < nx - 1; ++x)
        for (int y = 0; y < ny - 2; ++y) {
            double v = (get_scal(&mesh.uv.v, x, y) + get_scal(&mesh.uv.v, x, y + 1)) / 2.0;
            Iv += v * h * h;
        }

    double exact_u = (pow(Lx, 3) / 3) * Ly + Lx * (pow(Ly, 3) / 3);
    double exact_v = (pow(Lx, 2) * pow(Ly, 2)) / 2;

    double erru = fabs(Iu - exact_u);
    double errv = fabs(Iv - exact_v);

    mprintf("Integration -> uerr : %.3e, verr : %.3e\n", erru, errv);

    free_mesh(&mesh);
}


void test_poisson_solver(int N) {
    double h = 2.0 / N;
    
    // Initialize MPI domain with periodic boundaries
    MPIDomain domain = init_domain(N, N, 1, M_PERIODIC);  // 1 = periodic
    
    // Each process allocates its local portion with ghost cells
    ScalarField phi = allocate_field(domain.tnx, domain.tny);

    Poisson_data pdata;
    init_poisson_solver(&domain, &pdata, &phi, M_PERIODIC);

    // Set source term: f = -2π² sin(πx) sin(πy) in interior cells
    int gw = domain.ghost_width;
    for (int x = 0; x < domain.nx; ++x) {
        for (int y = 0; y < domain.ny; ++y) {
            int xval = (domain.start_x + x) * h;
            int yval = (domain.start_y + y) * h;
            
            double val = -2.0 * M_PI * M_PI * sin(M_PI * xval) * sin(M_PI * yval);
            set_scal(&phi, x + gw, y + gw, val);
        }
    }
    
    // Synchronize ghost cells (enforces periodic BC via MPI exchange)
    synchronize_field(&phi, &domain);

    // Solve Poisson equation
    poisson_solver(&pdata, &phi, &phi);
    op_field(&phi, h * h, mul_scal);
    
    // Synchronize solution ghost cells
    synchronize_field(&phi, &domain);

    // Compute error (check all cells including ghosts for periodic consistency)
    ScalarField num_error = allocate_field(domain.tnx, domain.tny);
    
    double local_rms_err = 0.0;
    double local_max_err = 0.0;
    int local_count = 0;
    
    // Check all cells for error computation
    for (int x = 0; x < domain.tnx; ++x) {
        for (int y = 0; y < domain.tny; ++y) {
            int xval = (domain.start_x + (x - gw)) * h;
            int yval = (domain.start_y + (y - gw)) * h;
            
            double analytical = sin(M_PI * xval) * sin(M_PI * yval);
            double numerical = get_scal(&phi, x, y);
            double err = numerical - analytical;
            
            set_scal(&num_error, x, y, err);
            
            local_rms_err += err * err;
            if (fabs(err) > local_max_err) local_max_err = fabs(err);
            local_count++;
        }
    }
    
    // Reduce errors across all processes
    double global_rms_err = 0.0;
    double global_max_err = 0.0;
    int global_count = 0;
    
#ifdef USE_MPI
    MPI_Allreduce(&local_rms_err, &global_rms_err, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_max_err, &global_max_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#else
    global_rms_err = local_rms_err;
    global_max_err = local_max_err;
    global_count = local_count;
#endif

    mprintf("Poisson solver -> RMS error: %.3e, Max error: %.3e\n", sqrt(global_rms_err / global_count), global_max_err);

    free_field(&num_error);
    free_field(&phi);
    free_poisson_solver(&pdata);
}

void test_mpidomain(int global_nx, int global_ny) {
    MPIDomain domain = init_domain(global_nx, global_ny, 1, M_PERIODIC);
    
    printf("rank : %d\n", domain.rank);
    printf("No domain : %d \n", MPI_PROC_NULL);
    printf("North : %d \n", domain.north);
    printf("East : %d \n", domain.east);
    printf("South : %d \n", domain.south);
    printf("West : %d \n", domain.west);
#ifdef USE_MPI
    if (domain.rank == 0) {
        printf("\n========================================\n");
        printf("MPI Ghost Cell Exchange Test\n");
        printf("Global grid: %dx%d\n", global_nx, global_ny);
        printf("Process grid: %dx%d\n", domain.dims[0], domain.dims[1]);
        printf("========================================\n");
    }
    
    // Allocate field
    int total_size = domain.tnx * domain.tny;
    double *field = (double*)malloc(total_size * sizeof(double));
    // Initialize: all cells start as -1 (including ghosts and interior)
    for (int i = 0; i < total_size; i++) field[i] = -1.0;
    
    // Fill interior cells with global coordinate values: global_x * 10 + global_y
    // This makes it easy to see which cell came from where
    int gw = domain.ghost_width;
    for (int x = gw; x < gw + domain.nx; x++) {
        for (int y = gw; y < gw + domain.ny; y++) {
            // Calculate global coordinates
            int global_x = (x - gw) + domain.start_x;
            int global_y = (y - gw) + domain.start_y;
            field[xytok(x, y, domain.tny)] = global_x * 10.0 + global_y;
        }
    }

    // Print BEFORE synchronization
    
    MPI_Barrier(MPI_COMM_WORLD);
    for (int r = 0; r < domain.size; r++) {
        if (r == domain.rank) print_field(field, &domain, "BEFORE GHOST EXCHANGE");
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    // Synchronize ghost cells
    synchronize_cells(field, &domain);
    
    // Print AFTER synchronization
    MPI_Barrier(MPI_COMM_WORLD);
    if (domain.rank == 0) {
        printf("\n\n========================================\n");
        printf("AFTER GHOST EXCHANGE\n");
        printf("========================================\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    for (int r = 0; r < domain.size; r++) {
        if (r == domain.rank) print_field(field, &domain, "AFTER GHOST EXCHANGE");

        MPI_Barrier(MPI_COMM_WORLD);
    }
#endif
}