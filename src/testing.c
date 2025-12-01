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

    // Fill pressure at cell centers
    for (int x = 0; x < nx; ++x)
        for (int y = 0; y < ny; ++y)
            set_scal(&P, x, y, sin_cos(x * h, y * h));

    grad_field(&num, &P, h, set_scal);

    double erru_rms = 0.0, errv_rms = 0.0;
    double maxu = 0.0, maxv = 0.0;
    int cntu = 0, cntv = 0;

    // ---- u-component: P_x forward diff ----
    // u is defined at (x+0.5, y)
    for (int x = 1; x < nx - 2; ++x)
        for (int y = 1; y < ny - 1; ++y) {
            double xc = (x + 0.5) * h;
            double yc = y * h;

            double ref = cos(xc) * cos(yc);
            double numu = get_scal(&num.u, x, y);
            double err = numu - ref;

            erru_rms += err * err;
            if (fabs(err) > maxu) maxu = fabs(err);
            cntu++;
        }

    // ---- v-component: P_y forward diff ----
    // v is defined at (x, y+0.5)
    for (int x = 1; x < nx - 1; ++x)
        for (int y = 1; y < ny - 2; ++y) {
            double xc = x * h;
            double yc = (y + 0.5) * h;

            double ref = -sin(xc) * sin(yc);
            double numv = get_scal(&num.v, x, y);
            double err = numv - ref;

            errv_rms += err * err;
            if (fabs(err) > maxv) maxv = fabs(err);
            cntv++;
        }

    mprintf("Pressure -> erru : (%.3e, %.3e); errv : (%.3e, %.3e)\n",
            sqrt(erru_rms / cntu), maxu,
            sqrt(errv_rms / cntv), maxv);

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

    printf("domain : tnx=%d tny=%d nx=%d ny=%d gw=%d start_x=%d start_y=%d\n", 
        domain.tnx, domain.tny, domain.nx, domain.ny, domain.ghost_width, domain.start_x, domain.start_y);
    Poisson_data pdata;
    init_poisson_solver(&domain, &pdata, h, M_PERIODIC);
     printf("domain : tnx=%d tny=%d nx=%d ny=%d gw=%d start_x=%d start_y=%d\n", 
        domain.tnx, domain.tny, domain.nx, domain.ny, domain.ghost_width, domain.start_x, domain.start_y);
        
    // Each process allocates its local portion with ghost cells
    ScalarField phi = allocate_field(domain.tnx, domain.tny);
    ScalarField num_error = allocate_field(domain.tnx, domain.tny);

    double local_rms_err = 0.0;
    double local_max_err = 0.0;
    int local_count = 0;

    // Reduce errors across all processes
    double global_rms_err = 0.0;
    double global_max_err = 0.0;
    int global_count = 0;

    // Set source term: f = -2π² sin(πx) sin(πy) in interior cells
    int gw = domain.ghost_width;
    for (int x = 0; x < domain.nx; ++x) {
        for (int y = 0; y < domain.ny; ++y) {
            double xval = (domain.start_x + x) * h;
            double yval = (domain.start_y + y) * h;
            
            double val = -2.0 * M_PI * M_PI * sin(M_PI * xval) * sin(M_PI * yval);
            set_scal(&phi, x + gw, y + gw, val);
        }
    }
    
    // Synchronize ghost cells (enforces periodic BC via MPI exchange)
    synchronize_field(&phi, &domain);

    for (int x = 0; x < domain.tnx; ++x) {
        for (int y = 0; y < domain.tny; ++y) {
            double xval = (domain.start_x + x - gw) * h;
            double yval = (domain.start_y + y - gw) * h;
            
            double val = -2.0 * M_PI * M_PI * sin(M_PI * xval) * sin(M_PI * yval);
            double err = val - get_scal(&phi, x, y);
            
            set_scal(&num_error, x, y, err);
            
            local_rms_err += err * err;
            if (fabs(err) > local_max_err) local_max_err = fabs(err);
            local_count++;
        }
    }

    #ifdef USE_MPI
    MPI_Allreduce(&local_rms_err, &global_rms_err, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_max_err, &global_max_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#else
    global_rms_err = local_rms_err;
    global_max_err = local_max_err;
    global_count = local_count;
#endif

    mprintf("Synchronize Test -> RMS error: %.3e, Max error: %.3e\n", sqrt(global_rms_err / global_count), global_max_err);


    // Solve Poisson equation
    poisson_solver(&pdata, &phi, &phi);
    synchronize_field(&phi, &domain);
    
    local_rms_err = 0.0;
    local_max_err = 0.0;
    local_count = 0;

    // Reduce errors across all processes
    global_rms_err = 0.0;
    global_max_err = 0.0;
    global_count = 0;

    // Check all cells for error computation
    for (int x = gw; x < domain.tnx; ++x) {
        for (int y = gw; y < domain.tny; ++y) {

            double xval = (domain.start_x + (x - gw)) * h;
            double yval = (domain.start_y + (y - gw)) * h;
            
            double analytical = sin(M_PI * xval) * sin(M_PI * yval);
            double numerical = get_scal(&phi, x, y);
            double err = numerical - analytical;
            
            set_scal(&num_error, x, y, err);
            
            local_rms_err += err * err;
            if (fabs(err) > local_max_err) local_max_err = fabs(err);
            local_count++;
        }
    }
    
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

#ifdef USE_MPI
/*
 * MPI-IO Diagnostic Test
 * Compile: mpicc -o mpi_io_test mpi_io_test.c
 * Run: mpirun -np 4 ./mpi_io_test
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

void test_basic_collectives(int rank, int size) {
    printf("Rank %d: Testing basic collectives...\n", rank);
    fflush(stdout);
    
    // Test 1: Barrier
    printf("Rank %d: Before barrier\n", rank);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Rank %d: After barrier\n", rank);
    fflush(stdout);
    
    // Test 2: Broadcast
    int data = (rank == 0) ? 42 : 0;
    printf("Rank %d: Before bcast (data=%d)\n", rank, data);
    fflush(stdout);
    MPI_Bcast(&data, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printf("Rank %d: After bcast (data=%d)\n", rank, data);
    fflush(stdout);
    
    // Test 3: Allreduce
    int local_val = rank + 1;
    int global_sum = 0;
    printf("Rank %d: Before allreduce (local=%d)\n", rank, local_val);
    fflush(stdout);
    MPI_Allreduce(&local_val, &global_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    printf("Rank %d: After allreduce (sum=%d)\n", rank, global_sum);
    fflush(stdout);
    
    if (rank == 0) printf("\n=== Basic collectives: PASSED ===\n\n");
}

void test_mpi_file_open(int rank, int size) {
    printf("Rank %d: Testing MPI_File_open...\n", rank);
    fflush(stdout);
    
    MPI_File fh;
    const char *filename = "test_mpi_io.dat";
    
    // Add explicit barrier before file open
    printf("Rank %d: Barrier before MPI_File_open\n", rank);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    
    printf("Rank %d: Calling MPI_File_open...\n", rank);
    fflush(stdout);
    
    double t_start = MPI_Wtime();
    int ret = MPI_File_open(MPI_COMM_WORLD, filename,
                           MPI_MODE_CREATE | MPI_MODE_WRONLY,
                           MPI_INFO_NULL, &fh);
    double t_elapsed = MPI_Wtime() - t_start;
    
    if (ret != MPI_SUCCESS) {
        char error_string[MPI_MAX_ERROR_STRING];
        int length;
        MPI_Error_string(ret, error_string, &length);
        fprintf(stderr, "Rank %d: MPI_File_open FAILED after %.6f s: %s\n", 
                rank, t_elapsed, error_string);
        fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    printf("Rank %d: MPI_File_open SUCCESS after %.6f s\n", rank, t_elapsed);
    fflush(stdout);
    
    MPI_File_close(&fh);
    printf("Rank %d: File closed\n", rank);
    fflush(stdout);
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("\n=== MPI_File_open: PASSED ===\n\n");
        remove(filename);
    }
}

void test_mpi_file_write_all(int rank, int size) {
    printf("Rank %d: Testing MPI_File_write_all...\n", rank);
    fflush(stdout);
    
    MPI_File fh;
    MPI_Status status;
    const char *filename = "test_write_all.dat";
    
    // Open file
    MPI_Barrier(MPI_COMM_WORLD);
    int ret = MPI_File_open(MPI_COMM_WORLD, filename,
                           MPI_MODE_CREATE | MPI_MODE_WRONLY,
                           MPI_INFO_NULL, &fh);
    if (ret != MPI_SUCCESS) {
        fprintf(stderr, "Rank %d: File open failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    printf("Rank %d: File opened\n", rank);
    fflush(stdout);
    
    // Each rank writes its rank number
    int data = rank;
    MPI_Offset offset = rank * sizeof(int);
    
    printf("Rank %d: Setting file view at offset %lld\n", rank, (long long)offset);
    fflush(stdout);
    MPI_File_set_view(fh, offset, MPI_INT, MPI_INT, "native", MPI_INFO_NULL);
    
    printf("Rank %d: Calling MPI_File_write_all...\n", rank);
    fflush(stdout);
    double t_start = MPI_Wtime();
    MPI_File_write_all(fh, &data, 1, MPI_INT, &status);
    double t_elapsed = MPI_Wtime() - t_start;
    
    printf("Rank %d: Write completed after %.6f s\n", rank, t_elapsed);
    fflush(stdout);
    
    MPI_File_close(&fh);
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("\n=== MPI_File_write_all: PASSED ===\n\n");
        remove(filename);
    }
}

void test_filesystem_info(int rank) {
    if (rank != 0) return;
    
    printf("=== Filesystem Information ===\n");
    
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        printf("Current directory: %s\n", cwd);
    }
    
    // Try to get filesystem info via MPI
    MPI_Info info;
    MPI_Info_create(&info);
    
    const char *test_file = "test_fs_info.dat";
    MPI_File fh;
    int ret = MPI_File_open(MPI_COMM_SELF, test_file,
                           MPI_MODE_CREATE | MPI_MODE_WRONLY,
                           MPI_INFO_NULL, &fh);
    
    if (ret == MPI_SUCCESS) {
        MPI_File_get_info(fh, &info);
        
        int nkeys;
        MPI_Info_get_nkeys(info, &nkeys);
        printf("MPI-IO Info hints (%d keys):\n", nkeys);
        
        for (int i = 0; i < nkeys; i++) {
            char key[MPI_MAX_INFO_KEY];
            char value[MPI_MAX_INFO_VAL];
            int flag;
            MPI_Info_get_nthkey(info, i, key);
            MPI_Info_get(info, key, MPI_MAX_INFO_VAL, value, &flag);
            if (flag) {
                printf("  %s = %s\n", key, value);
            }
        }
        
        MPI_File_close(&fh);
        remove(test_file);
    } else {
        printf("Could not open test file for info query\n");
    }
    
    MPI_Info_free(&info);
    printf("\n");
}

void test_with_timeout(int rank, int size) {
    printf("Rank %d: Testing with explicit timeout mechanism...\n", rank);
    fflush(stdout);
    
    const char *filename = "test_timeout.dat";
    
    // Set an alarm (Unix-specific)
    printf("Rank %d: Setting 5-second timeout\n", rank);
    fflush(stdout);
    alarm(5);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    MPI_File fh;
    printf("Rank %d: Opening file (will timeout if hung)...\n", rank);
    fflush(stdout);
    
    int ret = MPI_File_open(MPI_COMM_WORLD, filename,
                           MPI_MODE_CREATE | MPI_MODE_WRONLY,
                           MPI_INFO_NULL, &fh);
    
    alarm(0); // Cancel alarm
    
    if (ret == MPI_SUCCESS) {
        printf("Rank %d: File open succeeded\n", rank);
        MPI_File_close(&fh);
    } else {
        printf("Rank %d: File open failed (but didn't hang!)\n", rank);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("\n=== Timeout test: COMPLETED ===\n\n");
        remove(filename);
    }
}

#endif