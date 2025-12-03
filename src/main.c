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
#include <stdbool.h>
#include <time.h>

#ifdef USE_MPI
#include <mpi.h>
#define GET_TIME() MPI_Wtime()
#else
#include <sys/time.h>
static inline double GET_TIME() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}
#endif

// Generic performance profiling structure
#define MAX_PROFILE_SECTIONS 32
#define MAX_SECTION_NAME 64

typedef struct {
    char names[MAX_PROFILE_SECTIONS][MAX_SECTION_NAME];
    double times[MAX_PROFILE_SECTIONS];
    double start_times[MAX_PROFILE_SECTIONS];
    int num_sections;
    double total;
    int count;
} PerformanceProfile;

void init_profile(PerformanceProfile *prof) {
    memset(prof, 0, sizeof(PerformanceProfile));
}

int find_or_add_section(PerformanceProfile *prof, const char *name) {
    // First, try to find existing section
    for (int i = 0; i < prof->num_sections; i++) {
        if (strcmp(prof->names[i], name) == 0) {
            return i;
        }
    }
    
    // Not found, add new section
    if (prof->num_sections >= MAX_PROFILE_SECTIONS) {
        fprintf(stderr, "Warning: Maximum number of profile sections reached\n");
        return -1;
    }
    
    int id = prof->num_sections;
    strncpy(prof->names[id], name, MAX_SECTION_NAME - 1);
    prof->names[id][MAX_SECTION_NAME - 1] = '\0';
    prof->times[id] = 0.0;
    prof->start_times[id] = 0.0;
    prof->num_sections++;
    return id;
}

void start_record(PerformanceProfile *prof, const char *name) {
    int id = find_or_add_section(prof, name);
    if (id >= 0) {
        prof->start_times[id] = GET_TIME();
    }
}

void end_record(PerformanceProfile *prof, const char *name) {
    double end_time = GET_TIME();
    int id = find_or_add_section(prof, name);
    if (id >= 0) {
        prof->times[id] += end_time - prof->start_times[id];
    }
}

void print_profile(PerformanceProfile *prof) {
    if (mpi_rank() != 0) return;
    
    double total = prof->total;
    if (total < 1e-10) return;
    
    printf("\n=== Performance Profile (averaged over %d steps) ===\n", prof->count);
    printf("Total time:        %8.3f s (100.0%%)\n", total);
    
    for (int i = 0; i < prof->num_sections; i++) {
        printf("  %-18s %8.3f s (%5.1f%%)\n", 
               prof->names[i], 
               prof->times[i], 
               100.0 * prof->times[i] / total);
    }
    
    printf("Average time/step: %8.3f ms\n", 1000.0 * total / prof->count);
    printf("=====================================================\n\n");
}

double rectangle(double x, double y) {
    return (fabs(x-PI) < 0.5 && fabs(y-PI) < 0.5) ? 1.0 : 0.0;
}

void sim_step(Simulation *sim, PerformanceProfile *prof) {
    MPIDomain *domain = &(sim->params->domain);
    MACMesh *mesh = &(sim->mesh);
    VectorField *vstar = &(sim->vstar);
    VectorField *Hnm1 = &(sim->Hnm1);
    VectorField *vsn1 = &(sim->vsn1);
    VectorField *Hn = &(sim->Hn);
    ScalarField *phi = &(sim->phi);
    FishData *body = &(sim->params->body);

    double dt = sim->params->dt;
    double dtau = sim->params->dtau;

    VectorField *buffer = &sim->buffer;

    double t_start = GET_TIME();

    // Viscosity term
    start_record(prof, "Viscosity:");
    viscosity_term(mesh, vstar, set_scal);
    op_vecfield(vstar, dt * sim->params->nu, mul_scal);
    end_record(prof, "Viscosity:");

    // Gradient term
    start_record(prof, "Gradient:");
    grad_field(buffer, &(mesh->P), mesh->h, set_scal);
    op_vecfieldwise_mul(vstar, buffer, dt, sub_scal);
    end_record(prof, "Gradient:");

    // Convective term / divergence form
    start_record(prof, "Divergence form:");
    divergence_form(mesh, Hn, set_scal);
    op_vecfieldwise_mul(Hnm1, Hn, 3, sub_scal);
    op_vecfieldwise_mul(vstar, Hnm1, 0.5 * dt, add_scal);
    op_vecfieldwise(Hnm1, Hn, set_scal);
    end_record(prof, "Divergence form:");
    
    // Penalization computations
    start_record(prof, "Penalization:");
    compute_speed_mask(body, vsn1, sim->t);
    op_vecfieldwise(vstar, &mesh->uv, add_scal);

    // Adds Penalization terms (at the u and v centers)
    op_vecfield(&body->mask, dt / dtau, mul_scal);
    op_vecfieldwise(vsn1, &body->mask, mul_scal);
    op_vecfieldwise(vstar, vsn1, add_scal);
    op_vecfield(&body->mask, 1.0, add_scal);
    op_vecfieldwise(vstar, &body->mask, div_scal);
    end_record(prof, "Penalization:");
    
    // Ghost points update
    start_record(prof, "Ghost update:");
    update_ghost_points(domain, vstar, mesh, sim->mode, body->ufish, dt);
    synchronize_vecfield(vstar, domain);
    end_record(prof, "Ghost update:");

    // Poisson solver
    start_record(prof, "Poisson solver:");
    divergence(phi, vstar, mesh->h, set_scal);
    poisson_solver(&(sim->pdata), phi, phi);
    synchronize_field(phi, domain);
    end_record(prof, "Poisson solver:");

    // Projection
    start_record(prof, "Projection:");
    op_vecfieldwise(&(mesh->uv), vstar, set_scal);
    grad_field(&(mesh->uv), phi, mesh->h, sub_scal);
    op_fieldwise_mul(&(mesh->P), phi, (1/dt), add_scal);
    end_record(prof, "Projection:");

    // Compute forces
    start_record(prof, "Forces:");
    op_vecfield(&body->mask, 1.0, sub_scal);
    op_vecfieldwise(vstar, &body->mask, mul_scal);
    op_vecfieldwise(vstar, vsn1, sub_scal);
    compute_forces(body, vstar, sim->mode);
    end_record(prof, "Forces:");

    op_field(phi, 0.0, set_scal);
    divergence(phi, &mesh->uv, mesh->h, set_scal);
    mprintf(" Div : %.3e ", absmax_field(phi));

    assert(!(has_nan_vecfield(&(mesh->uv))));
    assert(!(has_nan_field(&(mesh->P))));
    
    double cfl = (absmax_field(&(mesh->uv.u)) + absmax_field(&(mesh->uv.v))) * sim->params->dt / mesh->h;
    mprintf(" CFL = %.2f ", cfl);

    double t_end = GET_TIME();
    prof->total += t_end - t_start;
    prof->count++;
    
    // Print timing for this step
    mprintf("(%.2f ms)\n", 1000.0 * (t_end - t_start));
}

int main(int argc, char *argv[]) {

    bool testing = false;
    int episodes = -1;
    int n = 256;
    double Re = 1000.0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--test") == 0) testing = true;
        if (strcmp(argv[i], "-Re") == 0) Re = atof(argv[i+1]);
        if (strcmp(argv[i], "-e") == 0) episodes = atoi(argv[i+1]);
        if (strcmp(argv[i], "-n") == 0) n = atoi(argv[i+1]);
    }

    init_mpi(argc, argv);

    PetscInitialize(&argc, &argv, 0, 0);
    
    if (testing) {
        if (mpi_rank() == 0) {
            mprintf("Testing the finite differences \n");
            test_grad_term(1.0, 1.0, 1.0 / (1ULL << 4));
            test_grad_term(1.0, 1.0, 1.0 / (1ULL << 8));

            test_viscosity(1.0, 1.0, 1.0 / (1ULL << 4));
            test_viscosity(1.0, 1.0, 1.0 / (1ULL << 8));

            test_convective(1.0, 1.0, 1.0 / (1ULL << 4));
            test_convective(1.0, 1.0, 1.0 / (1ULL << 8));

            test_divergence(1.0, 1.0, 1.0 / (1ULL << 4));
            test_divergence(1.0, 1.0, 1.0 / (1ULL << 8));

        }
        // test_mpidomain(12, 12);
        test_poisson_solver(256);
        test_poisson_solver(345);
        MPI_Finalize();
        exit(0);
    }

    int ep = 0;
    SimulationParams params;
    Simulation sim;

    default_params(&params, n, Re, 2.0, 1.0, M_PERIODIC);
    params.num_epsiodes = (episodes > 0) ? episodes : params.num_epsiodes;

    initialize_dump("dump");
    init_simulation(&sim, &params, argc, argv);

    PostProcessor post = initialize_postprocessor(&sim, "dump");
    mprintf("Fourrier :  %.3e, dt : %.3e, nu : %.3e, nx : %d, ny : %d, h : %.3e \n", 
            params.nu * params.dt / (sim.mesh.h * sim.mesh.h), params.dt, params.nu, 
            params.domain.tnx, params.domain.tny, params.h);
    dump_params(&params, &post);
    
    // Initialize performance profiling
    PerformanceProfile profile;
    init_profile(&profile);
    
    // Initializes the simulation
    compute_speed_mask(&sim.params->body, &sim.vsn1, 0.0);
    divergence_form(&(sim.mesh), &(sim.Hnm1), set_scal);
    update_ghost_points(&params.domain, &(sim.Hnm1), &(sim.mesh), sim.mode, params.body.ufish, params.dt);
    
    double t_simulation_start = GET_TIME();
    double t_io = 0.0;
    
    for(int e = ep; e <= params.num_epsiodes; ++e) {
        mprintf("e : %d ->", e);
        sim_step(&sim, &profile);
        
        if(e % params.dump_period == 0){
            double t_io_start = GET_TIME();
            dump_mesh(e, &post);
            // dump_fish_data(&sim.params->body, &post);
            // dump_mesh_data(&params, &post);
            double t_io_end = GET_TIME();
            t_io += t_io_end - t_io_start;
        }

        sim.t += params.dt;
        
        // Print intermediate profile every 100 steps
        if ((e > 0) && (e % 100 == 0)) {
            print_profile(&profile);
        }
    }
    
    double t_simulation_end = GET_TIME();
    double t_simulation_total = t_simulation_end - t_simulation_start;
    
    // Final performance report
    print_profile(&profile);
    
    if (mpi_rank() == 0) {
        printf("\n=== Overall Timing Summary ===\n");
        printf("Total simulation time: %.3f s\n", t_simulation_total);
        printf("  Computation:         %.3f s (%5.1f%%)\n", 
               profile.total, 100.0 * profile.total / t_simulation_total);
        printf("  I/O:                 %.3f s (%5.1f%%)\n", 
               t_io, 100.0 * t_io / t_simulation_total);
        printf("  Other (overhead):    %.3f s (%5.1f%%)\n", 
               t_simulation_total - profile.total - t_io,
               100.0 * (t_simulation_total - profile.total - t_io) / t_simulation_total);
        printf("==============================\n\n");
    }
    
    free_field(&(post.w));
    free_simulation(&sim);
    PetscFinalize();
#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 0;
}