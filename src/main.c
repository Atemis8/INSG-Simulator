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

double rectangle(double x, double y) {
    return (fabs(x-PI) < 0.5 && fabs(y-PI) < 0.5) ? 1.0 : 0.0;
}

void sim_step(Simulation *sim) {
    // ensure_debug_dir();
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

    // Computation of the viscosity term
    // synchronize_vecfield(&mesh->uv, domain);
    viscosity_term(mesh, vstar, set_scal);
    op_vecfield(vstar, dt * sim->params->nu, mul_scal);

    // Computation of the gradient term
    // synchronize_field(&mesh->P, domain);
    grad_field(buffer, &(mesh->P), mesh->h, set_scal);
    op_vecfieldwise_mul(vstar, buffer, dt, sub_scal);

    // Computation of convective term
    if (sim->mode == M_BOUNDARY) mesh->vmesh.x = body->ufish;
    
    // synchronize_vecfield(&mesh->uv, domain);
    divergence_form(mesh, Hn, set_scal);
    op_vecfieldwise_mul(Hnm1, Hn, 3, sub_scal);
    op_vecfieldwise_mul(vstar, Hnm1, 0.5 * dt, add_scal);
    op_vecfieldwise(Hnm1, Hn, set_scal);
    
    // Penalization computations
    compute_speed_mask(body, vsn1, sim->t);
    op_vecfieldwise(vstar, &mesh->uv, add_scal);

    // Adds Penalization terms
    op_vecfield(&body->mask, dt / dtau, mul_scal);
    op_vecfieldwise(vsn1, &body->mask, mul_scal);
    op_vecfieldwise(vstar, vsn1, add_scal);
    op_vecfield(&body->mask, 1.0, add_scal);
    op_vecfieldwise(vstar, &body->mask, div_scal);

    // Updates ghost points
    if (sim->mode == M_BOUNDARY) bc_outflow(mesh,vstar, body->ufish, dt);
    
    update_ghost_points(domain, vstar, mesh, sim->mode,body->ufish,dt);

    // Compute the poisson problem 
    // synchronize_vecfield(vstar, domain);
    divergence(phi, vstar, mesh->h, set_scal);
    poisson_solver(&(sim->pdata), phi, phi);

    op_field(phi, mesh->h * mesh->h, mul_scal);
    op_vecfieldwise(&(mesh->uv), vstar, set_scal);
    // synchronize_field(phi, domain);
    grad_field(&(mesh->uv), phi, mesh->h, sub_scal);

    // Computes forces on the fish
    op_vecfield(&body->mask, 1.0, sub_scal); // X * dt / dtau
    op_vecfieldwise(vstar, &body->mask, mul_scal); // v* * X * dt / dtau 
    op_vecfieldwise(vstar, vsn1, sub_scal); // (v*-vsn1) * X * dt / dtau 
    compute_forces(body, vstar, sim->mode);

    // Now simply add this to the original values
    op_fieldwise_mul(&(mesh->P), phi, (1/dt), add_scal);
    body->cont->up(body);

    assert(!(has_nan_vecfield(&(mesh->uv))));
    assert(!(has_nan_field(&(mesh->P))));

    double cfl = (absmax_field(&(mesh->uv.u)) + absmax_field(&(mesh->uv.v))) * sim->params->dt / mesh->h;
    mprintf(" CFL = %.2f \n", cfl);
}

int main(int argc, char *argv[]) {

    bool testing = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--test") == 0) testing = true;
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
        test_mpidomain(16, 16);
        test_poisson_solver(100);
    }

    int ep = 0;
    SimulationParams params;
    Simulation sim;


    default_params(&params, 256, 1000, 1.0, 1.5, M_PERIODIC);
    initialize_dump("dump");
    init_simulation(&sim, &params, argc, argv);

    PostProcessor post = initialize_postprocessor(&sim, "dump");
    mprintf("Fourrier :  %.3e, dt : %.3e, nu : %.3e, nx : %d, ny : %d, h : %.3e \n", params.nu * params.dt / (sim.mesh.h * sim.mesh.h), params.dt, params.nu, params.domain.tnx, params.domain.tny,params.h);
    dump_params(&params, &post);
    
    // Initializes the simulation
    compute_speed_mask(&sim.params->body, &sim.vsn1, 0.0);
    divergence_form(&(sim.mesh), &(sim.Hnm1), set_scal);
    for(int e = ep; e <= params.num_epsiodes; ++e) {
        mprintf("e : %d ->", e);
        sim_step(&sim);
        if(e % params.dump_period == 0 && mpi_rank() == 0){
            dump_mesh(e, &post);
            dump_fish_data(&sim.params->body, &post);
            dump_mesh_data(&params, &post);
        }

        sim.t += params.dt;
    }
    free_field(&(post.w));
    free_simulation(&sim);
    PetscFinalize();
    return 0;
}