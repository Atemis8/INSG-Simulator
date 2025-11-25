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
#endif

double rectangle(double x, double y) {
    return (fabs(x-PI) < 0.5 && fabs(y-PI) < 0.5) ? 1.0 : 0.0;
}

void sim_step(Simulation *sim) {
    // ensure_debug_dir();
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
    viscosity_term(mesh, vstar, set_scal);
    op_vecfield(vstar, dt * sim->params->nu, mul_scal);

    // Computation of the gradient term
    grad_field(buffer, &(mesh->P), mesh->h, set_scal);
    op_vecfieldwise_mul(vstar, buffer, dt, sub_scal);

    // Computation of convective term
    if (sim->mode == M_BOUNDARY) mesh->vmesh.x = body->ufish;
    
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
    
    update_ghost_points(vstar,mesh, sim->mode,body->ufish,dt);

    // Compute the poisson problem 
    divergence(phi, vstar, mesh->h, set_scal);
    poisson_solver(&(sim->pdata), phi, phi);

    op_field(phi, mesh->h * mesh->h, mul_scal);
    op_vecfieldwise(&(mesh->uv), vstar, set_scal);
    grad_field(&(mesh->uv), phi, mesh->h, sub_scal);

    // Computes forces on the fish
    op_vecfield(&body->mask, 1.0, sub_scal); // X * dt / dtau
    op_vecfieldwise(vstar, &body->mask, mul_scal); // v* * X * dt / dtau 
    op_vecfieldwise(vstar, vsn1, sub_scal); // (v*-vsn1) * X * dt / dtau 
    compute_forces(body,vstar, sim->mode);

    // Now simply add this to the original values
    op_fieldwise_mul(&(mesh->P), phi, (1/dt), add_scal);
    body->cont->up(body);

    assert(!(has_nan_vecfield(&(mesh->uv))));
    assert(!(has_nan_field(&(mesh->P))));

    double cfl = (absmax_field(&(mesh->uv.u)) + absmax_field(&(mesh->uv.v))) * sim->params->dt / mesh->h;
    mprintf(" CFL = %.2f \
        \n", cfl 
        /*, dts: %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f
        (double)(t1 - t0) / (double) ttot, // Viscosity
        (double)(t2 - t1) / (double) ttot, // Gradient
        (double)(t3 - t2) / (double) ttot, // Convective
        (double)(t4 - t3) / (double) ttot, // Penalization comp
        (double)(t5 - t4) / (double) ttot, // Penalization add
        (double)(t7 - t5) / (double) ttot, // Poisson
        (double)(t8 - t7) / (double) ttot, // Grad
        (double)(t9 - t8) / (double) ttot  // Final comps */
    );
}

int main(int argc, char *argv[]) {

    bool load_sim = false;
    bool testing = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--load") == 0) load_sim = true;
        if (strcmp(argv[i], "--test") == 0) testing = true;
    }

#ifdef USE_MPI
    MPI_Finalize();
    // init_mpi(argc, argv);
#endif

    // test_vectorfield_integration(1.5, 1.5, 0.01);
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
        test_mpidomain(argc, argv, 16, 16);
        test_poisson_solver(100);
    }

    int ep = 0;
    SimulationParams params;
    Simulation sim;

    if(load_sim) {
        params = load_params("dump");
        sim = load_simulation(&params, "dump", &ep);
    } else {
        params = default_params(256, 1000, 2.0, 1.0, M_PERIODIC);
        sim = init_simulation(&params);
        initialize_dump("dump");
    }

    PostProcessor post = initialize_postprocessor(&sim, "dump");
    mprintf("Fourrier :  %.3e, dt : %.3e, nu : %.3e, nx : %d, ny : %d, h : %.3e \n", params.nu * params.dt / (sim.mesh.h * sim.mesh.h), params.dt, params.nu, params.body.nx, params.body.ny,params.h);
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