#include "../headers/config.h"
#include "../headers/mesh.h"
#include "../headers/utils.h"

#define min(a, b) (a) < (b) ? (a) : (b)
#define max(a, b) (a) > (b) ? (a) : (b)

// Default params assume r = 0.1, L = 1.0 and T = 1.0 N is the number of point in the x direction
SimulationParams default_params(double N, double Re, double Lstar, double Hstar,int mode) {
    // Params to choose 
    double L = 1.0;
    double T = 1.0;
    double r_max = 0.2;
    double CFL_max = 0.6;
    double u_max_guess = 1.0;

    double Lfish = L / Lstar;
    double H = L * Hstar / Lstar;
    double h = L / (N - 1);
    double nu = Lfish * Lfish / (Re * T);

    double dt_cfl = CFL_max * h / u_max_guess;
    double dt_r = r_max * h * h / nu; 

    mprintf("DT CFL : %.3e, DT R : %.3e \n", dt_cfl, dt_r);

    double dt = min(dt_cfl, dt_r);
    dt = 1e-3;

    SimulationParams params = {
        .Re = Re,
        .num_epsiodes = 20000,
        .dump_period = 50,
        .nu = nu,
        .h = h,
        .dt = dt,
        .dtau = dt / 1e4,
        .mode = mode,
        .body = initialize_body(L, H, h, dt, Lfish, N + 2, 3 + H / h, mode)
    };
    // params.body.xfish = 0.4 * L;
    params.body.cont = create_controller(-0.05, 1.0, no_control);
    return params;
}

Simulation init_simulation(SimulationParams *params) {
    MACMesh mesh = allocate_mesh(params->body.nx - 2, params->body.ny - 2, params->h);
    Simulation sim = {
        .params = params,
        .mesh = mesh,
        .Hn = vecfield_like(&(mesh.uv)),
        .Hnm1 = vecfield_like(&(mesh.uv)),
        .vstar = vecfield_like(&(mesh.uv)),
        .vsn1 = vecfield_like(&(mesh.uv)),
        .buffer = vecfield_like(&(mesh.uv)),
        .phi = field_like(&mesh.P),
        .mode = params->mode,
        .t = 0.0
    };
    sim.vstar.u.type = 0;
    sim.vstar.v.type = 1;
    initialize_poisson_solver(&(sim.pdata), &(sim.phi), params->mode);
    return sim;
}

void free_simulation(Simulation* sim) {
    free_poisson_solver(&(sim->pdata));
    free_field(&(sim->phi));
    free_vecfield(&(sim->vstar));
    free_vecfield(&(sim->Hnm1));
    free_vecfield(&(sim->Hn));
    free_vecfield(&(sim->vsn1));
    free_vecfield(&(sim->buffer));
    free_body(&(sim->params->body));
    free_mesh(&(sim->mesh));
}