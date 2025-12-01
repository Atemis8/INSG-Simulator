#include "../headers/config.h"
#include "../headers/mesh.h"
#include "../headers/utils.h"
#include "../headers/mpi_domain.h"

#define min(a, b) (a) < (b) ? (a) : (b)
#define max(a, b) (a) > (b) ? (a) : (b)

// Default params assume r = 0.1, L = 1.0 and T = 1.0 N is the number of point in the x direction
void default_params(SimulationParams *params, double N, double Re, double Lstar, double Hstar, Mode mode) {
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

    params->domain = init_domain(N, 1 + H / h, 1, mode);
    params->Re = Re;
    params->num_epsiodes = 5000;
    params->dump_period = 20;
    params->nu = nu;
    params->h = h;
    params->dt = 0.5 * min(dt_cfl, dt_r);
    params->dtau = params->dt / 1e4;
    params->mode = mode;

    params->body = initialize_body(&params->domain, L, H, h, params->dt, Lfish, mode);
    params->body.cont = create_controller(-0.05, 1.0, no_control);
}


void init_simulation(Simulation *sim, SimulationParams *params, int argc, char **argv) {
    init_poisson_solver(&params->domain, &(sim->pdata), params->h, params->mode);

    MACMesh mesh = allocate_mesh(params->domain.tnx, params->domain.tny, params->h);
    params->body.mask = allocate_vecfield(params->domain.tnx, params->domain.tny);
    params->body.mask.u.type = 0;
    params->body.mask.v.type = 1;

    sim->params = params;
    sim->mesh = mesh;
    sim->Hn = vecfield_like(&(mesh.uv));
    sim->Hnm1 = vecfield_like(&(mesh.uv));
    sim->vstar = vecfield_like(&(mesh.uv));
    sim->vsn1 = vecfield_like(&(mesh.uv));
    sim->buffer = vecfield_like(&(mesh.uv));
    sim->phi = field_like(&mesh.P);
    sim->mode = params->mode;
    sim->t = 0.0;
    mprintf("tnx : %d, tny : %d, nx : %d, ny : %d\n", params->domain.tnx, params->domain.tny, params->domain.nx, params->domain.ny);
    sim->vstar.u.type = 0;
    sim->vstar.v.type = 1;
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