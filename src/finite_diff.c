#include "../headers/finite_diff.h"
#include "../headers/finite_diff.h"
#include "../headers/config.h"
#include <assert.h>


/*
Uses values of the scalar field from 0 to nx-1 and ny-1 (entire field)
*/
void grad_field(VectorField* o, ScalarField *i, double h, SOP_SIG(op_scal)) {
    ScalarField *dx = &o->u;
    ScalarField *dy = &o->v;

    PARALLEL(2)
    for(int x = 0; x < dx->nx-1; ++x) 
        for(int y = 0; y < dx->ny-1; ++y) {
            op_scal(dx, x, y, (get_scal(i, x + 1, y) - get_scal(i, x, y)) / (h));
            op_scal(dy, x, y, (get_scal(i, x, y + 1) - get_scal(i, x, y)) / (h)); 
        }
}

void vorticity(ScalarField *o, MACMesh *mesh) {
    ScalarField *u = &(mesh->uv.u);
    ScalarField *v = &(mesh->uv.v);

    PARALLEL(2)
    for(int x = 0; x < u->nx-1; ++x)
        for(int y = 0; y < u->ny-1; ++y) {
            float dv_dx = (get_scal(v, x + 1, y) - get_scal(v, x, y)) / mesh->h;
            float du_dy = (get_scal(u, x, y + 1) - get_scal(u, x, y)) / mesh->h;
            set_scal(o, x, y, dv_dx - du_dy);
        }
}

void divergence(ScalarField *o, VectorField *i, double h, SOP_SIG(op_scal)) {
    ScalarField *u = &i->u;
    ScalarField *v = &i->v;
    PARALLEL(2)
    for(int x = 1; x < u->nx-1; ++x) {
        for(int y = 1; y < u->ny-1; ++y) {
            op_scal(o, x, y, (
                get_scal(u, x, y) - get_scal(u, x-1, y) 
              + get_scal(v, x, y) - get_scal(v, x, y-1)) / h);
        }
    }
}

void viscosity_term(MACMesh *mesh, VectorField *o, SOP_SIG(op_scal)) {
    ScalarField *u = &(mesh->uv.u);
    ScalarField *v = &(mesh->uv.v);
    assert(u->nx == o->u.nx && u->ny == o->u.ny);
    double h = mesh->h;
    PARALLEL(2)
    for(int x = 1; x < u->nx-2; ++x) 
        for(int y = 1; y < u->ny-1; ++y) {
            op_scal(&o->u, x, y, 
                ( get_scal(u, x-1, y) + get_scal(u, x+1, y)
                + get_scal(u, x, y-1) + get_scal(u, x, y+1)-4*get_scal(u, x, y)) / (h*h));
            }
    
    PARALLEL(2)
    for(int x = 1; x < u->nx-1; ++x) 
        for(int y = 1; y < u->ny-2; ++y) {
            op_scal(&o->v, x, y, 
                ( get_scal(v, x-1, y) + get_scal(v, x+1, y)
                + get_scal(v, x, y-1) + get_scal(v, x, y+1)-4*get_scal(v, x, y)) / (h*h));
            }
}

void divergence_form(MACMesh *mesh, VectorField *ab, SOP_SIG(op_scal)) {
    ScalarField *u = &(mesh->uv.u);
    ScalarField *v = &(mesh->uv.v);

    ScalarField *a = &ab->u;
    ScalarField *b = &ab->v;

    ScalarField *uu = &(mesh->ubuff.u);
    ScalarField *uv = &(mesh->ubuff.v);
    ScalarField *vv = &(mesh->vbuff.u);
    ScalarField *vu = &(mesh->vbuff.v);

    Vector *vmesh = &(mesh->vmesh);

    // Compute u(u-u0)_{x, y} from u_{x+1/2,y} (pressure centers)
    PARALLEL(2)
    for(int x = 1; x < uu->nx-1; ++x) {
        for(int y = 0; y < uu->ny; ++y) {
            double val = (get_scal(u, x - 1, y) + get_scal(u, x, y)) / 2.0;
            set_scal(uu, x, y, val * (val - vmesh->x));
        }
    }

    // Compute vv_{x, y} from v_{x, y+1/2} (pressure centers)
    PARALLEL(2)
    for(int x = 0; x < vv->nx; ++x) { 
        for(int y = 1; y < vv->ny-1; ++y) {
            double val = (get_scal(v, x, y - 1) + get_scal(v, x, y)) / 2.0;
            set_scal(vv, x, y, val * (val - vmesh->y));
        }
    }

    // Computes uv_{x+1/2, y+1/2} from u_{x+1/2, y} and v_{x, y+1/2} (vorticity center)
    PARALLEL(2)
    for(int x = 0; x < uv->nx-1; ++x) {
        for(int y = 0; y < uv->ny-1; ++y) {
            double uavg = (get_scal(u, x, y) + get_scal(u, x, y + 1)) / 2.0;
            double vavg = (get_scal(v, x, y) + get_scal(v, x + 1, y)) / 2.0;
            set_scal(uv, x, y, uavg * (vavg - vmesh->y));
        }
    }

    // Computes vu_{x+1/2, y+1/2} at the vorticity centers (computations coule be avoided if no vmesh)
    PARALLEL(2)
    for(int x = 0; x < vu->nx-1; ++x) {
        for(int y = 0; y < vu->ny-1; ++y) {
            double uavg = (get_scal(u, x, y) + get_scal(u, x, y + 1)) / 2.0;
            double vavg = (get_scal(v, x, y) + get_scal(v, x + 1, y)) / 2.0;
            set_scal(vu, x, y, (uavg - vmesh->x) * vavg);
        }
    }

    // Computes a_{x+1/2, y} from uu_{x, y} and uv_{x+1/2, y+1/2} (at the u centers)
    PARALLEL(2)
    for(int x = 1; x < a->nx-1; ++x) {
        for(int y = 1; y < a->ny-1; ++y) {
            double duu = (get_scal(uu, x + 1, y) - get_scal(uu, x, y)) / (mesh->h);
            double duv = (get_scal(uv, x, y) - get_scal(uv, x, y - 1)) / (mesh->h);
            op_scal(a, x, y, duu + duv);
        }
    }

    // Computes b_{x, y+1/2} from vv_{x, y} and uv_{x+1/2, y+1/2} (at the v centers)
    PARALLEL(2)
    for(int x = 1; x < b->nx-1; ++x) {
        for(int y = 1; y < b->ny-1; ++y) {
            double duv = (get_scal(vu, x, y) - get_scal(vu, x - 1, y)) / (mesh->h);
            double dvv = (get_scal(vv, x, y + 1) - get_scal(vv, x, y)) / (mesh->h);
            op_scal(b, x, y, duv + dvv);
        }
    }
}
void update_ghost_points(MPIDomain *domain, VectorField *uv, MACMesh *mesh, Mode periodicityflag, double umesh, double dt) {

    ScalarField *u = &(uv->u);
    ScalarField *v = &(uv->v);
    ScalarField *un = &(mesh->uv.u);
    ScalarField *vn = &(mesh->uv.v);
    double u_gamma;
    double vstar;
    double wn;
    double wn1;
    double w_star;

    switch (periodicityflag) {
#ifndef USE_MPI
    v->type = -1;
    u->type = -1;
    case M_PERIODIC:
        for (int x = 0; x < u->nx-1; ++x) {
            set_scal(u, x, 0, get_scal(u, x, u->ny - 2));
            set_scal(u, x, u->ny - 1, get_scal(u, x, 1));
        }

        for (int x = 0; x < v->nx; x++) { 
            // set_scal(v, x, 0, get_scal(v, x, v->ny - 2));
            set_scal(v, x, v->ny - 2, get_scal(v, x, 0));
        }
        // West and East v rows from interior of domain
        for (int y = 0; y < u->ny-1; y++) {
            set_scal(v, 0, y, get_scal(v, v->nx - 2, y));
            set_scal(v, v->nx-1, y, get_scal(v, 1, y));
        }

        // Left u column from right u column
        for (int y = 0; y < u->ny; y++) {
            set_scal(u, u->nx-2, y, get_scal(u, 0, y));
        }
        break;
#endif
    case M_BOUNDARY: // Not periodic

        // Top and bottom conditions
        for (int x = 0; x < u->nx; x++) {
            set_scal(v, x, 0, 0.0);
            set_scal(v, x, u->ny - 2, 0.0);
        }
        
        // Apply boundary on left wall
        for (int y = 0; y < u->ny; y++) {
            set_scal(u, 0, y, 0.);
            double dx = (get_scal(un, u->nx-2, y) - get_scal(un, u->nx-3, y)) / mesh->h;
            set_scal(u, u->nx-2, y, get_scal(un, u->nx-2, y) + umesh * dt * dx);
        }

        // Apply boundary condition on Top and bottom walls
        for (int x = 0; x < u->nx-1; x++) {
            set_scal(u, x, 0, get_scal(u, x, 1));
            set_scal(u, x, u->ny - 1, get_scal(u, x, u->ny - 2));
        }

        for (int y = 0; y < u->ny-1; y++) { 
                wn = ((get_scal(vn, vn->nx-1, y) - get_scal(vn, vn->nx-2, y)) - ((get_scal(un, un->nx-2, y+1) - get_scal(un, un->nx-2, y)))) / mesh->h;
                wn1 =  ((get_scal(vn, vn->nx-2, y) - get_scal(vn, vn->nx-3, y)) - ((get_scal(un, un->nx-3, y+1) - get_scal(un, un->nx-3, y)))) / mesh->h;
                w_star = wn  + umesh * dt * (wn - wn1) / mesh->h;
                vstar = mesh->h * w_star + get_scal(u, un->nx-2, y+1) - get_scal(u, un->nx-2, y) +  get_scal(v, vn->nx-2, y);

                set_scal(v, 0, y, get_scal(v, 1, y)); // set vorticity to 0 v_i+1 = v_i
                set_scal(v, v->nx - 1, y, vstar);
        }
        break;

    default:
        break;
    }
}

void bc_outflow(MACMesh *mesh, VectorField *vstar,double umesh,double dt) {

    double u_gamma;
    double v_gamma;

    ScalarField *u = &(vstar->u);
    ScalarField *un = &(mesh->uv.u);

    for (int y = 0; y < u->ny; y++) {
        u_gamma = get_scal(un,un->nx - 2,y) + umesh * dt/mesh->h + (get_scal(un,un->nx-2,y - get_scal(un,un->nx-3,y)));
        set_scal(u, u->nx - 2, y, u_gamma); // set boundary ustar
    }
    

}