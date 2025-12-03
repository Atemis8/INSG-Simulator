#include "../headers/finite_diff.h"
#include "../headers/finite_diff.h"
#include "../headers/config.h"
#include <assert.h>

/*
Computes u* in [1, nx-3]x[1, ny-2]
Computes v* in [1, nx-2]x[1, ny-3]
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

    // Uses u in [1, nx-2]x[1, ny-1]
    // Uses v in [1, nx-1]x[1, ny-2]
    // Computes w in [0, nx-2]x[0, ny-2]
    PARALLEL(2)
    for(int x = 0; x < o->nx; ++x)
        for(int y = 0; y < o->ny; ++y) {
            float du_dy = (get_scal(u, x + 1, y + 2) - get_scal(u, x + 1, y + 1)) / mesh->h;
            float dv_dx = (get_scal(v, x + 2, y + 1) - get_scal(v, x + 1, y + 1)) / mesh->h;
            set_scal(o, x, y, dv_dx - du_dy);
        }
}

void divergence(ScalarField *o, VectorField *i, double h, SOP_SIG(op_scal)) {
    ScalarField *u = &i->u;
    ScalarField *v = &i->v;
    PARALLEL(2)
    // Uses u in [0, nx-2]x[1, ny-2]
    // Uses v in [1, nx-2]x[0, ny-2]
    // Computes P in [1, nx-2]x[1, ny-2]
    for(int x = 1; x < u->nx-1; ++x) {
        for(int y = 1; y < u->ny-1; ++y) {
            op_scal(o, x, y, (
                get_scal(u, x, y) - get_scal(u, x - 1, y) 
              + get_scal(v, x, y) - get_scal(v, x, y - 1)) / h);
        }
    }
}

/*
Computes u* in [1, nx-3]x[1, ny-2]
Computes v* in [1, nx-2]x[1, ny-3]
*/
void viscosity_term(MACMesh *mesh, VectorField *o, SOP_SIG(op_scal)) {
    ScalarField *u = &(mesh->uv.u);
    ScalarField *v = &(mesh->uv.v);
    assert(u->nx == o->u.nx && u->ny == o->u.ny);
    double h = mesh->h;

    // Uses u in [0, nx-2]x[0, ny-1]
    // Computes u* in [1, nx-3]x[1, ny-2]
    PARALLEL(2)
    for(int x = 1; x < u->nx-2; ++x)
        for(int y = 1; y < u->ny-1; ++y) {
            op_scal(&o->u, x, y, 
                    ( get_scal(u, x-1, y) + get_scal(u, x+1, y)
                    + get_scal(u, x, y-1) + get_scal(u, x, y+1)-4*get_scal(u, x, y)) / (h*h));
        }

    // Uses v in [0, nx-1]x[0, ny-2]
    // Computes v* in [1, nx-2]x[1, ny-3]
    PARALLEL(2)
    for(int x = 1; x < v->nx-1; ++x) 
        for(int y = 1; y < v->ny-2; ++y) {
            op_scal(&o->v, x, y, 
                ( get_scal(v, x-1, y) + get_scal(v, x+1, y)
                + get_scal(v, x, y-1) + get_scal(v, x, y+1)-4*get_scal(v, x, y)) / (h*h));
            }
}

/*
Computes a in [1, nx-3]x[1, ny-2]
Computes b in [1, nx-2]x[1, ny-3]
*/
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

    // Uses u in [0, nx-2]x[1, ny-2]
    // Computes uu in [1, nx-2]x[1, ny-2] (Pressure Centers)
    PARALLEL(2)
    for(int x = 1; x < uu->nx-1; ++x) {
        for(int y = 1; y < uu->ny-1; ++y) {
            double val = (get_scal(u, x - 1, y) + get_scal(u, x, y)) / 2.0;
            set_scal(uu, x, y, val * (val - vmesh->x));
        }
    }

    // Uses v in [1, nx-2]x[0, ny-2]
    // Computes vv in [1, nx-2]x[1, ny-2] (Pressure Centers)
    PARALLEL(2)
    for(int x = 1; x < vv->nx-1; ++x) { 
        for(int y = 1; y < vv->ny-1; ++y) {
            double val = (get_scal(v, x, y - 1) + get_scal(v, x, y)) / 2.0;
            set_scal(vv, x, y, val * (val - vmesh->y));
        }
    }

    // Uses u in [0, nx-2]x[1, ny-1] 
    // Uses v in [0, nx-1]x[0, ny-2]
    // Computes uv in [0, nx-2]x[0, ny-2] (Vorticity centers)
    // Computes vu in [0, nx-2]x[0, ny-2] (Vorticity centers)
    PARALLEL(2)
    for(int x = 0; x < uv->nx-1; ++x) {
        for(int y = 0; y < uv->ny-1; ++y) {
            double uavg = (get_scal(u, x, y) + get_scal(u, x, y + 1)) / 2.0;
            double vavg = (get_scal(v, x, y) + get_scal(v, x + 1, y)) / 2.0;
            set_scal(uv, x, y, uavg * (vavg - vmesh->y));
            set_scal(vu, x, y, (uavg - vmesh->x) * vavg);
        }
    }

    // Defined uu in [1, nx-2]x[1, ny-2] (Pressure Centers)
    // Defined uv in [0, nx-2]x[0, ny-2] (Vorticity centers)
    // Uses uu in [1, nx-2]x[1, ny-2]
    // Uses uv in [1, nx-3]x[0, ny-2]
    // Computes a in [1, nx-3]x[1, ny-2]
    PARALLEL(2)
    for(int x = 1; x < a->nx-2; ++x) {
        for(int y = 1; y < a->ny-1; ++y) {
            double duu = (get_scal(uu, x + 1, y) - get_scal(uu, x, y)) / (mesh->h);
            double duv = (get_scal(uv, x, y) - get_scal(uv, x, y - 1)) / (mesh->h);
            op_scal(a, x, y, duu + duv);
        }
    }


    // Defined vv in [1, nx-2]x[1, ny-2] (Pressure Centers)
    // Defined vu in [0, nx-2]x[0, ny-2] (Vorticity centers)
    // Uses vv in [1, nx-2]x[1, ny-2]
    // Uses vu in [0, nx-2]x[1, ny-3]
    // Computes b in [1, nx-2]x[1, ny-3]
    PARALLEL(2)
    for(int x = 1; x < b->nx-1; ++x) {
        for(int y = 1; y < b->ny-2; ++y) {
            double dvv = (get_scal(vv, x, y + 1) - get_scal(vv, x, y)) / (mesh->h);
            double duv = (get_scal(vu, x, y) - get_scal(vu, x - 1, y)) / (mesh->h);
            op_scal(b, x, y, duv + dvv);
        }
    }
}


void apply_periodic_bc(ScalarField *s) {
    int nx = s->nx;
    int ny = s->ny;

    for (int y = 0; y < ny; ++y) {
        set_scal(s, 0, y, get_scal(s, nx - 2, y));
        set_scal(s, nx - 1, y, get_scal(s, 1, y));
    }

    for (int x = 0; x < nx; ++x) {
        set_scal(s, x, 0, get_scal(s, x, ny - 2));
        set_scal(s, x, ny - 1, get_scal(s, x, 1));
    }

    set_scal(s, 0, 0, get_scal(s, nx - 2, ny - 2));
    set_scal(s, 0, ny - 1, get_scal(s, nx - 2, 1));
    set_scal(s, nx - 1, 0, get_scal(s, 1, ny - 2));
    set_scal(s, nx - 1, ny - 1, get_scal(s, 1, 1));
}


void update_ghost_points(MPIDomain *domain, VectorField *uv, MACMesh *mesh, Mode periodicityflag, double umesh, double dt) {

    ScalarField *u = &(uv->u);
    ScalarField *v = &(uv->v);

    switch (periodicityflag) {

    case M_PERIODIC:
    #ifndef USE_MPI
        v->type = -1;
        u->type = -1;
        apply_periodic_bc(u);
        apply_periodic_bc(v);
    #endif
        break;
    case M_BOUNDARY:
        break;
    default:
        break;
    }
}