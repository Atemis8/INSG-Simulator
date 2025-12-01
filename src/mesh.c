#include "../headers/mesh.h"
#include "../headers/utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <execinfo.h>
#include <stdbool.h>

#define ASSERT(x) \
    do { \
        if (!(x)) { \
            fprintf(stderr, "Assertion failed: %s, file %s, line %d\n", #x, __FILE__, __LINE__); \
            print_stacktrace(); \
            abort(); \
        } \
    } while (0)

void print_stacktrace() {
    void *buffer[100];
    int nptrs = backtrace(buffer, 100);
    backtrace_symbols_fd(buffer, nptrs, fileno(stderr));
}

static bool g_assertions_enabled = true;

void check(ScalarField *f, int x, int y) {
    if(g_assertions_enabled) {
        ASSERT(x < f->nx && y < f->ny);
        ASSERT(!(f->type == 0 && x == f->nx-1));
        ASSERT(!(f->type == 1 && y == f->ny-1));
    }
}

void enable_assertions() {
    g_assertions_enabled = true;
}

void disable_assertions() {
    g_assertions_enabled = false;
}

// Define the scalar operators
void add_scal(ScalarField *f, int x, int y, double val) {
    check(f, x, y);
    f->v[xytok(x, y, f->ny)] += val;
}

void sub_scal(ScalarField *f, int x, int y, double val) {
    check(f, x, y);
    f->v[xytok(x, y, f->ny)] -= val;
}

void mul_scal(ScalarField *f, int x, int y, double val) {
    check(f, x, y);
    f->v[xytok(x, y, f->ny)] *= val;
}

void div_scal(ScalarField *f, int x, int y, double val) {
    check(f, x, y);
    f->v[xytok(x, y, f->ny)] /= val;
}
 
void set_scal(ScalarField *f, int x, int y, double val) {
    check(f, x, y);
    f->v[xytok(x, y, f->ny)] = val;
}

double get_scal(ScalarField *f, int x, int y) {
    check(f, x, y);
    return f->v[xytok(x, y, f->ny)];
}

int has_nan_field(ScalarField *f) {
    for (int i = 0; i < f->nx * f->ny; ++i) if (isnan(f->v[i])) return 1;
    return 0;
}

double reduce_field(ScalarField *f) {
    double val = 0.0;
    for (int i = 0; i < f->nx * f->ny; ++i) val += f->v[i];
    double global_sum = 0.0;

    
    #ifdef USE_MPI
    MPI_Allreduce(&val, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    #else
    global_sum = val;
    #endif
    
    return global_sum;
}

double absmax_field(ScalarField *f) {
    double val = 0.0;
    for (int i = 0; i < f->nx * f->ny; ++i) if(fabs(f->v[i]) > val) val = fabs(f->v[i]);
    double global_max = 0.0;

    
    #ifdef USE_MPI
    MPI_Allreduce(&val, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    #else
    global_max = val;
    #endif
    
    return global_max;
}

void fill_field(ScalarField *f, double h, double (*func)(double, double)) {
    disable_assertions();
    PARALLEL(2)
    for(int x = 0; x < f->nx; ++x)
        for(int y = 0; y < f->ny; ++y)
            set_scal(f, x, y, func(x * h, y * h));
    enable_assertions();
}

void op_field(ScalarField *f, double val, SOP_SIG(op)) {
    disable_assertions();
    PARALLEL(2)
    for(int x = 0; x < f->nx; ++x)
        for(int y = 0; y < f->ny; ++y)
            op(f, x, y, val);
    enable_assertions();
}

void op_fieldwise(ScalarField *t, ScalarField *s, SOP_SIG(op)) {
    disable_assertions();
    PARALLEL(2)
    for(int x = 0; x < s->nx; ++x)
        for(int y = 0; y < s->ny; ++y)
            op(t, x, y, get_scal(s, x, y));
    enable_assertions();
}

void op_fieldwise_mul(ScalarField *t, ScalarField *s, double m, SOP_SIG(op)) {
    disable_assertions();
    PARALLEL(2)
    for(int x = 0; x < s->nx; ++x)
        for(int y = 0; y < s->ny; ++y)
            op(t, x, y, m * get_scal(s, x, y));
    enable_assertions();
}

ScalarField allocate_field(int nx, int ny) {
    double *v = calloc(nx * ny, sizeof(double));
    return (ScalarField) {.v=v, .nx=nx, .ny=ny, .type=-1};
}

ScalarField field_like(ScalarField *f) {
    double *v = calloc(f->nx * f->ny, sizeof(double));
    return (ScalarField) {.v=v, .nx=f->nx, .ny=f->ny, .type=-1};
}

void free_field(ScalarField *field) {
    free(field->v);
}

void vec_norm(ScalarField *n, VectorField *f) {
    PARALLEL(2)
    for(int x = 0; x < f->u.nx; ++x)
        for(int y = 0; y < f->u.ny; ++y) {
            double u = get_scal(&f->u, x, y);
            double v = get_scal(&f->v, x, y);
            set_scal(n, x, y, sqrt(u * u + v * v));
        }
}

VectorField allocate_vecfield(int nx, int ny) {
    return (VectorField) {.u = allocate_field(nx, ny), .v = allocate_field(nx, ny)};
}

VectorField vecfield_like(VectorField* f) {
    return (VectorField) {.u = field_like(&(f->u)), .v = field_like(&(f->v))};
}

void free_vecfield(VectorField *field) {
    free_field(&(field->u));
    free_field(&(field->v));
}

int has_nan_vecfield(VectorField *f) {
    return has_nan_field(&(f->u)) || has_nan_field(&(f->v));
}

void op_scal_vecfield(VectorField *t, ScalarField *s, SOP_SIG(op)) {
    op_fieldwise(&t->u, s, op);
    op_fieldwise(&t->v, s, op);
}

void op_vecfield(VectorField *f, double val, SOP_SIG(op)) {
    op_field(&f->u, val, op);
    op_field(&f->v, val, op);
}

void op_vecfieldwise(VectorField *t, VectorField *s, SOP_SIG(op)) {
    op_fieldwise(&t->u, &s->u, op);
    op_fieldwise(&t->v, &s->v, op);
}

void op_vecfieldwise_mul(VectorField *t, VectorField *s, double mul, SOP_SIG(op)) {
    op_fieldwise_mul(&t->u, &s->u, mul, op);
    op_fieldwise_mul(&t->v, &s->v, mul, op);
}

/*
We will add an additional ghost point that may seem strange 
but it will greatly help with boundary conditions. Adding those 
points will also allow for all fields to have the same size
This means that the last line and last column of the vorticity field are not used
v   -   v   -   v   -   v   -   v   -

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
<------> is h
We choose the following axes :
^ y
|
| 
|----> x
*/
MACMesh allocate_mesh(int nx, int ny, double h) {
    MACMesh mesh = (MACMesh) {
        .P = allocate_field(nx, ny), 
        .uv = allocate_vecfield(nx, ny),
        .ubuff = allocate_vecfield(nx, ny),
        .vbuff = allocate_vecfield(nx, ny),
        .h = h
    };
    mesh.uv.u.type = 0;
    mesh.uv.v.type = 1;
    return mesh;
}

void free_mesh(MACMesh *mesh) {
    free_field(&(mesh->P));
    free_vecfield(&(mesh->uv));
    free_vecfield(&(mesh->ubuff));
    free_vecfield(&(mesh->vbuff));
}