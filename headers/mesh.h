#pragma once
#include <assert.h>
#ifdef USE_OPENMP
    #define PRAGMA(x) _Pragma(#x)
    #define PARALLEL(N) PRAGMA(omp parallel for collapse(N))
#else
    #define PARALLEL(N)
#endif

#define xytok(x, y, ny) ((x) * (ny) + (y))
/*
Definition of the mesh structure MAC mesh used to represent u,v and P
*/

typedef struct {
    double *v;
    int nx;
    int ny;
    int type;
} ScalarField;

// Operators on scalar fields
#define SOP_SIG(op) void (*op)(ScalarField *f, int x, int y, double val)
void add_scal(ScalarField *f, int x, int y, double val);
void sub_scal(ScalarField *f, int x, int y, double val);
void set_scal(ScalarField *f, int x, int y, double val);
void mul_scal(ScalarField *f, int x, int y, double val);
void div_scal(ScalarField *f, int x, int y, double val);
double get_scal(ScalarField *f, int i, int j);

// Field memory management
ScalarField allocate_field(int nx, int ny);
ScalarField field_like(ScalarField *field);
void free_field(ScalarField *field);

// Total field operators
double reduce_field(ScalarField *f);
double absmax_field(ScalarField *f);
int has_nan_field(ScalarField *f);
void op_field(ScalarField *f, double val, SOP_SIG(op));
void op_fieldwise(ScalarField *target, ScalarField *source, SOP_SIG(op));
void op_fieldwise_mul(ScalarField *target, ScalarField *source, double m, SOP_SIG(op));
void fill_field(ScalarField *f, double h, double (*func)(double, double));

typedef struct {
    ScalarField u;
    ScalarField v;
} VectorField;

typedef struct {
    double x;
    double y;
} Vector;

// Vecfield memory management
VectorField allocate_vecfield(int nx, int ny);
VectorField vecfield_like(VectorField* vecf);
void free_vecfield(VectorField *field);

// Total field operators
int has_nan_vecfield(VectorField *field);
void vec_norm(ScalarField *n, VectorField *f);
void op_scal_vecfield(VectorField *target, ScalarField *source, SOP_SIG(op));
void op_vecfield(VectorField *f, double val, SOP_SIG(op));
void op_vecfieldwise(VectorField *target, VectorField *source, SOP_SIG(op));
void op_vecfieldwise_mul(VectorField *target, VectorField *source, double mul, SOP_SIG(op)); // Multiplies source 

typedef struct {
    VectorField uv;
    VectorField ubuff;
    VectorField vbuff;
    ScalarField P;
    Vector vmesh;
    double h;
} MACMesh;

// MACMesh memory management
MACMesh allocate_mesh(int nx, int ny, double h);
void free_mesh(MACMesh *mesh);

