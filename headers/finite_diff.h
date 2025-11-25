#pragma once
#include "mesh.h"

enum Axis {
    AXIS_X,
    AXIS_Y,
    AXIS_BOTH
};

void interp_u(MACMesh *mesh, ScalarField *u);
void interp_v(MACMesh *mesh, ScalarField *u);
void divergence(ScalarField *o, VectorField *i, double h, SOP_SIG(op_scal));
void vorticity(ScalarField *o, MACMesh *mesh);
void divergence_form(MACMesh *mesh, VectorField *ab, SOP_SIG(op_scal));
void viscosity_term(MACMesh *mesh, VectorField *dvv, SOP_SIG(op_scal));
void grad_field(VectorField* o, ScalarField *i, double h, SOP_SIG(op_scal));

void update_ghost_points(VectorField *uv,MACMesh *mesh, int periodicityflag,double umesh,double dt);

void bc_outflow(MACMesh *mesh, VectorField *vstar,double umesh,double dt); // update vstar for natural outflow