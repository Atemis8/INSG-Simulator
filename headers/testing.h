#pragma once

void test_mpidomain(int argc, char **argv, int global_nx, int global_ny);
void test_grad_term(double Lx, double Ly, double h);
void test_viscosity(double Lx, double Ly, double h);
void test_divergence(double Lx, double Ly, double h);
void test_convective(double Lx, double Ly, double h);
void test_vectorfield_integration(double Lx, double Ly, double h);
void test_poisson_solver(int N);