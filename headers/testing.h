#pragma once

void test_mpidomain(int global_nx, int global_ny);
void test_grad_term(double Lx, double Ly, double h);
void test_viscosity(double Lx, double Ly, double h);
void test_divergence(double Lx, double Ly, double h);
void test_convective(double Lx, double Ly, double h);
void test_vectorfield_integration(double Lx, double Ly, double h);
void test_poisson_solver(int N);

#ifdef USE_MPI
void test_basic_collectives(int rank, int size);
void test_mpi_file_open(int rank, int size);
void test_mpi_file_write_all(int rank, int size);
void test_filesystem_info(int rank);
void test_with_timeout(int rank, int size);
#endif