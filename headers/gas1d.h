#ifndef _CUDISC_GAS1D_H_
#define _CUDISC_GAS1D_H_

#include "grid.h"
#include "star.h"
#include "cuda_array.h"
#include "dustdynamics1D.h"

void update_gas_sigma(Grid& g, CudaArray<double>& Sig_g, double dt, const CudaArray<double>& nu, int bound, double floor);

void update_gas_sigma(Grid& g, CudaArray<double>& Sig_g, double dt, const Field<double>& nu, int bound, double floor);

void update_gas_vel(Grid& g, CudaArray<double>& Sig_g, CudaArray<double>& u_gas, double alpha, Star& star);

void calc_gas_velocities(Grid& g, CudaArray<double>& Sig_g, Field<Prims>& wg, Field<double>& cs2, CudaArray<double>& nu, double alpha, Star& star, int bound, double floor, double cav=0.) ;
void calc_gas_velocities_wind(Grid& g, Field<Prims>& wg, CudaArray<double>& Sig_g, Field<double>& cs2, CudaArray<double>& nu, CudaArray<double>& Sig_dot_w,
                            double alpha, Star& star, int bound, double floor, double cav) ;

void update_gas_sources(Grid& g, CudaArray<double>& Sig_g, CudaArray<double>& Sigdot, double dt, int bound, double gfloor);

double calc_dt(Grid& g, const CudaArray<double>& nu);
double calc_dt(Grid& g, const Field<double>& nu);

void calc_wind_surface(Grid& g, const Field<Prims>& wg, CudaArray<double>& h_w, double col);

void calculate_ubar(Grid& g, CudaArray<double>& sig, CudaArray<double>& sig_g, 
                    CudaArray<double>& ubar, CudaArray<double>& u_gas,
                    double t, double u_f, double rho_s, double alpha, double a0, Star& star, int, int);

void update_dust_sigma(Grid& g, CudaArray<double>& sig, CudaArray<double>& sig_g, 
                    CudaArray<double>& ubar, CudaArray<double>& D, double dt, int bound);

double compute_CFL(Grid& g, CudaArray<double>& ubar, CudaArray<double>& D,
                        double CFL_adv, double CFL_diff);

// Prims1D functions

void update_gas_sigma(Grid& g, Field<Prims1D>& W_g, double dt, const CudaArray<double>& nu, int bound, double floor);
void calc_v_gas(Grid& g, Field<Prims1D>& W_g, const Field<double>& cs, CudaArray<double>& nu, double GMstar, double gasfloor);

#endif//_CUDISC_HEADERS_GAS1D_H_