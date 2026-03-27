#ifndef _CUDISC_GAS1D_H_
#define _CUDISC_GAS1D_H_

#include "grid.h"
#include "star.h"
#include "cuda_array.h"
#include "dustdynamics1D.h"

void update_gas_sigma(Grid& g, CudaArray<double>& Sig_g, double dt, const CudaArray<double>& nu, int bound, double floor);

void update_gas_sigma(Grid& g, CudaArray<double>& Sig_g, double dt, const Field<double>& nu, int bound, double floor);

void update_gas_vel(Grid& g, CudaArray<double>& Sig_g, CudaArray<double>& u_gas, double alpha, Star& star);

/**
 * Calculates the gas radial velocity using a temperature profile derived from the inputted viscosity, nu.
 * This T is used to calculate the 2D gas density from Sig_g via hydrostatic equilibrium.
 * The final radial velocities are adjusted so that the radial Mdot is correct given the actual 2D gas density:
 *  
 *  vR = vR_param * rho_g_param / rho_g_actual
 * 
 * The azimuthal velocity is still calculated from the full temperature profile.
 */
void calc_gas_velocities_from_nu(Grid& g, CudaArray<double>& Sig_g, Field<Prims>& wg, Field<double>& cs2, CudaArray<double>& nu, double alpha, Star& star, int bound, double floor, double cav=0.) ;

/**
 * Calculates the gas radial velocity using a temperature profile derived from the inputted viscosity, nu.
 * This T is used to calculate the 2D gas density from Sig_g via hydrostatic equilibrium.
 * The final radial velocities are adjusted so that the radial Mdot is correct given the actual 2D gas density:
 *
 *  vR = vR_param * rho_g_param / rho_g_actual
 * 
 * The azimuthal velocity is still calculated from the full temperature profile.
 */
void calc_gas_velocities_from_nu(Grid& g, CudaArray<double>& Sig_g, Field<Prims>& wg, Field<double>& cs2, CudaArray<double>& nu, CudaArray<double>& alpha, Star& star, int bound, double floor, double cav=0.);
    
/**
 * Calculates the gas radial velocity using a parameterised temperature profile:
 * 
 * T = (6.25e-3 * Lstar / (pi r^2 sigma_SB)^0.25
 * 
 * This T is used to calculate the 2D gas density from Sig_g via hydrostatic equilibrium. 
 * The azimuthal velocity is still calculated from the full temperature profile.
 */
void calc_gas_velocities_parameterised(Grid& g, CudaArray<double>& Sig_g, Field<Prims>& wg, Field<double>& cs2, CudaArray<double>& nu, double alpha, Star& star, int bound, double floor, double cav=0.) ;

/**
 * Calculates the radial and azimuthal gas velocities from the full temperature profile.
 */
void calc_gas_velocities(Grid& g, CudaArray<double>& Sig_g, Field<Prims>& wg, Field<double>& cs2, CudaArray<double>& nu, double alpha, Star& star, int bound, double floor, double cav=0.) ;

/**
 * Calculates the gas radial velocity using a parameterised temperature profile:
 * 
 * T = (6.25e-3 * Lstar / (pi r^2 sigma_SB)^0.25
 * 
 * This T is used to calculate the 2D gas density from Sig_g via hydrostatic equilibrium. 
 * The azimuthal velocity is still calculated from the full temperature profile.
 * 
 * Vertical velocities are calculated using the wind mass-loss rate
 */
void calc_gas_velocities_wind(Grid& g, Field<Prims>& wg, CudaArray<double>& Sig_g, Field<double>& cs2, CudaArray<double>& nu, CudaArray<double>& Sig_dot_w,
                            double alpha, Star& star, int bound, double floor, double cav) ;

void update_gas_sources(Grid& g, CudaArray<double>& Sig_g, CudaArray<double>& Sigdot, CudaArray<double>& nu, double dt, int bound, double gfloor);

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