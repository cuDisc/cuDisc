#ifndef _CUDISC_HEADERS_DRAG_CONST_H_
#define _CUDISC_HEADERS_DRAG_CONST_H_

#include "constants.h"
#include "dustdynamics.h"
#include "dustdynamics1D.h"

__device__ __host__
inline double calc_C_D(double a, double rho_g, double cs, double u_rel, double mu) {

    // https://www.aanda.org/articles/aa/pdf/2018/03/aa31824-17.pdf

    double mfp = mu*m_H/(rho_g*2e-15);

    double sqrtgamma = pow(1.+2.*mu/5.,0.5);   

    double nu_mol = 1.595769122 * 0.5 * mfp*cs;
    double Re = (2.*a*u_rel)/nu_mol;

    double M = u_rel/cs;
    double G = pow(10.,2.5*pow(Re/312.,0.6688) / (1.+pow(Re/312.,0.6688)));
    double H = 4.6/(1.+M) + 1.7;

    double C_inc = 24./Re * (1.+0.15*pow(Re,0.687)) + 0.407*Re / (Re+8710);
    double C_D = 2.+(C_inc-2.)*exp(-(3.07*sqrtgamma*M*G)/Re) + H * exp(-Re/(2*M)) / (sqrtgamma*M);
    
    return C_D;
}

__device__ __host__
inline double calc_C_D_step(double a, double rho_g, double cs, double u_rel, double mu, double mfp) {

    double nu_mol = 1.595769122 * 0.5 * mfp*cs;
    double Re = (2.*a*u_rel)/nu_mol;

    if (Re < 1.) {
        return 24./Re;
    }
    else if (Re < 800.) {
        return 24./pow(Re,0.6);
    }
    else {
        return 0.44;
    }
}

__device__ __host__
inline double calc_t_s(Prims wd, Prims wg, double a, double rho_m, double cs, double mu) {

    double mfp = mu*m_H/(wg.rho*2e-15);

    if (a<2.25*mfp) 
        return 0.6266570686577501 * rho_m * a / (wg.rho*cs) ;
    else {
        double u_rel = sqrt((wd.v_phi-wg.v_phi)*(wd.v_phi-wg.v_phi) 
                            + (wd.v_R-wg.v_R)*(wd.v_R-wg.v_R) + (wd.v_Z-wg.v_Z)*(wd.v_Z-wg.v_Z));
        return 2.6666666666666667 * rho_m * a / (wg.rho * calc_C_D_step(a,wg.rho,cs,u_rel,mu,mfp) * u_rel);
    }
}

__device__ __host__
inline double calc_t_s(Prims1D wd, Prims1D wg, double a, double rho_m, double cs, double mu, double Om) {

    double rho_g = wg.Sig/(2.506628275 * cs/Om);
    double mfp = mu*m_H/(rho_g*2e-15);

    if (a<2.25*mfp) {
        return 0.6266570686577501 * rho_m * a / (rho_g*cs) ;
    }
    else {
        double u_rel = sqrt((wd.v_R-wg.v_R)*(wd.v_R-wg.v_R) + wd.dv_phi*wd.dv_phi + wd.dv_Z*wd.dv_Z);
        return 2.6666666666666667 * rho_m*a/ (rho_g * calc_C_D_step(a,rho_g,cs,u_rel,mu,mfp) * u_rel);
    }
}



#endif//_CUDISC_HEADERS_DRAG_CONST_H_