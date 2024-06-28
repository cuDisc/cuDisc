#ifndef _CUDISC_HEADERS_DRAG_CONST_H_
#define _CUDISC_HEADERS_DRAG_CONST_H_

#include "constants.h"

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
inline double calc_C_D_step(double a, double rho_g, double cs, double u_rel, double mu) {

    double mfp = mu*m_H/(rho_g*2e-15);

    // if (a < 9./4. * mfp) {
    //     return 4.255384324 * cs/u_rel;
    // }

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



#endif//_CUDISC_HEADERS_DRAG_CONST_H_