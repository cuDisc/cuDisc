#ifndef _CUDISC_HEADERS_DRAG_CONST_H_
#define _CUDISC_HEADERS_DRAG_CONST_H_

#include "constants.h"
#include "dustdynamics.h"
#include "dustdynamics1D.h"

__device__ __host__
inline double calc_vC_D(double a, double rho_g, double cs, double u_rel, double mu) {

    // https://www.aanda.org/articles/aa/pdf/2018/03/aa31824-17.pdf

    double mfp = mu*m_H/(rho_g*2e-15);

    double sqrtgamma = pow(1.+2.*mu/5.,0.5);   

    double nu_mol = 1.595769122 * 0.5 * mfp*cs;
    double Re = (2.*a*u_rel)/nu_mol;

    double M = u_rel/cs;
    double G = pow(10.,2.5*pow(Re/312.,0.6688) / (1.+pow(Re/312.,0.6688)));
    double H = 4.6/(1.+M) + 1.7;

    double vC_inc = 12 * nu_mol / a * (1.+0.15*pow(Re,0.687)) + 0.407*u_rel * Re / (Re+8710);
    double vC_D = 2 * u_rel + (vC_inc-2*u_rel)*exp(-1.535*sqrtgamma*G*nu_mol/(a*cs)) + H * exp(-a*cs/nu_mol) * cs / sqrtgamma;
    
    return vC_D;
}

__device__ __host__
inline double calc_vC_D_step(double a, double rho_g, double cs, double u_rel, double mu, double mfp) {

    double nu_mol = 1.595769122 * 0.5 * mfp*cs;
    double Re = (2.*a*u_rel)/nu_mol;

    if (Re < 1.) {
        return 12 * nu_mol / a ;
    }
    else if (Re < 800.) {
        return 24. * u_rel / pow(Re,0.6);
    }
    else {
        return 0.44 * u_rel ;
    }
}

template<bool use_full_stokes=false>
__device__ __host__
inline double calc_t_s(Prims wd, Prims wg, double a, double rho_m, double cs, double mu) {

    double vC_D ;

    if (use_full_stokes) {
        double u_rel = sqrt((wd.v_phi-wg.v_phi)*(wd.v_phi-wg.v_phi) 
                                + (wd.v_R-wg.v_R)*(wd.v_R-wg.v_R) + (wd.v_Z-wg.v_Z)*(wd.v_Z-wg.v_Z));
        vC_D = calc_vC_D(a,wg.rho,cs,u_rel,mu);
    } else {
        double mfp = mu*m_H/(wg.rho*2e-15);
        
        if (a<2.25*mfp) {
            vC_D = 4.255384324 * cs ;
        }
        else {
            double u_rel = sqrt((wd.v_phi-wg.v_phi)*(wd.v_phi-wg.v_phi) 
                                + (wd.v_R-wg.v_R)*(wd.v_R-wg.v_R) + (wd.v_Z-wg.v_Z)*(wd.v_Z-wg.v_Z));
            vC_D = calc_vC_D_step(a,wg.rho,cs,u_rel,mu,mfp);
        }
    }
    
    return 2.6666666666666667 * rho_m*a / (wg.rho * vC_D);
}

template<bool use_full_stokes=false>
__device__ __host__
inline double calc_t_s(Prims1D wd, Prims1D wg, double a, double rho_m, double cs, double mu, double Om) {

    double rho_g = wg.Sig/(2.506628275 * cs/Om);

    double vC_D ;
    if (use_full_stokes) {
        double u_rel = sqrt((wd.v_R-wg.v_R)*(wd.v_R-wg.v_R) + wd.dv_phi*wd.dv_phi + wd.dv_Z*wd.dv_Z);
        vC_D = calc_vC_D(a,rho_g,cs,u_rel,mu) ;
    } 
    else {    
        double mfp = mu*m_H/(rho_g*2e-15);        
        if (a<2.25*mfp) {
            vC_D = 4.255384324 * cs ;
        }
        else {
            double u_rel = sqrt((wd.v_R-wg.v_R)*(wd.v_R-wg.v_R) + wd.dv_phi*wd.dv_phi + wd.dv_Z*wd.dv_Z);
            vC_D = calc_vC_D_step(a,rho_g,cs,u_rel,mu,mfp) ;
        }
    }

    return 2.6666666666666667 * rho_m*a / (rho_g * vC_D);
}



#endif//_CUDISC_HEADERS_DRAG_CONST_H_