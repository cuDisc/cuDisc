#include <iostream>
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>

#include "advection.h"
#include "diffusion_device.h"
#include "field.h"
#include "grid.h"
#include "reductions.h"
#include "dustdynamics.h"
#include "scan.h"
#include "constants.h"
#include "utils.h"
#include "sources.h"
#include "van_leer.h"
#include "icevapour.h"

// Advection-Diffusion Solver


__global__
void _set_boundaries(GridRef g, Field3DRef<Prims> w, int bound, double floor) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.z * blockDim.z ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) {
            for (int k=kidx; k<w.Nd; k+=kstride) {

                if (i < g.Nghost) {
                    if (bound & BoundaryFlags::open_R_inner) {  //outflow
                        if (w(g.Nghost,j,k).v_R < 0.) {
                            w(i,j,k) = w(g.Nghost,j,k) ;
                        }
                        else {
                            w(i,j,k) = w(2*g.Nghost-1-i,j,k) ;
                            w(i,j,k).v_R *= -1 ;
                        }
                    }
                    else {  //reflecting
                        w(i,j,k) = w(2*g.Nghost-1-i,j,k) ;
                        w(i,j,k).v_R *= -1 ;
                    }
                }

                if (j>=g.Nphi+g.Nghost) {
                    if (bound & BoundaryFlags::open_Z_outer) {
                        if (w(i,g.Nphi+g.Nghost-1,k).v_R*g.face_normal_Z(i,g.Nphi+g.Nghost).R + 
                            w(i,g.Nphi+g.Nghost-1,k).v_Z*g.face_normal_Z(i,g.Nphi+g.Nghost).Z > 0.) {
                                
                            w(i,j,k) = w(i,g.Nphi+g.Nghost-1,k) ;
                        }
                        else {
                            w(i,j,k)[0] = w(i,2*(g.Nphi+g.Nghost)-1-j,k)[0];
                            w(i,j,k)[1] = w(i,2*(g.Nphi+g.Nghost)-1-j,k)[1] * (g.cos_th(g.Nphi+g.Nghost)*g.cos_th(g.Nphi+g.Nghost) - g.sin_th(g.Nphi+g.Nghost)*g.sin_th(g.Nphi+g.Nghost)) 
                                            + 2.*w(i,2*(g.Nphi+g.Nghost)-1-j,k)[3]*g.sin_th(g.Nphi+g.Nghost)*g.cos_th(g.Nphi+g.Nghost);
                            w(i,j,k)[2] = w(i,2*(g.Nphi+g.Nghost)-1-j,k)[2];
                            w(i,j,k)[3] = w(i,2*(g.Nphi+g.Nghost)-1-j,k)[3] * (-g.cos_th(g.Nphi+g.Nghost)*g.cos_th(g.Nphi+g.Nghost) + g.sin_th(g.Nphi+g.Nghost)*g.sin_th(g.Nphi+g.Nghost))
                                        + 2.*w(i,2*(g.Nphi+g.Nghost)-1-j,k)[1]*g.sin_th(g.Nphi+g.Nghost)*g.cos_th(g.Nphi+g.Nghost);                           
                        }
                    }
                    else {
                        w(i,j,k)[0] = w(i,2*(g.Nphi+g.Nghost)-1-j,k)[0];
                        w(i,j,k)[1] = w(i,2*(g.Nphi+g.Nghost)-1-j,k)[1] * (g.cos_th(g.Nphi+g.Nghost)*g.cos_th(g.Nphi+g.Nghost) - g.sin_th(g.Nphi+g.Nghost)*g.sin_th(g.Nphi+g.Nghost)) 
                                        + 2.*w(i,2*(g.Nphi+g.Nghost)-1-j,k)[3]*g.sin_th(g.Nphi+g.Nghost)*g.cos_th(g.Nphi+g.Nghost);
                        w(i,j,k)[2] = w(i,2*(g.Nphi+g.Nghost)-1-j,k)[2];
                        w(i,j,k)[3] = w(i,2*(g.Nphi+g.Nghost)-1-j,k)[3] * (-g.cos_th(g.Nphi+g.Nghost)*g.cos_th(g.Nphi+g.Nghost) + g.sin_th(g.Nphi+g.Nghost)*g.sin_th(g.Nphi+g.Nghost))
                                        + 2.*w(i,2*(g.Nphi+g.Nghost)-1-j,k)[1]*g.sin_th(g.Nphi+g.Nghost)*g.cos_th(g.Nphi+g.Nghost);
                    }
                }        

                if (i>=g.NR+g.Nghost) {
                    if (bound & BoundaryFlags::open_R_outer) {
                        if (w(g.NR+g.Nghost-1,j,k).v_R > 0.) {
                            w(i,j,k) = w(g.NR+g.Nghost-1,j,k);
                        }
                        else {
                            w(i,j,k) = w(g.NR+g.Nghost-1,j,k);
                            w(i,j,k).v_R *= -1 ;
                        }
                    }
                    else {
                        w(i,j,k) = w(g.NR+g.Nghost-1,j,k);
                        w(i,j,k).v_R *= -1 ;
                    }
                }    
                
                if (j < g.Nghost) {
                    if (bound & BoundaryFlags::open_Z_inner) {  
                        if (w(i,g.Nghost,k).v_R*g.face_normal_Z(i,g.Nghost).R + 
                            w(i,g.Nghost,k).v_Z*g.face_normal_Z(i,g.Nghost).Z < 0.) {

                            w(i,j,k) = w(i,g.Nghost,k);
                        }
                        else {
                            w(i,j,k)[0] = w(i,2*g.Nghost-1-j,k)[0];
                            w(i,j,k)[1] = w(i,2*g.Nghost-1-j,k)[1] * (g.cos_th(g.Nghost)*g.cos_th(g.Nghost) - g.sin_th(g.Nghost)*g.sin_th(g.Nghost)) 
                                            + 2.*w(i,2*g.Nghost-1-j,k)[3]*g.sin_th(g.Nghost)*g.cos_th(g.Nghost);
                            w(i,j,k)[2] = w(i,2*g.Nghost-1-j,k)[2];
                            w(i,j,k)[3] = w(i,2*g.Nghost-1-j,k)[3] * (-g.cos_th(g.Nghost)*g.cos_th(g.Nghost) + g.sin_th(g.Nghost)*g.sin_th(g.Nghost))
                                        + 2.*w(i,2*g.Nghost-1-j,k)[1]*g.sin_th(g.Nghost)*g.cos_th(g.Nghost);                         
                        }
                        
                    }
                    else {  
                        w(i,j,k)[0] = w(i,2*g.Nghost-1-j,k)[0];
                        w(i,j,k)[1] = w(i,2*g.Nghost-1-j,k)[1] * (g.cos_th(g.Nghost)*g.cos_th(g.Nghost) - g.sin_th(g.Nghost)*g.sin_th(g.Nghost)) 
                                        + 2.*w(i,2*g.Nghost-1-j,k)[3]*g.sin_th(g.Nghost)*g.cos_th(g.Nghost);
                        w(i,j,k)[2] = w(i,2*g.Nghost-1-j,k)[2];
                        w(i,j,k)[3] = w(i,2*g.Nghost-1-j,k)[3] * (-g.cos_th(g.Nghost)*g.cos_th(g.Nghost) + g.sin_th(g.Nghost)*g.sin_th(g.Nghost))
                                        + 2.*w(i,2*g.Nghost-1-j,k)[1]*g.sin_th(g.Nghost)*g.cos_th(g.Nghost);
                        // w(i,j,k)[1] = w(i,2*g.Nghost-1-j,k)[1];
                        // w(i,j,k)[2] = w(i,2*g.Nghost-1-j,k)[2];
                        // w(i,j,k)[3] = -w(i,2*g.Nghost-1-j,k)[3];
                    }
                }    
            }
        }
    }
}


__global__
void _calc_conserved(GridRef g, Field3DRef<Quants> q, Field3DRef<Prims> w) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.z * blockDim.z ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) {   
            for (int k=kidx; k<q.Nd; k+=kstride) {
                q(i,j,k).rho = w(i,j,k).rho;
                q(i,j,k).mom_R = w(i,j,k).v_R * w(i,j,k).rho;
                q(i,j,k).amom_phi = w(i,j,k).v_phi * w(i,j,k).rho * g.Rc(i);
                q(i,j,k).mom_Z = w(i,j,k).v_Z * w(i,j,k).rho;
            } 
        }
    }
}

__global__
void _calc_prim(GridRef g, Field3DRef<Quants> q, Field3DRef<Prims> w) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.z * blockDim.z ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) {   
            for (int k=kidx; k<q.Nd; k+=kstride) {
                w(i,j,k).rho = q(i,j,k).rho;
                w(i,j,k).v_R = q(i,j,k).mom_R/q(i,j,k).rho;
                w(i,j,k).v_phi = q(i,j,k).amom_phi/(q(i,j,k).rho * g.Rc(i));
                w(i,j,k).v_Z = q(i,j,k).mom_Z/q(i,j,k).rho;
            } 
        }
    }
}

__global__
void _floor_prim(GridRef g, Field3DRef<Prims> w, FieldConstRef<Prims> w_gas, double floor) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.z * blockDim.z ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) {   
            for (int k=kidx; k<w.Nd; k+=kstride) {
                double f = w(i,j,k).rho / (floor * w_gas(i,j).rho) ;
                if (f < 1.1) {
                    w(i,j,k).rho   = w_gas(i,j).rho * floor;
                    w(i,j,k).v_R   = 0 ;
                    w(i,j,k).v_phi = w_gas(i,j).v_phi;
                    w(i,j,k).v_Z   = 0 ; 
                }
            } 
        }
    }
}

__device__
double compute_diff_fluxR(GridRef& g, Field3DConstRef<double>& D, Field3DConstRef<Prims>& w, FieldConstRef<Prims>& w_gas, 
                            FieldConstRef<double>& cs, int i, int j, int k, double gas_floor) {

    if (i < g.Nghost+1 || i > g.NR+g.Nghost-1) { return 0.; }
    else if ((w_gas(i-1,j).rho <= 1.1*gas_floor || w_gas(i,j).rho <= 1.1*gas_floor)) {return 0.;}


    fluxes fi, fj; 
    
    fi = left_side_flux(g, D, i, j, k) ;
    double flux = 0.;

    if (w_gas(i,j+1).rho <= 1.1*gas_floor || w_gas(i-1,j-1).rho <= 1.1*gas_floor) {
        flux += - fi.R.R * (w(i,j,k).rho/w_gas(i,j).rho - w(i-1,j,k).rho/w_gas(i-1,j).rho);
    }
    else {
        flux += - (1 - fi.w.R) * fi.R.R * (w(i,j,k).rho/w_gas(i,j).rho - w(i-1,j,k).rho/w_gas(i-1,j).rho); 
        
        flux += - (1 - fi.w.R) * fi.R.Z * (w(i,j,k).rho/w_gas(i,j).rho - w(i,j+1,k).rho/w_gas(i,j+1).rho);

        fj = right_side_flux(g, D, i-1, j, k) ;


        flux += - fj.w.R * fj.R.R * (w(i,j,k).rho/w_gas(i,j).rho - w(i-1,j,k).rho/w_gas(i-1,j).rho); 

        flux += - fj.w.R * fj.R.Z * (w(i-1,j-1,k).rho/w_gas(i-1,j-1).rho - w(i-1,j,k).rho/w_gas(i-1,j).rho);
    }
    double Flim = 0.5*(w(i,j,k).rho+w(i-1,j,k).rho) * cs(i,j);

    flux = flux * pow(1+pow(flux/Flim,10.),-1./10.);

    return flux;
}

__device__
double compute_diff_fluxZ(GridRef& g, Field3DConstRef<double>& D, Field3DConstRef<Prims>& w, FieldConstRef<Prims>& w_gas, 
                            FieldConstRef<double>& cs, int i, int j, int k, double gas_floor) {

    if (j < g.Nghost+1 || j > g.Nphi+g.Nghost-1) { return 0.; }
    else if ((w_gas(i,j-1).rho <= 1.1*gas_floor || w_gas(i,j).rho <= 1.1*gas_floor)) {return 0.;}


    fluxes fi, fj; 
    
    fi = right_side_flux(g, D, i, j, k) ;
    double flux = 0.;

    if (w_gas(i+1,j).rho <= 1.1*gas_floor || w_gas(i-1,j-1).rho <= 1.1*gas_floor) {
        flux += - fi.Z.Z * (w(i,j,k).rho/w_gas(i,j).rho - w(i,j-1,k).rho/w_gas(i,j-1).rho);        
    }
    else {
        flux += - (1-fi.w.Z) * fi.Z.R * (w(i,j,k).rho/w_gas(i,j).rho - w(i+1,j,k).rho/w_gas(i+1,j).rho);
        flux += - (1-fi.w.Z) * fi.Z.Z * (w(i,j,k).rho/w_gas(i,j).rho - w(i,j-1,k).rho/w_gas(i,j-1).rho);

        fj = left_side_flux(g, D, i, j-1, k) ;

        flux += - fj.w.Z * fj.Z.Z * (w(i,j,k).rho/w_gas(i,j).rho - w(i,j-1,k).rho/w_gas(i,j-1).rho);
        flux += - fj.w.Z * fj.Z.R * (w(i-1,j-1,k).rho/w_gas(i-1,j-1).rho - w(i,j-1,k).rho/w_gas(i,j-1).rho);
    }
    double Flim = 0.5*(w(i,j,k).rho+w(i,j-1,k).rho) * cs(i,j);

    flux = flux * pow(1+pow(flux/Flim,10.),-1./10.);

    return flux;
}

__device__ __host__ inline
Quants construct_fluxes(double v_l, double v_r, double v_av, double w_l[4], double w_r[4]) {

    if (v_l <= 0 && v_r >= 0) {
        return {0.,0.,0.,0.};
    }
    else {

        if (v_av > 0.) {   
            double m_l = w_l[0] * v_l ;
            return {m_l, w_l[1] * m_l, w_l[2] * m_l, w_l[3] * m_l} ;
        }

        else if (v_av < 0.) {     
            double m_r = w_r[0] * v_r ;
            return {m_r, w_r[1] * m_r, w_r[2] * m_r, w_r[3] * m_r} ;
        }

        else if (v_av == 0.) {
            double m_l = w_l[0] * v_l ;
            double m_r = w_r[0] * v_r ;
            return {0.5*(       m_l +        m_r), 0.5*(w_l[1]*m_l + w_r[1]*m_r), 
                    0.5*(w_l[2]*m_l + w_r[2]*m_r), 0.5*(w_l[3]*m_l + w_r[3]*m_r)};
        }
    }
}


__device__ __host__ inline
void add_diffive_fluxes(double w_l[4], double w_r[4], Quants& flux, double diff_flux) {

    if (diff_flux > 0) {
        flux.rho += diff_flux;
        flux.mom_R += w_l[1]*diff_flux; 
        flux.amom_phi += w_l[2]*diff_flux;
        flux.mom_Z += w_l[3]*diff_flux;
    }
    else {
        flux.rho += diff_flux;
        flux.mom_R += w_r[1]*diff_flux; 
        flux.amom_phi += w_r[2]*diff_flux;
        flux.mom_Z += w_r[3]*diff_flux;
    }
}

template<bool do_diffusion>
__device__ __host__ inline
void dust_fluxR(GridRef& g, Field3DConstRef<Prims>& w, int i, int j, int k, Field3DRef<Quants>& fluxR, double diff_fluxR) {

    double normR = g.face_normal_R(i,j).R;
    double normZ = g.face_normal_R(i,j).Z;

    double w_l[4] = {w(i-1,j,k).rho, w(i-1,j,k).v_R, w(i-1,j,k).v_phi, w(i-1,j,k).v_Z};
    double w_r[4] = {w(i,j,k).rho, w(i,j,k).v_R, w(i,j,k).v_phi, w(i,j,k).v_Z};     

    w_l[2] *= g.Re(i) ;
    w_r[2] *= g.Re(i) ;

    double v_l, v_r ;

    v_l = w_l[1] * normR + w_l[3] * normZ;
    v_r = w_r[1] * normR + w_r[3] * normZ; 

    double rhorat = std::sqrt(w_r[0]/w_l[0]);
    double v_av = (v_l + rhorat * v_r) / (1. + rhorat);

    // Construct fluxes depending on sign of interface velocities

    fluxR(i,j,k) = construct_fluxes(v_l, v_r, v_av, w_l, w_r);
    if (do_diffusion) 
        add_diffive_fluxes(w_l, w_r, fluxR(i,j,k), diff_fluxR);
}

template<bool do_diffusion>
__device__ __host__ inline
void dust_fluxZ(GridRef& g, Field3DConstRef<Prims>& w, int i, int j, int k, Field3DRef<Quants>& fluxZ, double diff_fluxZ) {

    double normR = g.face_normal_Z(i,j).R;
    double normZ = g.face_normal_Z(i,j).Z;

    double w_l[4] = {w(i,j-1,k).rho, w(i,j-1,k).v_R, w(i,j-1,k).v_phi, w(i,j-1,k).v_Z};
    double w_r[4] = {w(i,j,k).rho, w(i,j,k).v_R, w(i,j,k).v_phi, w(i,j,k).v_Z};  

    w_l[2] *= g.Rc(i) ;
    w_r[2] *= g.Rc(i) ;

    double v_l, v_r ;
    
    v_l = w_l[1] * normR + w_l[3] * normZ;
    v_r = w_r[1] * normR + w_r[3] * normZ; 

    double rhorat = std::sqrt(w_r[0]/w_l[0]);
    double v_av = (v_l + rhorat * v_r) / (1. + rhorat);

    fluxZ(i,j,k) = construct_fluxes(v_l, v_r, v_av, w_l, w_r);
    if (do_diffusion) 
        add_diffive_fluxes(w_l, w_r, fluxZ(i,j,k), diff_fluxZ);
}

template<bool do_diffusion>
__device__ __host__ inline
void dust_flux_vlR(GridRef& g, Field3DConstRef<Prims>& w, int i, int j, int k, Field3DRef<Quants>& fluxR, double diff_fluxR) {

    double normR = g.face_normal_R(i,j).R;
    double normZ = g.face_normal_R(i,j).Z;
    double dR_l = g.Re(i)-g.Rc(i-1);
    double dR_r = g.Re(i)-g.Rc(i);

    double w_l[4] = {w(i-1,j,k).rho + vl_R(g,w,i-1,j,k,0)*dR_l, w(i-1,j,k).v_R + vl_R(g,w,i-1,j,k,1)*dR_l, 
                w(i-1,j,k).v_phi + vl_R(g,w,i-1,j,k,2)*dR_l, w(i-1,j,k).v_Z + vl_R(g,w,i-1,j,k,3)*dR_l};

    double w_r[4] = {w(i,j,k).rho + vl_R(g,w,i,j,k,0)*dR_r, w(i,j,k).v_R + vl_R(g,w,i,j,k,1)*dR_r, 
                w(i,j,k).v_phi + vl_R(g,w,i,j,k,2)*dR_r, w(i,j,k).v_Z + vl_R(g,w,i,j,k,3)*dR_r};

    w_l[2] *= g.Re(i) ;
    w_r[2] *= g.Re(i) ;
    
    double v_l, v_r;

    v_l = w_l[1] * normR + w_l[3] * normZ;
    v_r = w_r[1] * normR + w_r[3] * normZ; 

    double rhorat = std::sqrt(w_r[0]/w_l[0]);
    double v_av = (v_l + rhorat * v_r) / (1. + rhorat);

    // Construct fluxes depending on sign of interface velocities

    fluxR(i,j,k) = construct_fluxes(v_l, v_r, v_av, w_l, w_r);
    if(do_diffusion)
        add_diffive_fluxes(w_l, w_r, fluxR(i,j,k), diff_fluxR);
}

template<bool do_diffusion>
__device__ __host__ inline
void dust_flux_vlZ(GridRef& g, Field3DConstRef<Prims>& w, int i, int j, int k, Field3DRef<Quants>& fluxZ, double diff_fluxZ) {

    double normR = g.face_normal_Z(i,j).R;
    double normZ = g.face_normal_Z(i,j).Z;
    double dZ_l = g.Ze(i,j)-g.Zc(i,j-1);
    double dZ_r = g.Ze(i,j)-g.Zc(i,j);

    double w_l[4] = {w(i,j-1,k).rho + vl_Z(g,w,i,j-1,k,0)*dZ_l, w(i,j-1,k).v_R + vl_Z(g,w,i,j-1,k,1)*dZ_l, 
                w(i,j-1,k).v_phi + vl_Z(g,w,i,j-1,k,2)*dZ_l, w(i,j-1,k).v_Z + vl_Z(g,w,i,j-1,k,3)*dZ_l};

    double w_r[4] = {w(i,j,k).rho + vl_Z(g,w,i,j,k,0)*dZ_r, w(i,j,k).v_R + vl_Z(g,w,i,j,k,1)*dZ_r, 
                w(i,j,k).v_phi + vl_Z(g,w,i,j,k,2)*dZ_r, w(i,j,k).v_Z + vl_Z(g,w,i,j,k,3)*dZ_r};


    w_l[2] *= g.Rc(i) ;
    w_r[2] *= g.Rc(i) ;
 
    double v_l, v_r;

    v_l = w_l[1] * normR + w_l[3] * normZ;
    v_r = w_r[1] * normR + w_r[3] * normZ; 

    double rhorat = std::sqrt(w_r[0]/w_l[0]);
    double v_av = (v_l + rhorat * v_r) / (1. + rhorat);

    // Construct fluxes depending on sign of interface velocities

    fluxZ(i,j,k) = construct_fluxes(v_l, v_r, v_av, w_l, w_r);
    if(do_diffusion)
        add_diffive_fluxes(w_l, w_r, fluxZ(i,j,k), diff_fluxZ);
}


template<bool do_diffusion>
__global__ void _calc_donor_flux(GridRef g, Field3DConstRef<Prims> w, FieldConstRef<Prims> w_gas, FieldConstRef<double> cs,
                                Field3DRef<Quants> fluxR, Field3DRef<Quants> fluxZ, Field3DConstRef<double> D, double gas_floor) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.z * blockDim.z ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost+1; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost+1; j+=jstride) {
            for (int k=kidx; k<w.Nd; k+=kstride) {    
                            
                double diff_flux = 0;
                if (do_diffusion) 
                    diff_flux = compute_diff_fluxR(g, D, w, w_gas, cs, i, j, k, gas_floor);
                dust_fluxR<do_diffusion>(g, w, i, j, k, fluxR, diff_flux);

                if (do_diffusion) 
                    diff_flux = compute_diff_fluxZ(g, D, w, w_gas, cs, i, j, k, gas_floor);
                dust_fluxZ<do_diffusion>(g, w, i, j, k, fluxZ, diff_flux); 
            } 
        }
    }

}

template<bool do_diffusion>
__global__ void _calc_diff_flux_vl(GridRef g, Field3DConstRef<Prims> w, FieldConstRef<Prims> w_gas, FieldConstRef<double> cs,
                                    Field3DRef<Quants> fluxR, Field3DRef<Quants> fluxZ, Field3DConstRef<double> D, double gas_floor) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.z * blockDim.z ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost+1; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost+1; j+=jstride) {
            for (int k=kidx; k<w.Nd; k+=kstride) { 

                double diff_flux = 0;
                if (do_diffusion) 
                    diff_flux = compute_diff_fluxR(g, D, w, w_gas, cs, i, j, k, gas_floor);
                dust_flux_vlR<do_diffusion>(g, w, i, j, k, fluxR, diff_flux);

                if (do_diffusion) 
                    diff_flux = compute_diff_fluxZ(g, D, w, w_gas, cs, i, j, k, gas_floor);
                dust_flux_vlZ<do_diffusion>(g, w, i, j, k, fluxZ, diff_flux); 
            }
        }
    }

}

double calc_massw(Grid& g, Field3D<Prims>& q) {

    double mass=0;

    for (int i=g.Nghost; i<g.NR+g.Nghost; i++) {
        for (int j=g.Nghost; j<g.Nphi+g.Nghost; j++) {
            for (int k=0; k<q.Nd; k++) {
                mass += 4.*M_PI*q(i,j,k).rho * g.volume(i,j);
            }
        }
    }

    return mass;
}
double calc_massq(Grid& g, Field3D<Quants>& q) {

    double mass=0;

    for (int i=g.Nghost; i<g.NR+g.Nghost; i++) {
        for (int j=g.Nghost; j<g.Nphi+g.Nghost; j++) {
            for (int k=0; k<q.Nd; k++) {
                mass += 4.*M_PI*q(i,j,k).rho * g.volume(i,j);
            }
        }
    }

    return mass;
}

__global__ void _update_quants(GridRef g, Field3DRef<Quants> q_mids, Field3DRef<Quants> q, double dt,
                                        Field3DRef<Quants> fluxR, Field3DRef<Quants> fluxZ) {
    
    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.z * blockDim.z ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost; j+=jstride) { 
            for (int k=kidx; k<q.Nd; k+=kstride) {
                for (int l=0; l<4; l++) {
                    double df = (fluxR(i,j,k)[l] * g.area_R(i,j) - fluxR(i+1,j,k)[l] * g.area_R(i+1,j)) 
                            + (fluxZ(i,j,k)[l] * g.area_Z(i,j) - fluxZ(i,j+1,k)[l] * g.area_Z(i,j+1));
                    q_mids(i,j,k)[l] = q(i,j,k)[l] + (dt/g.volume(i,j))*df;
                }
            }
        }
    }
}

__global__ void _update_quants(GridRef g, Field3DRef<Quants> q_mids, Field3DRef<Quants> q, Field3DRef<Quants> q_mids_trac, Field3DRef<Quants> q_trac, double dt,
                                        Field3DRef<Quants> fluxR, Field3DRef<Quants> fluxZ, Field3DRef<Quants> fluxR_trac, Field3DRef<Quants> fluxZ_trac) {
    
    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.z * blockDim.z ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost; j+=jstride) { 
            for (int k=kidx; k<q.Nd; k+=kstride) {
                double df = (fluxR(i,j,k).rho * g.area_R(i,j) - fluxR(i+1,j,k).rho * g.area_R(i+1,j)) 
                            + (fluxZ(i,j,k).rho * g.area_Z(i,j) - fluxZ(i,j+1,k).rho * g.area_Z(i,j+1));
                q_mids(i,j,k).rho = q(i,j,k).rho + (dt/g.volume(i,j))*df;

                df = (fluxR_trac(i,j,k).rho * g.area_R(i,j) - fluxR_trac(i+1,j,k).rho * g.area_R(i+1,j)) 
                            + (fluxZ_trac(i,j,k).rho * g.area_Z(i,j) - fluxZ_trac(i,j+1,k).rho * g.area_Z(i,j+1));
                q_mids_trac(i,j,k).rho = q_trac(i,j,k).rho + (dt/g.volume(i,j))*df;

                double mom_tot;
                
                for (int l=1; l<4; l++) {
                    df = (fluxR(i,j,k)[l] * g.area_R(i,j) - fluxR(i+1,j,k)[l] * g.area_R(i+1,j)) 
                            + (fluxZ(i,j,k)[l] * g.area_Z(i,j) - fluxZ(i,j+1,k)[l] * g.area_Z(i,j+1));

                    mom_tot = q(i,j,k)[l] + (dt/g.volume(i,j))*df;

                    df = (fluxR_trac(i,j,k)[l] * g.area_R(i,j) - fluxR_trac(i+1,j,k)[l] * g.area_R(i+1,j)) 
                            + (fluxZ_trac(i,j,k)[l] * g.area_Z(i,j) - fluxZ_trac(i,j+1,k)[l] * g.area_Z(i,j+1));

                    mom_tot += q_trac(i,j,k)[l] + (dt/g.volume(i,j))*df;
                    
                    q_mids(i,j,k)[l] = mom_tot * q_mids(i,j,k).rho/(q_mids(i,j,k).rho+q_mids_trac(i,j,k).rho);
                    q_mids_trac(i,j,k)[l] = mom_tot * q_mids_trac(i,j,k).rho/(q_mids(i,j,k).rho+q_mids_trac(i,j,k).rho);
                }
            }
        }
    }
}

__global__ void _set_boundary_flux(GridRef g, int bound, Field3DRef<Quants> fluxR, Field3DRef<Quants> fluxZ) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.z * blockDim.z ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) {
            for (int k=kidx; k<fluxR.Nd; k+=kstride) {

                if (i <= g.Nghost) {
                    if (bound & BoundaryFlags::open_R_inner) {  //outflow
                        if (fluxR(i,j,k).rho > 0) // prevent inflow
                            fluxR(i,j,k) = {0.,0.,0.,0.};
                    }
                    else {  //reflecting
                        fluxR(i,j,k) = {0.,0.,0.,0.};
                    }
                }

                if (j>=g.Nphi+g.Nghost) {
                    if (bound & BoundaryFlags::open_Z_outer) {
                        if (fluxZ(i,j,k).rho < 0) // prevent inflow
                            fluxZ(i,j,k) = {0.,0.,0.,0.};
                    }
                    else {
                        fluxZ(i,j,k) = {0.,0.,0.,0.};
                    }
                }        

                if (i>=g.NR+g.Nghost) {
                    if (bound & BoundaryFlags::open_R_outer) {
                        if (fluxR(i,j,k).rho < 0) // prevent inflow
                            fluxR(i,j,k) = {0.,0.,0.,0.};
                    }
                    else {
                        fluxR(i,j,k) = {0.,0.,0.,0.};
                    }
                }    
                
                if (j <= g.Nghost) {
                    if (bound & BoundaryFlags::open_Z_inner) {  
                        if (fluxZ(i,j,k).rho > 0) // prevent inflow
                            fluxZ(i,j,k) = {0.,0.,0.,0.};
                    }
                    else {  
                        fluxZ(i,j,k) = {0.,0.,0.,0.};
                    }
                }    
            }
        }
    }

}

__global__ void set_flux_to_zero(GridRef g, Field3DRef<Quants> flux) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.z * blockDim.z ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) {
            for (int k=kidx; k<flux.Nd; k+=kstride) {
                for (int l=0; l<4; l++) {flux(i,j,k)[l] = 0.;}
            }
        }
    }

}


void DustDynamics::operator() (Grid& g, Field3D<Prims>& w_dust, const Field<Prims>& w_gas, double dt) {

    if (g.Nghost < 2)
        throw std::invalid_argument("Dust dynamics requires at least 2 ghost cells") ;

    Field3D<Quants> q_mids = Field3D<Quants>(g.NR+2*g.Nghost, g.Nphi+2*g.Nghost, w_dust.Nd);
    Field3D<Quants> q = Field3D<Quants>(g.NR+2*g.Nghost, g.Nphi+2*g.Nghost, w_dust.Nd);

    Field3D<Quants> fluxR = Field3D<Quants>(g.NR+2*g.Nghost, g.Nphi+2*g.Nghost, w_dust.Nd);
    Field3D<Quants> fluxZ = Field3D<Quants>(g.NR+2*g.Nghost, g.Nphi+2*g.Nghost, w_dust.Nd);

    dim3 threads(16,8,4) ;
    dim3 blocks((g.NR + 2*g.Nghost+15)/16,(g.Nphi + 2*g.Nghost+7)/8, (q.Nd+3)/4) ;
    //dim3 blocks(4,4,4) ;

    _set_boundaries<<<blocks,threads>>>(g, w_dust, _boundary, _floor);
    check_CUDA_errors("_set_boundaries") ;
    _calc_conserved<<<blocks,threads>>>(g, q, w_dust);
    check_CUDA_errors("_calc_conserved") ;

    // Calc donor cell flux
    if (_DoDiffusion) {
        _calc_donor_flux<true><<<blocks,threads>>>(g, w_dust, w_gas, _cs, fluxR, fluxZ, _D, _gas_floor);
        check_CUDA_errors("_calc_donor_flux") ;
    }
    else {
        _calc_donor_flux<false><<<blocks,threads>>>(g, w_dust, w_gas, _cs, fluxR, fluxZ, _D, _gas_floor);
        check_CUDA_errors("_calc_donor_flux") ;
    }

    // Update quantities a half time step and and source terms.
    _set_boundary_flux<<<blocks,threads>>>(g, _boundary, fluxR, fluxZ);
    check_CUDA_errors("_set_boundary_flux") ;
    _update_quants<<<blocks,threads>>>(g, q_mids, q, dt/2., fluxR, fluxZ);
    check_CUDA_errors("_update_quants") ;
    _sources.source_exp(g, w_dust, q_mids, dt/2.);
    _calc_prim<<<blocks,threads>>>(g, q_mids, w_dust);
    check_CUDA_errors("_calc_prim") ; 
    _sources.source_imp(g, w_dust, dt/2.);
    _floor_prim<<<blocks,threads>>>(g, w_dust, w_gas, _floor);
    check_CUDA_errors("_floor_prim") ;
    
    _set_boundaries<<<blocks,threads>>>(g, w_dust, _boundary, _floor);
    check_CUDA_errors("_set_boundaries") ;

    // Compute fluxes with Van Leer
    if (_DoDiffusion) {
        _calc_diff_flux_vl<true><<<blocks,threads>>>(g, w_dust, w_gas, _cs, fluxR, fluxZ, _D, _gas_floor);
        check_CUDA_errors("_calc_diff_flux_vl") ;
    }
    else {
        _calc_diff_flux_vl<false><<<blocks,threads>>>(g, w_dust, w_gas, _cs, fluxR, fluxZ, _D, _gas_floor);
        check_CUDA_errors("_calc_diff_flux_vl") ;
    }

    // Update quantities a full time step and and source terms.

    _set_boundary_flux<<<blocks,threads>>>(g, _boundary, fluxR, fluxZ);
    check_CUDA_errors("_set_boundary_flux") ;
    // set_flux_to_zero<<<blocks,threads>>>(g, fluxR);
    _update_quants<<<blocks,threads>>>(g, q_mids, q, dt, fluxR, fluxZ);
    check_CUDA_errors("_update_quants") ;
    _sources.source_exp(g, w_dust, q_mids, dt);
    _calc_prim<<<blocks, threads>>>(g, q_mids, w_dust);
    check_CUDA_errors("_calc_prim") ; 
    _sources.source_imp(g, w_dust, dt);
    _floor_prim<<<blocks,threads>>>(g, w_dust, w_gas, _floor);
    check_CUDA_errors("_floor_prim") ;
}


__global__
void _compute_CFL_diff(GridRef g, Field3DConstRef<Prims> w, FieldConstRef<Prims> w_gas, FieldRef<double> CFL_grid, Field3DConstRef<double> D,
                        double CFL_adv, double CFL_diff, double floor) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost; j+=jstride) {
            double CFL_k = 1e308;
            for (int k=0; k<w.Nd; k++) {

                if (w(i,j,k).rho < 10.*w_gas(i,j).rho*floor) { continue; }

                double dtR = abs(g.dRe(i)/w(i,j,k).v_R);
                double dtZ = abs(g.dZe(i,j)/w(i,j,k).v_Z);

                double CFL_RZmin = min(dtR, dtZ);
                CFL_k = min(CFL_k, CFL_adv*CFL_RZmin);

                if (D(i,j,k) != 0) {
                    dtR = abs(g.dRe(i)*g.dRe(i) * w_gas(i,j).rho / D(i,j,k));
                    dtZ = abs(g.dZe(i,j)*g.dZe(i,j) * w_gas(i,j).rho / D(i,j,k));

                    CFL_RZmin = min(dtR, dtZ);
                    CFL_k = min(CFL_k, CFL_diff*CFL_RZmin);
                }
            }
            CFL_grid(i,j) = CFL_k;
        }
    } 
}

__global__
void _compute_CFL_diff(GridRef g, Field3DConstRef<Prims> w, FieldConstRef<Prims> w_gas, FieldRef<double> vap, FieldRef<double> CFL_grid, Field3DConstRef<double> D,
                        double CFL_adv, double CFL_diff, double floor) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost; j+=jstride) {
            double CFL_k = 1e308;
            if (vap(i,j) > 10.*w_gas(i,j).rho*1e-100*floor) {

                double dtR = abs(g.dRe(i)/w_gas(i,j).v_R);
                double dtZ = abs(g.dZe(i,j)/w_gas(i,j).v_Z);

                double CFL_RZmin = min(dtR, dtZ);
                CFL_k = min(CFL_k, CFL_adv*CFL_RZmin);

                if (D(i,j,0) != 0) {
                    dtR = abs(g.dRe(i)*g.dRe(i) * w_gas(i,j).rho / D(i,j,0));
                    dtZ = abs(g.dZe(i,j)*g.dZe(i,j) * w_gas(i,j).rho / D(i,j,0));

                    CFL_RZmin = min(dtR, dtZ);
                    CFL_k = min(CFL_k, CFL_diff*CFL_RZmin);
                }
            }
            for (int k=0; k<w.Nd; k++) {

                if (w(i,j,k).rho > 10.*w_gas(i,j).rho*floor) {

                    double dtR = abs(g.dRe(i)/w(i,j,k).v_R);
                    double dtZ = abs(g.dZe(i,j)/w(i,j,k).v_Z);

                    double CFL_RZmin = min(dtR, dtZ);
                    CFL_k = min(CFL_k, CFL_adv*CFL_RZmin);

                    if (D(i,j,k) != 0) {
                        dtR = abs(g.dRe(i)*g.dRe(i) * w_gas(i,j).rho / D(i,j,k));
                        dtZ = abs(g.dZe(i,j)*g.dZe(i,j) * w_gas(i,j).rho / D(i,j,k));

                        CFL_RZmin = min(dtR, dtZ);
                        CFL_k = min(CFL_k, CFL_diff*CFL_RZmin);
                    }
                }
            }
            CFL_grid(i,j) = CFL_k;
        }
    } 
}

double DustDynamics::get_CFL_limit(const Grid& g, const Field3D<Prims>& w, const Field<Prims>& w_gas) {

    dim3 threads(32,32) ;
    dim3 blocks((g.NR + 2*g.Nghost+31)/32,(g.Nphi + 2*g.Nghost+31)/32) ;

    Field<double> CFL_grid = create_field<double>(g);
    set_all(g, CFL_grid, std::numeric_limits<double>::max());

    _compute_CFL_diff<<<blocks,threads>>>(g, w, w_gas, CFL_grid, _D, _CFL_adv, _CFL_diff, _floor);
    check_CUDA_errors("_compute_CFL_diff") ;
    Reduction::scan_R_min(g, CFL_grid);

    double dt = CFL_grid(g.NR+g.Nghost-1,g.Nghost) ;
    for (int j=g.Nghost; j < g.Nphi+g.Nghost; j++) {
        dt = std::min(dt, CFL_grid(g.NR+g.Nghost-1, j)) ;
    }

    return dt;
}

double DustDynamics::get_CFL_limit(const Grid& g, const Field3D<Prims>& w, const Field<Prims>& w_gas, Molecule& mol) {

    dim3 threads(32,32) ;
    dim3 blocks((g.NR + 2*g.Nghost+31)/32,(g.Nphi + 2*g.Nghost+31)/32) ;

    Field<double> CFL_grid = create_field<double>(g);
    set_all(g, CFL_grid, std::numeric_limits<double>::max());

    _compute_CFL_diff<<<blocks,threads>>>(g, w, w_gas, mol.vap, CFL_grid, _D, _CFL_adv, _CFL_diff, _floor);
    check_CUDA_errors("_compute_CFL_diff") ;
    Reduction::scan_R_min(g, CFL_grid);

    double dt = CFL_grid(g.NR+g.Nghost-1,g.Nghost) ;
    for (int j=g.Nghost; j < g.Nphi+g.Nghost; j++) {
        dt = std::min(dt, CFL_grid(g.NR+g.Nghost-1, j)) ;
    }

    return dt;
}

double DustDynamics::get_CFL_limit_debug(const Grid& g, const Field3D<Prims>& w, const Field<Prims>& w_gas) {

    dim3 threads(32,32) ;
    dim3 blocks((g.NR + 2*g.Nghost+31)/32,(g.Nphi + 2*g.Nghost+31)/32) ;

    Field<double> CFL_grid = create_field<double>(g);
    set_all(g, CFL_grid, std::numeric_limits<double>::max());

    _compute_CFL_diff<<<blocks,threads>>>(g, w, w_gas, CFL_grid, _D, _CFL_adv, _CFL_diff, _floor);
    check_CUDA_errors("_compute_CFL_diff") ;

    double dt = std::numeric_limits<double>::max() ;
    int iind,jind;

    for (int i=g.Nghost; i < g.NR+g.Nghost; i++) {
        for (int j=g.Nghost; j < g.Nphi+g.Nghost; j++) {
            if (CFL_grid(i, j)<dt) {
                dt = CFL_grid(i, j) ;
                iind = i;
                jind = j;
            }
            
        }
    }
    printf("%d %d\n", iind, jind);
    return dt;
}

__global__ void _floor_above(GridRef g, Field3DRef<Prims> w, FieldRef<Prims> w_g, double* h, double _floor) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ; 
    int kstride = gridDim.z * blockDim.z ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) {   
            for (int k=kidx; k<w.Nd; k+=kstride) {
                if (g.Zc(i,j) > h[i] || h[i] == g.Zc(i,g.Nghost)) {
                    w(i,j,k).rho = _floor*w_g(i,j).rho ;
                    w(i,j,k).v_R = 0.;
                    w(i,j,k).v_phi = w_g(i,j).v_phi;
                    w(i,j,k).v_Z = 0.;
                }
            } 
        }
    }
}

void DustDynamics::floor_above(Grid& g, Field3D<Prims>& w_dust, Field<Prims>& w_gas, CudaArray<double>& h) {

    dim3 threads(16,8,8) ;
    dim3 blocks((g.NR + 2*g.Nghost+15)/16,(g.Nphi + 2*g.Nghost+7)/8, (w_dust.Nd+7)/8) ;

    _floor_above<<<blocks,threads>>>(g, w_dust, w_gas, h.get(), _floor);
}

// Ice-vapour dynamics

__global__ void _init_tracer_prims(GridRef g, Field3DRef<Prims> w, FieldConstRef<Prims> wg, Field3DRef<Prims> w_trac, Field3DRef<Prims> w_trac_vap, Field3DRef<double> tracers, MoleculeRef mol) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost; j+=jstride) {

            w_trac_vap(i,j,0).rho = mol.vap(i,j);

            for (int l=1; l<4; l++) {
                w_trac_vap(i,j,0)[l] = wg(i,j)[l];
            }

            for (int k=0; k<w.Nd; k++) {

                w_trac(i,j,k).rho = tracers(i,j,k);
                for (int l=1; l<4; l++) {
                    w_trac(i,j,k)[l] = w(i,j,k)[l];
                }

            }
        }
    }

}

__global__ void _update_tracers(GridRef g, Field3DRef<Prims> w_trac, Field3DRef<double> tracers) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.z * blockDim.z ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost; j+=jstride) {
            for (int k=kidx; k<w_trac.Nd; k+=kstride) {

                tracers(i,j,k) = w_trac(i,j,k).rho;

            }
        }
    }

}

__global__ void _update_tracers(GridRef g, Field3DRef<Quants> w_trac, FieldConstRef<Prims> w_gas, Field3DRef<double> tracers, double floor) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.z * blockDim.z ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost; j+=jstride) {
            for (int k=kidx; k<w_trac.Nd; k+=kstride) {

                double f = w_trac(i,j,k).rho / (floor * w_gas(i,j).rho) ;
                if (f < 1.1) {
                    tracers(i,j,k) = w_gas(i,j).rho * floor;
                }          
                else{
                    tracers(i,j,k) = w_trac(i,j,k).rho;
                }         

            }
        }
    }

}

__global__ void _update_tracer_vap(GridRef g, Field3DRef<Prims> w_trac, MoleculeRef mol) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost; j+=jstride) {

            mol.vap(i,j) = w_trac(i,j,0).rho;
        }
    }

}

void DustDynamics::operator() (Grid& g, Field3D<Prims>& w_dust, const Field<Prims>& w_gas, double dt, Molecule& mol) {

    if (g.Nghost < 2)
        throw std::invalid_argument("Dust dynamics requires at least 2 ghost cells") ;

    Field3D<Quants> q_mids = Field3D<Quants>(g.NR+2*g.Nghost, g.Nphi+2*g.Nghost, w_dust.Nd);
    Field3D<Quants> q = Field3D<Quants>(g.NR+2*g.Nghost, g.Nphi+2*g.Nghost, w_dust.Nd);

    Field3D<Quants> fluxR = Field3D<Quants>(g.NR+2*g.Nghost, g.Nphi+2*g.Nghost, w_dust.Nd);
    Field3D<Quants> fluxZ = Field3D<Quants>(g.NR+2*g.Nghost, g.Nphi+2*g.Nghost, w_dust.Nd);

    dim3 threads(16,8,4) ;
    dim3 blocks((g.NR + 2*g.Nghost+15)/16,(g.Nphi + 2*g.Nghost+7)/8, (q.Nd+3)/4) ;

    dim3 threads2D(32,32,1) ;
    dim3 blocks2D((g.NR + 2*g.Nghost+31)/32,(g.Nphi + 2*g.Nghost+31)/32,1) ;

    dim3 threads_vap(16,32,1) ;
    dim3 blocks_vap((g.NR + 2*g.Nghost+15)/16,(g.Nphi + 2*g.Nghost+31)/32, 1) ;

    // Initialise tracers

    Field3D<Prims> w_trac = create_field3D<Prims>(g, w_dust.Nd);
    Field3D<Prims> w_trac_vap = create_field3D<Prims>(g, 1);
    _init_tracer_prims<<<blocks2D,threads2D>>>(g, w_dust, w_gas, w_trac, w_trac_vap, mol.ice, mol);

    // Update main

    _set_boundaries<<<blocks,threads>>>(g, w_dust, _boundary, _floor);
    _calc_conserved<<<blocks,threads>>>(g, q, w_dust);

    // Calc donor cell flux
    if (_DoDiffusion)
        _calc_donor_flux<true><<<blocks,threads>>>(g, w_dust, w_gas, _cs, fluxR, fluxZ, _D, _gas_floor);
    else
        _calc_donor_flux<false><<<blocks,threads>>>(g, w_dust, w_gas, _cs, fluxR, fluxZ, _D, _gas_floor);

    // Update quantities a half time step and and source terms.
    _set_boundary_flux<<<blocks,threads>>>(g, _boundary, fluxR, fluxZ);
    _update_quants<<<blocks,threads>>>(g, q_mids, q, dt/2., fluxR, fluxZ);
    _sources.source_exp(g, w_dust, q_mids, dt/2.);
    _calc_prim<<<blocks,threads>>>(g, q_mids, w_dust);
    _sources.source_imp(g, w_dust, dt/2.);
    _floor_prim<<<blocks,threads>>>(g, w_dust, w_gas, _floor);
    
    _set_boundaries<<<blocks,threads>>>(g, w_dust, _boundary, _floor);

    // Compute fluxes with Van Leer
    if (_DoDiffusion)
        _calc_diff_flux_vl<true><<<blocks,threads>>>(g, w_dust, w_gas, _cs, fluxR, fluxZ, _D, _gas_floor);
    else
        _calc_diff_flux_vl<false><<<blocks,threads>>>(g, w_dust, w_gas, _cs, fluxR, fluxZ, _D, _gas_floor);

    // Update quantities a full time step and and source terms.

    _set_boundary_flux<<<blocks,threads>>>(g, _boundary, fluxR, fluxZ);

    _update_quants<<<blocks,threads>>>(g, q_mids, q, dt, fluxR, fluxZ);
    _sources.source_exp(g, w_dust, q_mids, dt);
    _calc_prim<<<blocks, threads>>>(g, q_mids, w_dust);
    _sources.source_imp(g, w_dust, dt);
    _floor_prim<<<blocks,threads>>>(g, w_dust, w_gas, _floor);



    // Update tracers

    _set_boundaries<<<blocks,threads>>>(g, w_trac, _boundary,1e-100*_floor);
    _calc_conserved<<<blocks,threads>>>(g, q, w_trac);

    // Calc donor cell flux
    if (_DoDiffusion)
        _calc_donor_flux<true><<<blocks,threads>>>(g, w_trac, w_gas, _cs, fluxR, fluxZ, _D, _gas_floor);
    else
        _calc_donor_flux<false><<<blocks,threads>>>(g, w_trac, w_gas, _cs, fluxR, fluxZ, _D, _gas_floor);

    // Update quantities a half time step and and source terms.
    _set_boundary_flux<<<blocks,threads>>>(g, _boundary, fluxR, fluxZ);
    _update_quants<<<blocks,threads>>>(g, q_mids, q, dt/2., fluxR, fluxZ);
    _sources.source_exp(g, w_trac, q_mids, dt/2.);
    _calc_prim<<<blocks,threads>>>(g, q_mids, w_trac);
    _sources.source_imp(g, w_trac, dt/2.);
    _floor_prim<<<blocks,threads>>>(g, w_trac, w_gas,1e-100*_floor);
    
    _set_boundaries<<<blocks,threads>>>(g, w_trac, _boundary,1e-100*_floor);

    // Compute fluxes with Van Leer
    if (_DoDiffusion)
        _calc_diff_flux_vl<true><<<blocks,threads>>>(g, w_trac, w_gas, _cs, fluxR, fluxZ, _D, _gas_floor);
    else
        _calc_diff_flux_vl<false><<<blocks,threads>>>(g, w_trac, w_gas, _cs, fluxR, fluxZ, _D, _gas_floor);

    // Update quantities a full time step and and source terms.

    _set_boundary_flux<<<blocks,threads>>>(g, _boundary, fluxR, fluxZ);

    _update_quants<<<blocks,threads>>>(g, q_mids, q, dt, fluxR, fluxZ);
    _sources.source_exp(g, w_trac, q_mids, dt);
    _calc_prim<<<blocks, threads>>>(g, q_mids, w_trac);
    _sources.source_imp(g, w_trac, dt);
    _floor_prim<<<blocks,threads>>>(g, w_trac, w_gas,1e-100*_floor);

    _update_tracers<<<blocks,threads>>>(g, w_trac, mol.ice);

    // Vap update

    _set_boundaries<<<blocks_vap, threads_vap>>>(g, w_trac_vap, _boundary, 1e-100*_floor);
    _calc_conserved<<<blocks_vap, threads_vap>>>(g, q, w_trac_vap);

    // Calc donor cell flux
    if (_DoDiffusion)
        _calc_donor_flux<true><<<blocks_vap, threads_vap>>>(g, w_trac_vap, w_gas, _cs, fluxR, fluxZ, _D, _gas_floor);
    else
        _calc_donor_flux<false><<<blocks_vap, threads_vap>>>(g, w_trac_vap, w_gas, _cs, fluxR, fluxZ, _D, _gas_floor);

    // Update quantities a half time step and and source terms.
    _set_boundary_flux<<<blocks,threads>>>(g, _boundary, fluxR, fluxZ);
    _update_quants<<<blocks_vap, threads_vap>>>(g, q_mids, q, dt/2., fluxR, fluxZ);
    _calc_prim<<<blocks_vap, threads_vap>>>(g, q_mids, w_trac_vap);
    _floor_prim<<<blocks_vap, threads_vap>>>(g, w_trac_vap, w_gas,1e-100*_floor);
    
    _set_boundaries<<<blocks_vap, threads_vap>>>(g, w_trac_vap, _boundary,1e-100*_floor);

    // Compute fluxes with Van Leer
    if (_DoDiffusion)
        _calc_diff_flux_vl<true><<<blocks_vap, threads_vap>>>(g, w_trac_vap, w_gas, _cs, fluxR, fluxZ, _D, _gas_floor);
    else
        _calc_diff_flux_vl<false><<<blocks_vap, threads_vap>>>(g, w_trac_vap, w_gas, _cs, fluxR, fluxZ, _D, _gas_floor);

    // Update quantities a full time step and and source terms.

    _set_boundary_flux<<<blocks_vap, threads_vap>>>(g, _boundary, fluxR, fluxZ);

    _update_quants<<<blocks_vap, threads_vap>>>(g, q_mids, q, dt, fluxR, fluxZ);
    _calc_prim<<<blocks_vap, threads_vap>>>(g, q_mids, w_trac_vap);
    _floor_prim<<<blocks_vap, threads_vap>>>(g, w_trac_vap, w_gas,1e-100*_floor);

    _update_tracer_vap<<<blocks2D, threads2D>>>(g, w_trac_vap, mol);

}

__global__ void _copy_dust_vels(GridRef g, Field3DRef<Prims> wd, Field3DRef<Prims> wtrac, Field3DRef<Quants> qtrac) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ; 
    int kstride = gridDim.z * blockDim.z ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) {   
            for (int k=kidx; k<wd.Nd; k+=kstride) {

                wtrac(i,j,k).rho = qtrac(i,j,k).rho;
                for (int l=1; l<4; l++) {
                    wtrac(i,j,k)[l] = wd(i,j,k)[l];
                }
            }
        }
    }
}

void DustDynamics::operator() (Grid& g, Field3D<Prims>& w_dust, const Field<Prims>& w_gas, double dt, Molecule& mol, SizeGridIce& sizes) {

    if (g.Nghost < 2)
        throw std::invalid_argument("Dust dynamics requires at least 2 ghost cells") ;

    Field3D<Quants> q_mids = Field3D<Quants>(g.NR+2*g.Nghost, g.Nphi+2*g.Nghost, w_dust.Nd);
    Field3D<Quants> q = Field3D<Quants>(g.NR+2*g.Nghost, g.Nphi+2*g.Nghost, w_dust.Nd);
    Field3D<Quants> q_mids_trac = Field3D<Quants>(g.NR+2*g.Nghost, g.Nphi+2*g.Nghost, w_dust.Nd);
    Field3D<Quants> q_trac = Field3D<Quants>(g.NR+2*g.Nghost, g.Nphi+2*g.Nghost, w_dust.Nd);

    Field3D<Quants> fluxR = Field3D<Quants>(g.NR+2*g.Nghost, g.Nphi+2*g.Nghost, w_dust.Nd);
    Field3D<Quants> fluxZ = Field3D<Quants>(g.NR+2*g.Nghost, g.Nphi+2*g.Nghost, w_dust.Nd);

    dim3 threads(16,8,4) ;
    dim3 blocks((g.NR + 2*g.Nghost+15)/16,(g.Nphi + 2*g.Nghost+7)/8, (q.Nd+3)/4) ;

    dim3 threads2D(32,32,1) ;
    dim3 blocks2D((g.NR + 2*g.Nghost+31)/32,(g.Nphi + 2*g.Nghost+31)/32,1) ;

    dim3 threads_vap(16,32,1) ;
    dim3 blocks_vap((g.NR + 2*g.Nghost+15)/16,(g.Nphi + 2*g.Nghost+31)/32, 1) ;

    // Initialise tracers

    Field3D<Prims> w_trac = create_field3D<Prims>(g, w_dust.Nd);
    Field3D<Prims> w_trac_vap = create_field3D<Prims>(g, 1);
    _init_tracer_prims<<<blocks2D,threads2D>>>(g, w_dust, w_gas, w_trac, w_trac_vap, mol.ice, mol);

    // Calc dust donor fluxes

    _set_boundaries<<<blocks,threads>>>(g, w_dust, _boundary, _floor);
    _calc_conserved<<<blocks,threads>>>(g, q, w_dust);

    // Calc donor cell flux
    if (_DoDiffusion)
        _calc_donor_flux<true><<<blocks,threads>>>(g, w_dust, w_gas, _cs, fluxR, fluxZ, _D, _gas_floor);
    else
        _calc_donor_flux<false><<<blocks,threads>>>(g, w_dust, w_gas, _cs, fluxR, fluxZ, _D, _gas_floor);

    // Update quantities a half time step and and source terms.
    _set_boundary_flux<<<blocks,threads>>>(g, _boundary, fluxR, fluxZ);
    _update_quants<<<blocks,threads>>>(g, q_mids, q, dt/2., fluxR, fluxZ);

    // Calc tracer donor fluxes

    _set_boundaries<<<blocks,threads>>>(g, w_trac, _boundary,1e-100*_floor);
    _calc_conserved<<<blocks,threads>>>(g, q_trac, w_trac);

    // Calc donor cell flux
    if (_DoDiffusion)
        _calc_donor_flux<true><<<blocks,threads>>>(g, w_trac, w_gas, _cs, fluxR, fluxZ, _D, _gas_floor);
    else
        _calc_donor_flux<false><<<blocks,threads>>>(g, w_trac, w_gas, _cs, fluxR, fluxZ, _D, _gas_floor);

    // Update quantities a half time step and and source terms.
    _set_boundary_flux<<<blocks,threads>>>(g, _boundary, fluxR, fluxZ);
    _update_quants<<<blocks,threads>>>(g, q_mids_trac, q_trac, dt/2., fluxR, fluxZ);

    // Update sizegrid for half-time quantities

    update_sizegrid(g, sizes, q_mids, q_mids_trac);

    // Update sources

    _sources.source_exp(g, w_dust, q_mids, dt/2.);
    _calc_prim<<<blocks,threads>>>(g, q_mids, w_dust);
    _sources.source_imp(g, w_dust, dt/2.);
    _floor_prim<<<blocks,threads>>>(g, w_dust, w_gas, _floor);
    
    _set_boundaries<<<blocks,threads>>>(g, w_dust, _boundary, _floor);

    // Copy dust velocities to tracers
    _copy_dust_vels<<<blocks,threads>>>(g, w_dust, w_trac, q_mids_trac);

    // Compute dust fluxes with Van Leer
    if (_DoDiffusion)
        _calc_diff_flux_vl<true><<<blocks,threads>>>(g, w_dust, w_gas, _cs, fluxR, fluxZ, _D, _gas_floor);
    else
        _calc_diff_flux_vl<false><<<blocks,threads>>>(g, w_dust, w_gas, _cs, fluxR, fluxZ, _D, _gas_floor);

    // Update quantities a full time step

    _set_boundary_flux<<<blocks,threads>>>(g, _boundary, fluxR, fluxZ);

    _update_quants<<<blocks,threads>>>(g, q_mids, q, dt, fluxR, fluxZ);

    // Calc tracer VL fluxes

    _floor_prim<<<blocks,threads>>>(g, w_trac, w_gas,1e-100*_floor);
    
    _set_boundaries<<<blocks,threads>>>(g, w_trac, _boundary,1e-100*_floor);

    if (_DoDiffusion)
        _calc_diff_flux_vl<true><<<blocks,threads>>>(g, w_trac, w_gas, _cs, fluxR, fluxZ, _D, _gas_floor);
    else
        _calc_diff_flux_vl<false><<<blocks,threads>>>(g, w_trac, w_gas, _cs, fluxR, fluxZ, _D, _gas_floor);

    // Update tracer quantities a full time step

    _set_boundary_flux<<<blocks,threads>>>(g, _boundary, fluxR, fluxZ);

    _update_quants<<<blocks,threads>>>(g, q_mids_trac, q_trac, dt, fluxR, fluxZ);

    // Update sizegrid for full-time quantities

    update_sizegrid(g, sizes, q_mids, q_mids_trac);

    // Update sources

    _sources.source_exp(g, w_dust, q_mids, dt);
    _calc_prim<<<blocks, threads>>>(g, q_mids, w_dust);
    _sources.source_imp(g, w_dust, dt);
    _floor_prim<<<blocks,threads>>>(g, w_dust, w_gas, _floor);

    // Update tracers

    _update_tracers<<<blocks,threads>>>(g, q_mids_trac, w_gas, mol.ice, 1e-100*_floor);

    // Vap update

    // cudaFree(&q);
    // cudaFree(&q_mids);
    // cudaFree(&q_trac);
    // cudaFree(&q_mids_trac);
    // cudaFree(&fluxR);
    // cudaFree(&fluxZ);

    Field3D<Quants> q_mids_vap = Field3D<Quants>(g.NR+2*g.Nghost, g.Nphi+2*g.Nghost, 1);
    Field3D<Quants> q_vap = Field3D<Quants>(g.NR+2*g.Nghost, g.Nphi+2*g.Nghost, 1);

    Field3D<Quants> fluxZ_vap = Field3D<Quants>(g.NR+2*g.Nghost, g.Nphi+2*g.Nghost, 1);
    Field3D<Quants> fluxR_vap = Field3D<Quants>(g.NR+2*g.Nghost, g.Nphi+2*g.Nghost, 1);

    _set_boundaries<<<blocks_vap, threads_vap>>>(g, w_trac_vap, _boundary, 1e-100*_floor);
    _calc_conserved<<<blocks_vap, threads_vap>>>(g, q_vap, w_trac_vap);

    // Calc donor cell flux
    if (_DoDiffusion)
        _calc_donor_flux<true><<<blocks_vap, threads_vap>>>(g, w_trac_vap, w_gas, _cs, fluxR_vap, fluxZ_vap, _D, _gas_floor);
    else
        _calc_donor_flux<false><<<blocks_vap, threads_vap>>>(g, w_trac_vap, w_gas, _cs, fluxR_vap, fluxZ_vap, _D, _gas_floor);

    // Update quantities a half time step and and source terms.
    _set_boundary_flux<<<blocks_vap,threads_vap>>>(g, _boundary, fluxR_vap, fluxZ_vap);
    _update_quants<<<blocks_vap, threads_vap>>>(g, q_mids_vap, q_vap, dt/2., fluxR_vap, fluxZ_vap);

    _calc_prim<<<blocks_vap, threads_vap>>>(g, q_mids_vap, w_trac_vap);
    _floor_prim<<<blocks_vap, threads_vap>>>(g, w_trac_vap, w_gas, 1e-100*_floor);

    _set_boundaries<<<blocks_vap, threads_vap>>>(g, w_trac_vap, _boundary, 1e-100*_floor);

    // Compute fluxes with Van Leer
    if (_DoDiffusion)
        _calc_diff_flux_vl<true><<<blocks_vap, threads_vap>>>(g, w_trac_vap, w_gas, _cs, fluxR_vap, fluxZ_vap, _D, _gas_floor);
    else
        _calc_diff_flux_vl<false><<<blocks_vap, threads_vap>>>(g, w_trac_vap, w_gas, _cs, fluxR_vap, fluxZ_vap, _D, _gas_floor);

    // Update quantities a full time step and and source terms.

    _set_boundary_flux<<<blocks_vap, threads_vap>>>(g, _boundary, fluxR_vap, fluxZ_vap);

    // cudaDeviceSynchronize();
    // for (int j=0; j<g.Nphi+2*g.Nghost; j++) {
    //     std::cout << fluxR_vap(2,j,0).rho << ", " ;
    // }
    // std::cout << "\n";
    // for (int j=0; j<g.Nphi+2*g.Nghost; j++) {
    //     std::cout << fluxR_vap(2,j,0).rho << ", " ;
    // }
    // std::cout << "\n";

    _update_quants<<<blocks_vap, threads_vap>>>(g, q_mids_vap, q_vap, dt, fluxR_vap, fluxZ_vap);

    // cudaDeviceSynchronize();
    // for (int j=0; j<g.Nphi+2*g.Nghost; j++) {
    //     std::cout << q_mids_vap(2,j,0).mom_R << ", " ;
    // }
    // std::cout << "\n";
    // for (int j=0; j<g.Nphi+2*g.Nghost; j++) {
    //     std::cout << q_mids_vap(2,j,0).mom_Z << ", " ;
    // }
    // std::cout << "\n";
    // for (int j=0; j<g.Nphi+2*g.Nghost; j++) {
    //     std::cout << q_mids_vap(2,j,0).rho << ", " ;
    // }
    // std::cout << "\n";

    _calc_prim<<<blocks_vap, threads_vap>>>(g, q_mids_vap, w_trac_vap);
    _floor_prim<<<blocks_vap, threads_vap>>>(g, w_trac_vap, w_gas, 1e-100*_floor);

    _update_tracer_vap<<<blocks2D, threads2D>>>(g, w_trac_vap, mol);

}