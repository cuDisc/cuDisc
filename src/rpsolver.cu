#include <iostream>
#include <cuda_runtime.h>
#include <string>

#include "advection.h"
#include "diffusion_device.h"
#include "field.h"
#include "grid.h"
#include "reductions.h"
#include "rpsolver.h"
#include "scan.h"
#include "constants.h"
#include "utils.h"
#include "dustdynamics.h"
/////////////////////////////////////////////////////////
/// Riemann problem solver //////////////////////////////
/////////////////////////////////////////////////////////


/*   Cells are indexed as follows:
 * 
 *   -----*------------*------------*------------*----- Edge j+2
 *        |            |            |            |
 *        | (i-1, j+1) | ( i , j+1) | (i+1, j+1) |
 *        |            |            |            |  
 *   -----*------------*------------*------------*----- Edge j+1
 *        |            |            |            |
 *        | (i-1,  j ) => ( i ,  j )| (i+1,  j ) |
 *        |            |Fi   /\ Fj  |            | 
 *   -----*------------*-----||-----*------------*----- Edge j
 *        |            |            |            |
 *        | (i-1, j-1) | ( i , j-1) | (i+1, j-1) |
 *        |            |            |            |
 *   -----*------------*------------*------------*----- Edge j-1
 *        |            |            |            |
 *     Edge i-1     Edge i       Edge i+1     Edge i+2   
 */

double _vanleer_slopeCPU(double dQF, double dQB, double cF, double cB) {

    if (dQF*dQB > 0.) {
        double v = dQB/dQF ;
        return dQB * (cF*v + cB) / (v*v + (cF + cB - 2)*v + 1.) ;
    } 
    else {
        return 0. ;
    }
    
}

double vl_RCPU(OrthGrid& g, Field3D<double>& Qty, int i, int j, int k) {

    double Rc = g.Rc(i);

    double cF = (g.Rc(i+1) - Rc) / (g.Re(i+1)-Rc) ;
    double cB = (g.Rc(i-1) - Rc) / (g.Re(i)-Rc) ;

    double dQF = (Qty(i+1, j, k) - Qty(i, j, k)) / (g.Rc(i+1) - Rc) ;
    double dQB = (Qty(i-1, j, k) - Qty(i, j, k)) / (g.Rc(i-1) - Rc) ;

    return _vanleer_slopeCPU(dQF, dQB, cF, cB) ;
}

double vl_ZCPU(OrthGrid& g, Field3D<double>& Qty, int i, int j, int k) {

    double Zc = g.Zc(i,j);

    double cF = (g.Zc(i,j+1) - Zc) / (g.Ze(i,j+1)-Zc) ;
    double cB = (g.Zc(i,j-1) - Zc) / (g.Ze(i,j)-Zc) ;

    double dQF = (Qty(i, j+1, k) - Qty(i, j, k)) / (g.Zc(i,j+1) - Zc) ;
    double dQB = (Qty(i, j-1, k) - Qty(i, j, k)) / (g.Zc(i,j-1) - Zc) ;

    return _vanleer_slopeCPU(dQF, dQB, cF, cB) ;
}

void construct_fluxesCPU(OrthGrid& g, double v_l, double v_r, double v_av, double u_l[4], double u_r[4], Field3D<double>& flux, int i, int j) {

    if (v_l < 0 && v_r > 0) {
        for (int k=0; k<4; k++) {
            flux(i,j,k) = 0.;
        }
    }
    
    if (v_av >= 0.) {  
        for (int k=0; k<4; k++) {
            flux(i,j,k) = u_l[k] * v_l;
        }
    }

    if (v_av < 0.) {     
        for (int k=0; k<4; k++) {
            flux(i,j,k) = u_r[k] * v_r;
        }   
    }
}

void dust_fluxRCPU(OrthGrid& g, Field3D<double>& q, int i, int j, Field3D<double>& fluxR) {

    double u_l[4] = {q(i-1,j,0), q(i-1,j,1), q(i-1,j,2), q(i-1,j,3)};
    double u_r[4] = {q(i,j,0), q(i,j,1), q(i,j,2), q(i,j,3)};     

    double v_l, v_r, rhorat, v_av;

    v_l = (u_l[1] / u_l[0]) * g.face_normal_R(i,j).R + (u_l[3] / u_l[0]) * g.face_normal_R(i,j).Z;
    v_r = (u_r[1] / u_r[0]) * g.face_normal_R(i,j).R + (u_r[3] / u_r[0]) * g.face_normal_R(i,j).Z; 

    rhorat = std::pow(u_r[0], 0.5)/std::pow(u_l[0], 0.5);
    v_av = (v_l + rhorat * v_r) / (1. + rhorat);

    // Construct fluxes depending on sign of interface velocities

    construct_fluxesCPU(g, v_l, v_r, v_av, u_l, u_r, fluxR, i, j);
}

void dust_fluxZCPU(OrthGrid& g, Field3D<double>& q, int i, int j, Field3D<double>& fluxZ) {

    double u_l[4] = {q(i,j-1,0), q(i,j-1,1), q(i,j-1,2), q(i,j-1,3)};
    double u_r[4] = {q(i,j,0), q(i,j,1), q(i,j,2), q(i,j,3)};     

    double v_l, v_r, rhorat, v_av;

    v_l = (u_l[1] / u_l[0]) * g.face_normal_Z(i,j).R + (u_l[3] / u_l[0]) * g.face_normal_Z(i,j).Z;
    v_r = (u_r[1] / u_r[0]) * g.face_normal_Z(i,j).R + (u_r[3] / u_r[0]) * g.face_normal_Z(i,j).Z; 

    rhorat = std::pow(u_r[0], 0.5)/std::pow(u_l[0], 0.5);
    v_av = (v_l + rhorat * v_r) / (1. + rhorat);

    // Construct fluxes depending on sign of interface velocities

    construct_fluxesCPU(g, v_l, v_r, v_av, u_l, u_r, fluxZ, i, j);
}

void dust_flux_vlRCPU(OrthGrid& g, Field3D<double>& w, int i, int j, Field3D<double>& fluxR) {

    double w_l[4] = {w(i-1,j,0) + vl_RCPU(g,w,i-1,j,0)*(g.Re(i)-g.Rc(i-1)), w(i-1,j,1) + vl_RCPU(g,w,i-1,j,1)*(g.Re(i)-g.Rc(i-1)), 
                w(i-1,j,2) + vl_RCPU(g,w,i-1,j,2)*(g.Re(i)-g.Rc(i-1)), w(i-1,j,3) + vl_RCPU(g,w,i-1,j,3)*(g.Re(i)-g.Rc(i-1))};

    double w_r[4] = {w(i,j,0) + vl_RCPU(g,w,i,j,0)*(g.Re(i)-g.Rc(i)), w(i,j,1) + vl_RCPU(g,w,i,j,1)*(g.Re(i)-g.Rc(i)), 
                w(i,j,2) + vl_RCPU(g,w,i,j,2)*(g.Re(i)-g.Rc(i)), w(i,j,3) + vl_RCPU(g,w,i,j,3)*(g.Re(i)-g.Rc(i))};

    double u_l[4] = {w_l[0], w_l[0] * w_l[1], w_l[0] * w_l[2] * g.Re(i), w_l[0] * w_l[3]};
    double u_r[4] = {w_r[0], w_r[0] * w_r[1], w_r[0] * w_r[2] * g.Re(i), w_r[0] * w_r[3]};
    
    double v_l, v_r, rhorat, v_av;

    v_l = w_l[1] * g.face_normal_R(i,j).R + w_l[3] * g.face_normal_R(i,j).Z;
    v_r = w_r[1] * g.face_normal_R(i,j).R + w_r[3] * g.face_normal_R(i,j).Z; 

    rhorat = std::pow(u_r[0], 0.5)/std::pow(u_l[0], 0.5);
    v_av = (v_l + rhorat * v_r) / (1. + rhorat);

    // Construct fluxes depending on sign of interface velocities

    construct_fluxesCPU(g, v_l, v_r, v_av, u_l, u_r, fluxR, i, j);
}

void dust_flux_vlZCPU(OrthGrid& g, Field3D<double>& w, int i, int j, Field3D<double>& fluxZ) {

    double w_l[4] = {w(i,j-1,0) + vl_ZCPU(g,w,i,j-1,0)*(g.Ze(i,j)-g.Zc(i,j-1)), w(i,j-1,1) + vl_ZCPU(g,w,i,j-1,1)*(g.Ze(i,j)-g.Zc(i,j-1)), 
                w(i,j-1,2) + vl_ZCPU(g,w,i,j-1,2)*(g.Ze(i,j)-g.Zc(i,j-1)), w(i,j-1,3) + vl_ZCPU(g,w,i,j-1,3)*(g.Ze(i,j)-g.Zc(i,j-1))};

    double w_r[4] = {w(i,j,0) + vl_ZCPU(g,w,i,j,0)*(g.Ze(i,j)-g.Zc(i,j)), w(i,j,1) + vl_ZCPU(g,w,i,j,1)*(g.Ze(i,j)-g.Zc(i,j)), 
                w(i,j,2) + vl_ZCPU(g,w,i,j,2)*(g.Ze(i,j)-g.Zc(i,j)), w(i,j,3) + vl_ZCPU(g,w,i,j,3)*(g.Ze(i,j)-g.Zc(i,j))};

    double u_l[4] = {w_l[0], w_l[0] * w_l[1], w_l[0] * w_l[2] * g.Rc(i), w_l[0] * w_l[3]};
    double u_r[4] = {w_r[0], w_r[0] * w_r[1], w_r[0] * w_r[2] * g.Rc(i), w_r[0] * w_r[3]};
 
    double v_l, v_r, rhorat, v_av;

    v_l = w_l[1] * g.face_normal_Z(i,j).R + w_l[3] * g.face_normal_Z(i,j).Z;
    v_r = w_r[1] * g.face_normal_Z(i,j).R + w_r[3] * g.face_normal_Z(i,j).Z; 

    rhorat = std::pow(u_r[0], 0.5)/std::pow(u_l[0], 0.5);
    v_av = (v_l + rhorat * v_r) / (1. + rhorat);

    // Construct fluxes depending on sign of interface velocities

    construct_fluxesCPU(g, v_l, v_r, v_av, u_l, u_r, fluxZ, i, j);
}

void set_boundariesCPU(OrthGrid& g, Field3D<double>& q, std::string bound) {

    if (bound == "periodic") {

        for (int j=g.Nghost; j<g.NZ+g.Nghost; j++) {
            for (int k=0; k<4; k++) {
                for (int i=0; i<g.Nghost; i++) { 
                    q(i,j,k) = q(g.NR+i,j,k);
                    q(g.NR+g.Nghost+i,j,k) = q(g.Nghost+i,j,k);
                }  
            }  
        }

        for (int i=g.Nghost; i<g.NR+g.Nghost; i++) {
            for (int k=0; k<4; k++) {
                for (int j=0; j<g.Nghost; j++) { 
                    q(i,j,k) = q(i,g.NZ+j,k);
                    q(i,g.NZ+g.Nghost+j,k) = q(i,g.Nghost+j,k);
                } 
            }  
        }
        for (int k=0; k<4; k++) {
            q(g.Nghost-1,g.Nghost-1,k) = q(g.NR+g.Nghost-1,g.NZ+g.Nghost-1,k);
            q(g.Nghost-1,g.NZ+g.Nghost,k) = q(g.NR+g.Nghost-1,g.Nghost,k);
            q(g.NR+g.Nghost,g.Nghost-1,k) = q(g.Nghost,g.NZ+g.Nghost-1,k);
            q(g.NR+g.Nghost,g.NZ+g.Nghost,k) = q(g.Nghost,g.Nghost,k);
        }
    }

    if (bound == "outflow") {

        for (int j=g.Nghost; j<g.NZ+g.Nghost; j++) {
            for (int k=0; k<4; k++) {
                for (int i=0; i<g.Nghost; i++) { 
                    q(i,j,k) = q(g.Nghost,j,k);
                    q(g.NR+g.Nghost+i,j,k) = q(g.NR+g.Nghost-1,j,k);
                }
            }  
        }

        for (int i=g.Nghost; i<g.NR+g.Nghost; i++) {
            for (int k=0; k<4; k++) {
                for (int j=0; j<g.Nghost; j++) { 
                    q(i,j,k) = q(i,g.Nghost,k);
                    q(i,g.NZ+g.Nghost+j,k) = q(i,g.NZ+g.Nghost-1,k);
                }
            }  
        }

        for (int k=0; k<4; k++) {
            q(g.Nghost-1,g.Nghost-1,k) = q(g.Nghost,g.Nghost,k);
            q(g.Nghost-1,g.NZ+g.Nghost,k) = q(g.Nghost,g.NZ+g.Nghost-1,k);
            q(g.NR+g.Nghost,g.Nghost-1,k) = q(g.NR+g.Nghost-1,g.Nghost,k);
            q(g.NR+g.Nghost,g.NZ+g.Nghost,k) = q(g.NR+g.Nghost-1,g.NZ+g.Nghost-1,k);
        }

    }

}


// CUDA function

__device__
double _vanleer_slope(double dQF, double dQB, double cF, double cB) {

    if (dQF*dQB > 0.) {
        double v = dQB/dQF ;
        return dQB * (cF*v + cB) / (v*v + (cF + cB - 2)*v + 1.) ;
    } 
    else {
        return 0. ;
    }
    
}

__device__
double vl_R(GridRef& g, Field3DConstRef<Quants>& Qty, int i, int j, int k, int qind) {

    double Rc = g.Rc(i);

    double cF = (g.Rc(i+1) - Rc) / (g.Re(i+1)-Rc) ;
    double cB = (g.Rc(i-1) - Rc) / (g.Re(i)-Rc) ;

    double dQF = (Qty(i+1, j, k)[qind] - Qty(i, j, k)[qind]) / (g.Rc(i+1) - Rc) ;
    double dQB = (Qty(i-1, j, k)[qind] - Qty(i, j, k)[qind]) / (g.Rc(i-1) - Rc) ;

    return _vanleer_slope(dQF, dQB, cF, cB) ;
}

__device__
double vl_Z(GridRef& g, Field3DConstRef<Quants>& Qty, int i, int j, int k, int qind) {

    double Zc = g.Zc(i,j);

    double cF = (g.Zc(i,j+1) - Zc) / (g.Ze(i,j+1)-Zc) ;
    double cB = (g.Zc(i,j-1) - Zc) / (g.Ze(i,j)-Zc) ;

    double dQF = (Qty(i, j+1, k)[qind] - Qty(i, j, k)[qind]) / (g.Zc(i,j+1) - Zc) ;
    double dQB = (Qty(i, j-1, k)[qind] - Qty(i, j, k)[qind]) / (g.Zc(i,j-1) - Zc) ;

    return _vanleer_slope(dQF, dQB, cF, cB) ;
}

__device__
void construct_fluxes(double v_l, double v_r, double v_av, 
                  double u_l[4], double u_r[4], Field3DRef<Quants>& flux, int i, int j, int k) {

    if (v_l < 0 && v_r > 0) {
        flux(i,j,k) = {0.,0.,0.,0.};
    }
    
    else if (v_av > 0.) {   
        flux(i,j,k) = {u_l[0] * v_l, u_l[1] * v_l, u_l[2] * v_l, u_l[3] * v_l};
    }

    else if (v_av < 0.) {     
        flux(i,j,k) = {u_r[0] * v_r, u_r[1] * v_r, u_r[2] * v_r, u_r[3] * v_r};
    }

    else if (v_av == 0.) {
        flux(i,j,k) = {0.5*(u_l[0]*v_l + u_r[0]*v_r), 0.5*(u_l[1]*v_l + u_r[1]*v_r), 
                        0.5*(u_l[2]*v_l + u_r[2]*v_r), 0.5*(u_l[3]*v_l + u_r[3]*v_r)};
    }
}

__device__
void dust_fluxR(GridRef& g, Field3DConstRef<Quants>& w, int i, int j, int k, Field3DRef<Quants>& fluxR) {

    double normR = g.face_normal_R(i,j).R;
    double normZ = g.face_normal_R(i,j).Z;

    double w_l[4] = {w(i-1,j,k).rho, w(i-1,j,k).mom_R, w(i-1,j,k).amom_phi, w(i-1,j,k).mom_Z};
    double w_r[4] = {w(i,j,k).rho, w(i,j,k).mom_R, w(i,j,k).amom_phi, w(i,j,k).mom_Z};     

    double u_l[4] = {w_l[0], w_l[0] * w_l[1], w_l[0] * w_l[2] * g.Re(i), w_l[0] * w_l[3]};
    double u_r[4] = {w_r[0], w_r[0] * w_r[1], w_r[0] * w_r[2] * g.Re(i), w_r[0] * w_r[3]};

    double v_l, v_r, rhorat, v_av;

    v_l = w_l[1] * normR + w_l[3] * normZ;
    v_r = w_r[1] * normR + w_r[3] * normZ; 

    rhorat = std::pow(w_r[0], 0.5)/std::pow(w_l[0], 0.5);
    v_av = (v_l + rhorat * v_r) / (1. + rhorat);

    // Construct fluxes depending on sign of interface velocities

    construct_fluxes(v_l, v_r, v_av, u_l, u_r, fluxR, i, j, k);
}

__device__
void dust_fluxZ(GridRef& g, Field3DConstRef<Quants>& w, int i, int j, int k, Field3DRef<Quants>& fluxZ) {

    double normR = g.face_normal_Z(i,j).R;
    double normZ = g.face_normal_Z(i,j).Z;

    double w_l[4] = {w(i,j-1,k).rho, w(i,j-1,k).mom_R, w(i,j-1,k).amom_phi, w(i,j-1,k).mom_Z};
    double w_r[4] = {w(i,j,k).rho, w(i,j,k).mom_R, w(i,j,k).amom_phi, w(i,j,k).mom_Z};  

    double u_l[4] = {w_l[0], w_l[0] * w_l[1], w_l[0] * w_l[2] * g.Rc(i), w_l[0] * w_l[3]};
    double u_r[4] = {w_r[0], w_r[0] * w_r[1], w_r[0] * w_r[2] * g.Rc(i), w_r[0] * w_r[3]};   

    double v_l, v_r, rhorat, v_av;

    v_l = w_l[1] * normR + w_l[3] * normZ;
    v_r = w_r[1] * normR + w_r[3] * normZ; 

    rhorat = std::pow(w_r[0], 0.5)/std::pow(w_l[0], 0.5);
    v_av = (v_l + rhorat * v_r) / (1. + rhorat);

    // Construct fluxes depending on sign of interface velocities

    construct_fluxes(v_l, v_r, v_av, u_l, u_r, fluxZ, i, j, k);
}

__device__
void dust_flux_vlR(GridRef& g, Field3DConstRef<Quants>& w, int i, int j, int k, Field3DRef<Quants>& fluxR) {

    double normR = g.face_normal_R(i,j).R;
    double normZ = g.face_normal_R(i,j).Z;
    double dR_l = g.Re(i)-g.Rc(i-1);
    double dR_r = g.Re(i)-g.Rc(i);

    double w_l[4] = {w(i-1,j,k).rho + vl_R(g,w,i-1,j,k,0)*dR_l, w(i-1,j,k).mom_R + vl_R(g,w,i-1,j,k,1)*dR_l, 
                w(i-1,j,k).amom_phi + vl_R(g,w,i-1,j,k,2)*dR_l, w(i-1,j,k).mom_Z + vl_R(g,w,i-1,j,k,3)*dR_l};

    double w_r[4] = {w(i,j,k).rho + vl_R(g,w,i,j,k,0)*dR_r, w(i,j,k).mom_R + vl_R(g,w,i,j,k,1)*dR_r, 
                w(i,j,k).amom_phi + vl_R(g,w,i,j,k,2)*dR_r, w(i,j,k).mom_Z + vl_R(g,w,i,j,k,3)*dR_r};

    // if (w_l[0] < 1.e-40) { w_l[0] = 1.e-40; }
    // if (w_r[0] < 1.e-40) { w_r[0] = 1.e-40; }

    double u_l[4] = {w_l[0], w_l[0] * w_l[1], w_l[0] * w_l[2] * g.Re(i), w_l[0] * w_l[3]};
    double u_r[4] = {w_r[0], w_r[0] * w_r[1], w_r[0] * w_r[2] * g.Re(i), w_r[0] * w_r[3]};
    
    double v_l, v_r, rhorat, v_av;

    v_l = w_l[1] * normR + w_l[3] * normZ;
    v_r = w_r[1] * normR + w_r[3] * normZ; 

    rhorat = std::pow(u_r[0], 0.5)/std::pow(u_l[0], 0.5);
    v_av = (v_l + rhorat * v_r) / (1. + rhorat);

    // Construct fluxes depending on sign of interface velocities

    construct_fluxes(v_l, v_r, v_av, u_l, u_r, fluxR, i, j, k);
}

__device__
void dust_flux_vlZ(GridRef& g, Field3DConstRef<Quants>& w, int i, int j, int k, Field3DRef<Quants>& fluxZ) {

    double normR = g.face_normal_Z(i,j).R;
    double normZ = g.face_normal_Z(i,j).Z;
    double dZ_l = g.Ze(i,j)-g.Zc(i,j-1);
    double dZ_r = g.Ze(i,j)-g.Zc(i,j);

    double w_l[4] = {w(i,j-1,k).rho + vl_Z(g,w,i,j-1,k,0)*dZ_l, w(i,j-1,k).mom_R + vl_Z(g,w,i,j-1,k,1)*dZ_l, 
                w(i,j-1,k).amom_phi + vl_Z(g,w,i,j-1,k,2)*dZ_l, w(i,j-1,k).mom_Z + vl_Z(g,w,i,j-1,k,3)*dZ_l};

    double w_r[4] = {w(i,j,k).rho + vl_Z(g,w,i,j,k,0)*dZ_r, w(i,j,k).mom_R + vl_Z(g,w,i,j,k,1)*dZ_r, 
                w(i,j,k).amom_phi + vl_Z(g,w,i,j,k,2)*dZ_r, w(i,j,k).mom_Z + vl_Z(g,w,i,j,k,3)*dZ_r};

    // if (w_l[0] < 1.e-40) { w_l[0] = 1.e-40; }
    // if (w_r[0] < 1.e-40) { w_r[0] = 1.e-40; }

    double u_l[4] = {w_l[0], w_l[0] * w_l[1], w_l[0] * w_l[2] * g.Rc(i), w_l[0] * w_l[3]};
    double u_r[4] = {w_r[0], w_r[0] * w_r[1], w_r[0] * w_r[2] * g.Rc(i), w_r[0] * w_r[3]};
 
    double v_l, v_r, rhorat, v_av;

    v_l = w_l[1] * normR + w_l[3] * normZ;
    v_r = w_r[1] * normR + w_r[3] * normZ; 

    rhorat = std::pow(u_r[0], 0.5)/std::pow(u_l[0], 0.5);
    v_av = (v_l + rhorat * v_r) / (1. + rhorat);

    // Construct fluxes depending on sign of interface velocities

    construct_fluxes(v_l, v_r, v_av, u_l, u_r, fluxZ, i, j, k);
}

__global__
void _set_boundariesRP(GridRef g, Field3DRef<Quants> q, int bound, double floor) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.z * blockDim.z ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) {
            for (int k=kidx; k<q.Nd; k+=kstride) {

                if (i < g.Nghost) {
                    if (bound & BoundaryFlags::open_R_inner) {  //outflow
                        if (q(g.Nghost,j,k).mom_R < 0.) {
                            for (int l=0; l<4; l++) {
                                q(i,j,k)[l] = q(g.Nghost,j,k)[l];
                            }
                        }
                        else {
                            q(i,j,k).rho = 1.e-10 * floor;
                            q(i,j,k).mom_R = 0.;
                            q(i,j,k).amom_phi = 0.;
                            q(i,j,k).mom_Z = 0.;
                        }
                    }
                    else {  //reflecting
                        q(i,j,k)[0] = q(2*g.Nghost-1-i,j,k)[0];
                        q(i,j,k)[1] = -q(2*g.Nghost-1-i,j,k)[1];
                        q(i,j,k)[2] = q(2*g.Nghost-1-i,j,k)[2];
                        q(i,j,k)[3] = q(2*g.Nghost-1-i,j,k)[3];
                    }
                }

                if (j>=g.Nphi+g.Nghost) {
                    if (bound & BoundaryFlags::open_Z_outer) {
                        if (q(i,g.Nphi+g.Nghost-1,k).mom_R*g.face_normal_Z(i,g.Nphi+g.Nghost).R + 
                            q(i,g.Nphi+g.Nghost-1,k).mom_Z*g.face_normal_Z(i,g.Nphi+g.Nghost).Z > 0.) {
                            for (int l=0; l<4; l++) {
                                q(i,j,k)[l] = q(i,g.Nphi+g.Nghost-1,k)[l];
                            }
                        }
                        else {
                            q(i,j,k).rho = 1.e-10*floor;
                            q(i,j,k).mom_R = 0.;
                            q(i,j,k).amom_phi = 0.;
                            q(i,j,k).mom_Z = 0.;                            
                        }
                    }
                    else if (bound & BoundaryFlags::zero_Z_outer) {
                        q(i,j,k).rho = 1.e-10*floor;
                        q(i,j,k).mom_R = 0.;
                        q(i,j,k).amom_phi = 0.;
                        q(i,j,k).mom_Z = 0.;
                    }
                    else {
                        q(i,j,k)[0] = q(i,2*(g.Nphi+g.Nghost)-1-j,k)[0];
                        q(i,j,k)[1] = q(i,2*(g.Nphi+g.Nghost)-1-j,k)[1] * (g.cos_th(j)*g.cos_th(j) - g.sin_th(j)*g.sin_th(j)) 
                                        + 2.*q(i,2*(g.Nphi+g.Nghost)-1-j,k)[3]*g.sin_th(j)*g.cos_th(j);
                        q(i,j,k)[2] = q(i,2*(g.Nphi+g.Nghost)-1-j,k)[2];
                        q(i,j,k)[3] = q(i,2*(g.Nphi+g.Nghost)-1-j,k)[3] * (-g.cos_th(j)*g.cos_th(j) + g.sin_th(j)*g.sin_th(j))
                                        + 2.*q(i,2*(g.Nphi+g.Nghost)-1-j,k)[1]*g.sin_th(j)*g.cos_th(j);
                    }
                }        

                if (i>=g.NR+g.Nghost) {
                    if (bound & BoundaryFlags::open_R_outer) {
                        if (q(g.NR+g.Nghost-1,j,k).mom_R > 0.) {
                            for (int l=0; l<4; l++) {
                                q(i,j,k)[l] = q(g.NR+g.Nghost-1,j,k)[l];
                            }
                        }
                        else {
                            // for (int l=0; l<4; l++) {
                            //     q(i,j,k)[l] = q(g.NR+g.Nghost-1,j,k)[l];
                            // }
                            q(i,j,k).rho = 1.e-10*floor;
                            q(i,j,k).mom_R = 0.;
                            q(i,j,k).amom_phi = 0.;
                            q(i,j,k).mom_Z = 0.;
                        }
                    }
                    else if (bound & BoundaryFlags::zero_R_outer) {
                        q(i,j,k).rho = 1.e-10*floor;
                        q(i,j,k).mom_R = 0.;
                        q(i,j,k).amom_phi = 0.;
                        q(i,j,k).mom_Z = 0.;
                    }
                    else {
                        q(i,j,k)[0] = q(2*(g.NR+g.Nghost)-1-i,j,k)[0];
                        q(i,j,k)[1] = -q(2*(g.NR+g.Nghost)-1-i,j,k)[1];
                        q(i,j,k)[2] = q(2*(g.NR+g.Nghost)-1-i,j,k)[2];
                        q(i,j,k)[3] = q(2*(g.NR+g.Nghost)-1-i,j,k)[3];
                    }
                }    
                
                if (j < g.Nghost) {
                    if (bound & BoundaryFlags::open_Z_inner) {  
                        if (q(i,g.Nghost,k).mom_R*g.face_normal_Z(i,g.Nghost).R + 
                            q(i,g.Nghost,k).mom_Z*g.face_normal_Z(i,g.Nghost).Z < 0.) {
                            for (int l=0; l<4; l++) {
                                q(i,j,k)[l] = q(i,g.Nghost,k)[l];
                            }
                        }
                        else {
                            q(i,j,k).rho = 1.e-10*floor;
                            q(i,j,k).mom_R = 0.;
                            q(i,j,k).amom_phi = 0.;
                            q(i,j,k).mom_Z = 0.;                            
                        }
                        
                    }
                    else {  
                        q(i,j,k)[0] = q(i,2*g.Nghost-1-j,k)[0];
                        q(i,j,k)[1] = q(i,2*g.Nghost-1-j,k)[1] * (g.cos_th(j+1)*g.cos_th(j+1) - g.sin_th(j+1)*g.sin_th(j+1)) 
                                        + 2.*q(i,2*g.Nghost-1-j,k)[3]*g.sin_th(j+1)*g.cos_th(j+1);
                        q(i,j,k)[2] = q(i,2*g.Nghost-1-j,k)[2];
                        q(i,j,k)[3] = q(i,2*g.Nghost-1-j,k)[3] * (-g.cos_th(j+1)*g.cos_th(j+1) + g.sin_th(j+1)*g.sin_th(j+1))
                                        + 2.*q(i,2*g.Nghost-1-j,k)[1]*g.sin_th(j+1)*g.cos_th(j+1);
                        // q(i,j,k)[1] = q(i,2*g.Nghost-1-j,k)[1];
                        // q(i,j,k)[2] = q(i,2*g.Nghost-1-j,k)[2];
                        // q(i,j,k)[3] = -q(i,2*g.Nghost-1-j,k)[3];
                    }
                }    
            }
        }
    }
}

__global__ void _calc_flux(GridRef g, Field3DConstRef<Quants> w, Field3DRef<Quants> fluxR, Field3DRef<Quants> fluxZ) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.z * blockDim.z ;

    //if (i >= g.Nghost && i < g.NR+2*g.Nghost-1 && j >= g.Nghost && j < g.Nphi+2*g.Nghost-1 ) {
    for (int i=iidx+2; i<g.NR+2*g.Nghost-1; i+=istride) {
        for (int j=jidx+2; j<g.Nphi+2*g.Nghost-1; j+=jstride) {
            for (int k=kidx; k<w.Nd; k+=kstride) {
                dust_fluxR(g, w, i, j, k, fluxR);
                dust_fluxZ(g, w, i, j, k, fluxZ);
            } 

        }
    }

}

__global__ void _calc_flux_vl(GridRef g, Field3DConstRef<Quants> w, Field3DRef<Quants> fluxR, Field3DRef<Quants> fluxZ) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.z * blockDim.z ;

    //for (int l=0; l<g.NR+2*g.Nghost; l++) {printf("%g\n", vl_R(g,w,l,40,0,0));}

    //if (i >= g.Nghost && i < g.NR+2*g.Nghost-1 && j >= g.Nghost && j < g.Nphi+2*g.Nghost-1 ) {
    for (int i=iidx+2; i<g.NR+2*g.Nghost-1; i+=istride) {
        for (int j=jidx+2; j<g.Nphi+2*g.Nghost-1; j+=jstride) { 
            for (int k=kidx; k<w.Nd; k+=kstride) { 
                dust_flux_vlR(g, w, i, j, k, fluxR);
                dust_flux_vlZ(g, w, i, j, k, fluxZ);
            }
        }
    }
}

__global__ void _update_mid_quants(GridRef g, Field3DRef<Quants> q_mids, Field3DRef<Quants> q, double dt,
                                        Field3DRef<Quants> fluxR, Field3DRef<Quants> fluxZ) {
    
    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.z * blockDim.z ;

    for (int i=iidx+g.Nghost; i<g.NR+2*g.Nghost-2; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+2*g.Nghost-2; j+=jstride) { 
            for (int k=kidx; k<q.Nd; k+=kstride) {
                for (int l=0; l<4; l++) {
                    double df = (fluxR(i,j,k)[l] * g.area_R(i,j) - fluxR(i+1,j,k)[l] * g.area_R(i+1,j)) 
                            + (fluxZ(i,j,k)[l] * g.area_Z(i,j) - fluxZ(i,j+1,k)[l] * g.area_Z(i,j+1));
                    q_mids(i,j,k)[l] = q(i,j,k)[l] + (0.5*dt/g.volume(i,j))*df;
                }
            }
        }
    }
}

__global__ void _update_quants(GridRef g, Field3DRef<Quants> q, double dt,
                            Field3DRef<Quants> fluxR, Field3DRef<Quants> fluxZ, double floor) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.z * blockDim.z ;

    for (int i=iidx+2; i<g.NR+2*g.Nghost-2; i+=istride) {
        for (int j=jidx+2; j<g.Nphi+2*g.Nghost-2; j+=jstride) {    
            for (int k=kidx; k<q.Nd; k+=kstride) {
                for (int l=0; l<4; l++) {
                    double df = (fluxR(i,j,k)[l] * g.area_R(i,j) - fluxR(i+1,j,k)[l] * g.area_R(i+1,j)) 
                            + (fluxZ(i,j,k)[l] * g.area_Z(i,j) - fluxZ(i,j+1,k)[l] * g.area_Z(i,j+1));
                    q(i,j,k)[l] = q(i,j,k)[l] + (dt/g.volume(i,j))*df;                            
                }
                if (q(i,j,k).rho < 10.*floor ) {
                    q(i,j,k).rho = floor;
                    q(i,j,k).mom_R = 0.;
                    q(i,j,k).amom_phi = 0.;
                    q(i,j,k).mom_Z = 0.;
                }
            }
        }
    }
}
__global__ void _update_quants(GridRef g, Field3DRef<Quants> q, double dt,
                            Field3DRef<Quants> fluxR, Field3DRef<Quants> fluxZ, double floor, double* h_w, int* coagbool) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.z * blockDim.z ;

    for (int i=iidx+2; i<g.NR+2*g.Nghost-2; i+=istride) {
        for (int j=jidx+2; j<g.Nphi+2*g.Nghost-2; j+=jstride) {    
            for (int k=kidx; k<q.Nd; k+=kstride) {
                double rhoold = q(i,j,k).rho; 
                for (int l=0; l<4; l++) {
                    double df = (fluxR(i,j,k)[l] * g.area_R(i,j) - fluxR(i+1,j,k)[l] * g.area_R(i+1,j)) 
                            + (fluxZ(i,j,k)[l] * g.area_Z(i,j) - fluxZ(i,j+1,k)[l] * g.area_Z(i,j+1));
                    q(i,j,k)[l] = q(i,j,k)[l] + (dt/g.volume(i,j))*df;                            
                }
                if (q(i,j,k).rho < 10.*floor || g.Zc(i,j) > h_w[i] || h_w[i] == g.Zc(i,g.Nghost)) {
                    q(i,j,k).rho = floor;
                    q(i,j,k).mom_R = 0.;
                    q(i,j,k).amom_phi = 0.;
                    q(i,j,k).mom_Z = 0.;
                }
                if (i < g.NR+g.Nghost-10 && q(i,j,k).rho > floor && (rhoold/q(i,j,k).rho < 1e-4 || rhoold/q(i,j,k).rho > 1e4)){
                    coagbool[0]=1;
                    // printf("%d, %d, %d\n",i,j,k);
                }
            }
        }
    }
}

__global__
void _calc_primRP(GridRef g, Field3DRef<Quants> q, Field3DRef<Quants> w) {

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
                w(i,j,k).mom_R = q(i,j,k).mom_R/q(i,j,k).rho;
                w(i,j,k).amom_phi = q(i,j,k).amom_phi/(q(i,j,k).rho * g.Rc(i));
                w(i,j,k).mom_Z = q(i,j,k).mom_Z/q(i,j,k).rho;
            } 
        }
    }
}


void vl_advect(Grid& g, Field3D<Quants>& q, double dt, int bound, double floor) {

    Field3D<Quants> q_mids = Field3D<Quants>(g.NR+2*g.Nghost, g.Nphi+2*g.Nghost, q.Nd);
    Field3D<Quants> w = Field3D<Quants>(g.NR+2*g.Nghost, g.Nphi+2*g.Nghost, q.Nd);

    Field3D<Quants> fluxR = Field3D<Quants>(g.NR+2*g.Nghost, g.Nphi+2*g.Nghost, q.Nd);
    Field3D<Quants> fluxZ = Field3D<Quants>(g.NR+2*g.Nghost, g.Nphi+2*g.Nghost, q.Nd);

    dim3 threads(16,16,1) ;
    //dim3 blocks((g.NR + 2*g.Nghost+15)/16,(g.Nphi + 2*g.Nghost+15)/16) ;
    dim3 blocks(48,48,48) ;

    _set_boundariesRP<<<blocks,threads>>>(g, q, bound, floor);
    _calc_primRP<<<blocks,threads>>>(g, q, w);

    // Calc donor cell flux

    _calc_flux<<<blocks,threads>>>(g, w, fluxR, fluxZ);

    // Update quantities a half time step

    _update_mid_quants<<<blocks,threads>>>(g, q_mids, q, dt, fluxR, fluxZ);
    _set_boundariesRP<<<blocks,threads>>>(g, q_mids, bound, floor);
    _calc_primRP<<<blocks,threads>>>(g, q_mids, w);

    // Compute fluxes with Van Leer

    _calc_flux_vl<<<blocks,threads>>>(g, w, fluxR, fluxZ);

    _update_quants<<<blocks,threads>>>(g, q, dt, fluxR, fluxZ, floor);

}

/// Class functions

void VL_Advect::operator() (Grid& g, Field3D<Quants>& q, double dt) {

    Field3D<Quants> q_mids = Field3D<Quants>(g.NR+2*g.Nghost, g.Nphi+2*g.Nghost, q.Nd);
    Field3D<Quants> w = Field3D<Quants>(g.NR+2*g.Nghost, g.Nphi+2*g.Nghost, q.Nd);

    Field3D<Quants> fluxR = Field3D<Quants>(g.NR+2*g.Nghost, g.Nphi+2*g.Nghost, q.Nd);
    Field3D<Quants> fluxZ = Field3D<Quants>(g.NR+2*g.Nghost, g.Nphi+2*g.Nghost, q.Nd);

    dim3 threads(16,16,1) ;
    //dim3 blocks((g.NR + 2*g.Nghost+15)/16,(g.Nphi + 2*g.Nghost+15)/16) ;
    dim3 blocks(48,48,48) ;

    _set_boundariesRP<<<blocks,threads>>>(g, q, _boundary, _floor);
    _calc_primRP<<<blocks,threads>>>(g, q, w);

    // Calc donor cell flux

    _calc_flux<<<blocks,threads>>>(g, w, fluxR, fluxZ);

    // Update quantities a half time step

    _update_mid_quants<<<blocks,threads>>>(g, q_mids, q, dt, fluxR, fluxZ);
    _set_boundariesRP<<<blocks,threads>>>(g, q_mids, _boundary, _floor);
    _calc_primRP<<<blocks,threads>>>(g, q_mids, w);

    // Compute fluxes with Van Leer and update quants to full timestep

    _calc_flux_vl<<<blocks,threads>>>(g, w, fluxR, fluxZ);
    _update_quants<<<blocks,threads>>>(g, q, dt, fluxR, fluxZ, _floor);

}

__global__
void _compute_CFL(GridRef g, Field3DConstRef<Quants> q, FieldRef<double> CFL_grid) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) {
            double CFL_k = 1e308;
            for (int k=0; k<q.Nd; k++) {

                double dtR = abs(g.dRe(i) *  q(i,j,k).rho/q(i,j,k).mom_R);
                double dtZ = abs(g.dZe(i,j) *  q(i,j,k).rho/q(i,j,k).mom_Z);

                double CFL_RZmin = min(dtR, dtZ);

                CFL_k = min(CFL_k, CFL_RZmin);
            }
            CFL_grid(i,j) = CFL_k;
        }
    } 
}

double VL_Advect::get_CFL_limit(const Grid& g, const Field3D<Quants>& q) {

    dim3 threads(16,16) ;
    //dim3 blocks((g.NR + 2*g.Nghost+15)/16,(g.Nphi + 2*g.Nghost+15)/16) ;
    dim3 blocks(48,48) ;

    Field<double> CFL_grid = create_field<double>(g);

    _compute_CFL<<<blocks,threads>>>(g, q, CFL_grid);
    Reduction::scan_R_min(g, CFL_grid);

    double dt = CFL_grid(g.NR+2*g.Nghost-1,0) ;
    for (int j=0; j < g.Nphi+2*g.Nghost; j++) {
        dt = std::min(dt, CFL_grid(g.NR+2*g.Nghost-1, j)) ;
    }

    return _CFL * dt;
}


// Flux-coefficients for the left and upper faces.
__device__ 
fluxes left_side_flux_no_area(GridRef& g, Field3DConstRef<double>& D, int i, int j, int k) {
    double w_I = weights_R(g, D, i, j, k) ;
    double w_J = weights_Z(g, D, i, j+1, k) ;
    // Do the (j,j+1) edge first, then (i,i-1)
    double d1, d2 = 0;
    d1 = (g.Ze(i,j+1) - g.Zc(i,j)) * g.cos_th(j+1) ;
    if (j+1 < g.Nphi + 2*g.Nghost)
        d2 = (g.Zc(i,j+1) - g.Ze(i,j+1)) * g.cos_th(j+1) ;
    vec dy1, dy2 = {0, 0} ;
    dy1 = vec{- d1 * g.sin_th(j+1), + d1 * g.cos_th(j+1)};
    if (j+1 < g.Nphi + 2*g.Nghost)
        dy2 = vec{+ d2 * g.sin_th(j+1), g.dZc(i,j) - d2 * g.cos_th(j+1)};
    dy2 = (dy1*w_J + dy2*(1-w_J)) ;
    dy1.R = g.Re(i) - g.Rc(i) ;
    if (i > 0)
        dy1.Z = (g.Zc(i-1,j) - g.Zc(i,j)) * w_I ; // (1 - w_I) ;
    else
        dy1.Z = 0 ;
    fluxes flux ;
    flux.w.R = w_I ;
    flux.w.Z = w_J ;
    if (i == 0) w_I = 1 ;
    if (j+1 == g.Nphi + 2*g.Nghost) w_J = 0 ;
    vec w_12 = decompose_vector(g.face_normal_R(i,j)*-1, dy1, dy2) ;
    // if (i == 60 && j == 2 && k == 10) { printf("%g", w_12.R);}
    flux.R.R =    w_I *D(i,j,k) * w_12.R ;
    flux.R.Z = (1-w_J)*D(i,j,k) * w_12.Z ;
    
    w_12 = decompose_vector(g.face_normal_Z(i,j+1), dy1, dy2) ;
    flux.Z.R =    w_I *D(i,j,k) * w_12.R ;
    flux.Z.Z = (1-w_J)*D(i,j,k) * w_12.Z ;
    return flux ;
}
// Flux-coefficients for the right and lower faces.
__device__ 
fluxes right_side_flux_no_area(GridRef& g, Field3DConstRef<double>& D, int i, int j, int k) {
    double w_I = weights_R(g, D, i+1, j, k) ;
    double w_J = weights_Z(g, D, i, j, k) ;
    // Do the (j,j-1) edge first, then (i,i+1)
    double d1=0, d2 ;
    if (j > g.Nghost)
        d1 = (g.Ze(i,j) - g.Zc(i,j-1)) * g.cos_th(j) ;
    d2 = (g.Zc(i,j) - g.Ze(i,j)) * g.cos_th(j) ;
    vec dy1 = {0, 0}, dy2 ;
    if (j > g.Nghost)
        dy1 = vec{- d1 * g.sin_th(j), -g.dZc(i,j-1) + d1 * g.cos_th(j)};
    dy2 = vec{+ d2 * g.sin_th(j),  - d2 * g.cos_th(j)};
    dy2 = (dy1*w_J + dy2*(1-w_J)) ;
    dy1.R = g.Re(i+1) - g.Rc(i) ;
    if (i+1 < g.NR + 2*g.Nghost)
        dy1.Z = (g.Zc(i+1,j) - g.Zc(i,j)) * (1-w_I) ;
    else
        dy1.Z = 0 ;
    fluxes flux ;
    flux.w.R = w_I ;
    flux.w.Z = w_J ;
    if (i+1 == g.NR + 2*g.Nghost) w_I = 0 ;
    if (j == g.Nghost) w_J = 1 ;
    vec w_12 = decompose_vector(g.face_normal_R(i+1,j), dy1, dy2) ;
    flux.R.R = (1-w_I)*D(i,j,k) * w_12.R ;
    flux.R.Z =    w_J *D(i,j,k) * w_12.Z ;
    w_12 = decompose_vector(g.face_normal_Z(i,j)*-1, dy1, dy2) ;
    flux.Z.R = (1-w_I)*D(i,j,k) * w_12.R ;
    flux.Z.Z =    w_J *D(i,j,k) * w_12.Z ;
    return flux ;
}


__device__
double compute_diff_fluxR(GridRef& g, Field3DConstRef<double>& D, Field3DConstRef<Quants>& w, FieldConstRef<Quants>& w_gas, int i, int j, int k) {

    if (i < g.Nghost+1 || i > g.NR+g.Nghost-1) { return 0.; }
    else if ((w(i-1,j,k).rho/w_gas(i-1,j).rho)/(w(i,j,k).rho/w_gas(i,j).rho) > 1e5 || (w(i-1,j,k).rho/w_gas(i-1,j).rho)/(w(i,j,k).rho/w_gas(i,j).rho) < 1e-5) {return 0.;}
    // else if ((w(i-1,j,k).rho)/(w(i,j,k).rho) > 1e3 || (w(i-1,j,k).rho)/(w(i,j,k).rho) < 1e-5) {return 0.;}

    fluxes fi, fj; 
    
    fi = left_side_flux_no_area(g, D, i, j, k) ;
    double left_flux = 0.;

    // left interface flux

    left_flux -= - (1 - fi.w.R) * fi.R.R * (w(i-1,j,k).rho/w_gas(i-1,j).rho);
    left_flux += - (1 - fi.w.R) * fi.R.R * (w(i,j,k).rho/w_gas(i,j).rho); 
    
    left_flux -= - (1 - fi.w.R) * fi.R.Z * (w(i,j+1,k).rho/w_gas(i,j+1).rho);
    left_flux += - (1 - fi.w.R) * fi.R.Z * (w(i,j,k).rho/w_gas(i,j).rho);

    fj = right_side_flux_no_area(g, D, i-1, j, k) ;


    left_flux -= - fj.w.R * fj.R.R * (w(i-1,j,k).rho/w_gas(i-1,j).rho);
    left_flux += - fj.w.R * fj.R.R * (w(i,j,k).rho/w_gas(i,j).rho); 

    left_flux -= - fj.w.R * fj.R.Z * (w(i-1,j,k).rho/w_gas(i-1,j).rho);
    left_flux += - fj.w.R * fj.R.Z * (w(i-1,j-1,k).rho/w_gas(i-1,j-1).rho);

    // if (w_gas(i-1,j).rho/w_gas(i,j).rho > 1.e2 || w_gas(i-1,j).rho/w_gas(i,j).rho < 1.e-2) {left_flux = 0;}

    return left_flux;
}

__device__
double compute_diff_fluxZ(GridRef& g, Field3DConstRef<double>& D, Field3DConstRef<Quants>& w, FieldConstRef<Quants>& w_gas, int i, int j, int k) {

    if (j < g.Nghost+1 || j > g.Nphi+g.Nghost-1) { return 0.; }
    else if ((w(i,j-1,k).rho/w_gas(i,j-1).rho)/(w(i,j,k).rho/w_gas(i,j).rho) > 1e5 || (w(i,j-1,k).rho/w_gas(i,j-1).rho)/(w(i,j,k).rho/w_gas(i,j).rho) < 1e-5) {return 0.;}
    else if ((w(i-1,j,k).rho/w_gas(i-1,j).rho)/(w(i,j,k).rho/w_gas(i,j).rho) > 1e5 || (w(i-1,j,k).rho/w_gas(i-1,j).rho)/(w(i,j,k).rho/w_gas(i,j).rho) < 1e-5) {return 0.;}
    // else if ((w(i,j-1,k).rho)/(w(i,j,k).rho) > 1e3 || (w(i,j-1,k).rho)/(w(i,j,k).rho) < 1e-3) {return 0.;}

    fluxes fi, fj; 
    
    fi = right_side_flux_no_area(g, D, i, j, k) ;
    double lower_flux = 0.;

    // lower interface flux

    lower_flux -= - (1-fi.w.Z) * fi.Z.R * (w(i+1,j,k).rho/w_gas(i+1,j).rho);
    lower_flux += - (1-fi.w.Z) * fi.Z.R * (w(i,j,k).rho/w_gas(i,j).rho);

    lower_flux -= - (1-fi.w.Z) * fi.Z.Z * (w(i,j-1,k).rho/w_gas(i,j-1).rho);
    lower_flux += - (1-fi.w.Z) * fi.Z.Z * (w(i,j,k).rho/w_gas(i,j).rho);

    fj = left_side_flux_no_area(g, D, i, j-1, k) ;

    lower_flux -= - fj.w.Z * fj.Z.Z * (w(i,j-1,k).rho/w_gas(i,j-1).rho);
    lower_flux += - fj.w.Z * fj.Z.Z * (w(i,j,k).rho/w_gas(i,j).rho);

    lower_flux -= - fj.w.Z * fj.Z.R * (w(i,j-1,k).rho/w_gas(i,j-1).rho);
    lower_flux += - fj.w.Z * fj.Z.R * (w(i-1,j-1,k).rho/w_gas(i-1,j-1).rho);

    return lower_flux;
}

__device__
void construct_diff_fluxes(double v_l, double v_r, double v_av, 
                  double u_l[4], double u_r[4], Field3DRef<Quants>& flux, double diff_flux, int i, int j, int k) {

    if (diff_flux > 0) {
        flux(i,j,k).rho += diff_flux;
        flux(i,j,k).mom_R += (u_l[1]/u_l[0])*diff_flux; 
        flux(i,j,k).amom_phi += (u_l[2]/u_l[0])*diff_flux;
        flux(i,j,k).mom_Z += (u_l[3]/u_l[0])*diff_flux;
    }
    else {
        flux(i,j,k).rho += diff_flux;
        flux(i,j,k).mom_R += (u_r[1]/u_r[0])*diff_flux; 
        flux(i,j,k).amom_phi += (u_r[2]/u_r[0])*diff_flux;
        flux(i,j,k).mom_Z += (u_r[3]/u_r[0])*diff_flux;
    }
}



__device__
void construct_diff_fluxes_novdiff(GridRef& g, double v_l, double v_r, double v_av, 
                  double u_l[4], double u_r[4], Field3DRef<Quants>& flux, double diff_flux, int i, int j, int k) {

    if (v_l < 0 && v_r > 0) {
        flux(i,j,k) = {diff_flux, 0., 0., 0.};
        return;
    }
    
    if (v_av > 0.) {  
        if (diff_flux > 0) {
            flux(i,j,k) = {u_l[0]*v_l + diff_flux, u_l[1]*v_l, 
                    u_l[2]*v_l, u_l[3]*v_l};
        }  
        else {
            flux(i,j,k) = {u_l[0]*v_l + diff_flux, u_l[1]*v_l, 
                    u_l[2]*v_l, u_l[3]*v_l};
        }
        return;
    }

    if (v_av < 0.) {     
        if (diff_flux > 0) {
            flux(i,j,k) = {u_r[0]*v_r + diff_flux, u_r[1]*v_r, 
                    u_r[2]*v_r, u_r[3]*v_r};
        }  
        else {
            flux(i,j,k) = {u_r[0]*v_r + diff_flux, u_r[1]*v_r, 
                    u_r[2]*v_r, u_r[3]*v_r};
        }
        return;
    }

    if (v_av == 0.) {
        if (diff_flux > 0) {
            flux(i,j,k) = {0.5*(u_l[0]*v_l + u_r[0]*v_r + 2*diff_flux), 0.5*(u_l[1]*v_l + u_r[1]*v_r), 
                        0.5*(u_l[2]*v_l + u_r[2]*v_r), 0.5*(u_l[3]*v_l + u_r[3]*v_r)};
        }
        else {
            flux(i,j,k) = {0.5*(u_l[0]*v_l + u_r[0]*v_r + 2*diff_flux), 0.5*(u_l[1]*v_l + u_r[1]*v_r), 
                        0.5*(u_l[2]*v_l + u_r[2]*v_r), 0.5*(u_l[3]*v_l + u_r[3]*v_r)};
        }
        return;
    }
}

__device__
void dust_diff_fluxR(GridRef& g, Field3DConstRef<Quants>& w, int i, int j, int k, Field3DRef<Quants>& fluxR, double diff_fluxR) {

    double normR = g.face_normal_R(i,j).R;
    double normZ = g.face_normal_R(i,j).Z;

    double w_l[4] = {w(i-1,j,k).rho, w(i-1,j,k).mom_R, w(i-1,j,k).amom_phi, w(i-1,j,k).mom_Z};
    double w_r[4] = {w(i,j,k).rho, w(i,j,k).mom_R, w(i,j,k).amom_phi, w(i,j,k).mom_Z};     

    double u_l[4] = {w_l[0], w_l[0] * w_l[1], w_l[0] * w_l[2] * g.Re(i), w_l[0] * w_l[3]};
    double u_r[4] = {w_r[0], w_r[0] * w_r[1], w_r[0] * w_r[2] * g.Re(i), w_r[0] * w_r[3]};

    double v_l, v_r, rhorat, v_av;

    v_l = w_l[1] * normR + w_l[3] * normZ;
    v_r = w_r[1] * normR + w_r[3] * normZ; 

    rhorat = std::sqrt(w_r[0]/w_l[0]);
    v_av = (v_l + rhorat * v_r) / (1. + rhorat);

    // Construct fluxes depending on sign of interface velocities

    construct_fluxes(v_l, v_r, v_av, u_l, u_r, fluxR, i,j,k);
    construct_diff_fluxes(v_l, v_r, v_av, u_l, u_r, fluxR, diff_fluxR, i,j,k);
}

__device__
void dust_diff_fluxZ(GridRef& g, Field3DConstRef<Quants>& w, int i, int j, int k, Field3DRef<Quants>& fluxZ, double diff_fluxZ) {

    double normR = g.face_normal_Z(i,j).R;
    double normZ = g.face_normal_Z(i,j).Z;

    double w_l[4] = {w(i,j-1,k).rho, w(i,j-1,k).mom_R, w(i,j-1,k).amom_phi, w(i,j-1,k).mom_Z};
    double w_r[4] = {w(i,j,k).rho, w(i,j,k).mom_R, w(i,j,k).amom_phi, w(i,j,k).mom_Z};  

    double u_l[4] = {w_l[0], w_l[0] * w_l[1], w_l[0] * w_l[2] * g.Rc(i), w_l[0] * w_l[3]};
    double u_r[4] = {w_r[0], w_r[0] * w_r[1], w_r[0] * w_r[2] * g.Rc(i), w_r[0] * w_r[3]};   

    double v_l, v_r, rhorat, v_av;

    v_l = w_l[1] * normR + w_l[3] * normZ;
    v_r = w_r[1] * normR + w_r[3] * normZ; 

    rhorat = std::sqrt(w_r[0]/w_l[0]);
    v_av = (v_l + rhorat * v_r) / (1. + rhorat);

    // Construct fluxes depending on sign of interface velocities

    construct_fluxes(v_l, v_r, v_av, u_l, u_r, fluxZ, i,j,k);
    construct_diff_fluxes(v_l, v_r, v_av, u_l, u_r, fluxZ, diff_fluxZ, i,j,k);
}

__device__
void dust_diff_flux_vlR(GridRef& g, Field3DConstRef<Quants>& w, int i, int j, int k, Field3DRef<Quants>& fluxR, double diff_fluxR) {

    double normR = g.face_normal_R(i,j).R;
    double normZ = g.face_normal_R(i,j).Z;
    double dR_l = g.Re(i)-g.Rc(i-1);
    double dR_r = g.Re(i)-g.Rc(i);

    double w_l[4] = {w(i-1,j,k).rho + vl_R(g,w,i-1,j,k,0)*dR_l, w(i-1,j,k).mom_R + vl_R(g,w,i-1,j,k,1)*dR_l, 
                w(i-1,j,k).amom_phi + vl_R(g,w,i-1,j,k,2)*dR_l, w(i-1,j,k).mom_Z + vl_R(g,w,i-1,j,k,3)*dR_l};

    double w_r[4] = {w(i,j,k).rho + vl_R(g,w,i,j,k,0)*dR_r, w(i,j,k).mom_R + vl_R(g,w,i,j,k,1)*dR_r, 
                w(i,j,k).amom_phi + vl_R(g,w,i,j,k,2)*dR_r, w(i,j,k).mom_Z + vl_R(g,w,i,j,k,3)*dR_r};

    // if (w_l[0] < 1.e-40) { w_l[0] = 1.e-40; }
    // if (w_r[0] < 1.e-40) { w_r[0] = 1.e-40; }

    double u_l[4] = {w_l[0], w_l[0] * w_l[1], w_l[0] * w_l[2] * g.Re(i), w_l[0] * w_l[3]};
    double u_r[4] = {w_r[0], w_r[0] * w_r[1], w_r[0] * w_r[2] * g.Re(i), w_r[0] * w_r[3]};
    
    double v_l, v_r, rhorat, v_av;

    v_l = w_l[1] * normR + w_l[3] * normZ;
    v_r = w_r[1] * normR + w_r[3] * normZ; 

    rhorat = std::sqrt(u_r[0]/u_l[0]);
    v_av = (v_l + rhorat * v_r) / (1. + rhorat);

    // Construct fluxes depending on sign of interface velocities

    construct_fluxes(v_l, v_r, v_av, u_l, u_r, fluxR, i,j,k);
    construct_diff_fluxes(v_l, v_r, v_av, u_l, u_r, fluxR, diff_fluxR, i,j,k);
    

}

__device__
void dust_diff_flux_vlZ(GridRef& g, Field3DConstRef<Quants>& w, int i, int j, int k, Field3DRef<Quants>& fluxZ, double diff_fluxZ) {

    double normR = g.face_normal_Z(i,j).R;
    double normZ = g.face_normal_Z(i,j).Z;
    double dZ_l = g.Ze(i,j)-g.Zc(i,j-1);
    double dZ_r = g.Ze(i,j)-g.Zc(i,j);

    double w_l[4] = {w(i,j-1,k).rho + vl_Z(g,w,i,j-1,k,0)*dZ_l, w(i,j-1,k).mom_R + vl_Z(g,w,i,j-1,k,1)*dZ_l, 
                w(i,j-1,k).amom_phi + vl_Z(g,w,i,j-1,k,2)*dZ_l, w(i,j-1,k).mom_Z + vl_Z(g,w,i,j-1,k,3)*dZ_l};

    double w_r[4] = {w(i,j,k).rho + vl_Z(g,w,i,j,k,0)*dZ_r, w(i,j,k).mom_R + vl_Z(g,w,i,j,k,1)*dZ_r, 
                w(i,j,k).amom_phi + vl_Z(g,w,i,j,k,2)*dZ_r, w(i,j,k).mom_Z + vl_Z(g,w,i,j,k,3)*dZ_r};

    // if (w_l[0] < 1.e-40) { w_l[0] = 1.e-40; }
    // if (w_r[0] < 1.e-40) { w_r[0] = 1.e-40; }

    double u_l[4] = {w_l[0], w_l[0] * w_l[1], w_l[0] * w_l[2] * g.Rc(i), w_l[0] * w_l[3]};
    double u_r[4] = {w_r[0], w_r[0] * w_r[1], w_r[0] * w_r[2] * g.Rc(i), w_r[0] * w_r[3]};
 
    double v_l, v_r, rhorat, v_av;

    v_l = w_l[1] * normR + w_l[3] * normZ;
    v_r = w_r[1] * normR + w_r[3] * normZ; 

    rhorat = std::sqrt(u_r[0]/u_l[0]);
    v_av = (v_l + rhorat * v_r) / (1. + rhorat);

    // Construct fluxes depending on sign of interface velocities

    construct_fluxes(v_l, v_r, v_av, u_l, u_r, fluxZ, i,j,k);
    construct_diff_fluxes(v_l, v_r, v_av, u_l, u_r, fluxZ, diff_fluxZ, i,j,k);
}


__global__ void _calc_diff_flux(GridRef g, Field3DConstRef<Quants> w, FieldConstRef<Quants> w_gas, 
                                Field3DRef<Quants> fluxR, Field3DRef<Quants> fluxZ, Field3DConstRef<double> D) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.z * blockDim.z ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost+1; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost+1; j+=jstride) {
            for (int k=kidx; k<w.Nd; k+=kstride) {    
                            
                double diff_fluxR = compute_diff_fluxR(g, D, w, w_gas, i, j, k);
                double diff_fluxZ = compute_diff_fluxZ(g, D, w, w_gas, i, j, k);
                
                dust_diff_fluxR(g, w, i, j, k, fluxR, diff_fluxR);
                dust_diff_fluxZ(g, w, i, j, k, fluxZ, diff_fluxZ); 
            } 
        }
    }

}


__global__ void _calc_diff_flux_vl(GridRef g, Field3DConstRef<Quants> w, FieldConstRef<Quants> w_gas,
                                    Field3DRef<Quants> fluxR, Field3DRef<Quants> fluxZ, Field3DConstRef<double> D) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.z * blockDim.z ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost+1; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost+1; j+=jstride) {
            for (int k=kidx; k<w.Nd; k+=kstride) { 

                double diff_fluxR = compute_diff_fluxR(g, D, w, w_gas, i, j, k);
                double diff_fluxZ = compute_diff_fluxZ(g, D, w, w_gas, i, j, k);   

                dust_diff_flux_vlR(g, w, i, j, k, fluxR, diff_fluxR);
                dust_diff_flux_vlZ(g, w, i, j, k, fluxZ, diff_fluxZ);
            }
        }
    }

}

void VL_Diff_Advect::operator() (Grid& g, Field3D<Quants>& q, const Field<Quants>& w_gas, const Field3D<double>& D, double dt) {

    Field3D<Quants> q_mids = Field3D<Quants>(g.NR+2*g.Nghost, g.Nphi+2*g.Nghost, q.Nd);
    Field3D<Quants> w = Field3D<Quants>(g.NR+2*g.Nghost, g.Nphi+2*g.Nghost, q.Nd);

    Field3D<Quants> fluxR = Field3D<Quants>(g.NR+2*g.Nghost, g.Nphi+2*g.Nghost, q.Nd);
    Field3D<Quants> fluxZ = Field3D<Quants>(g.NR+2*g.Nghost, g.Nphi+2*g.Nghost, q.Nd);

    dim3 threads(16,8,4) ;
    dim3 blocks((g.NR + 2*g.Nghost+15)/16,(g.Nphi + 2*g.Nghost+7)/8, (q.Nd+3)/4) ;
    //dim3 blocks(4,4,4) ;

    _set_boundariesRP<<<blocks,threads>>>(g, q, _boundary, _floor);
    _calc_primRP<<<blocks,threads>>>(g, q, w);

    // Calc donor cell flux

    _calc_diff_flux<<<blocks,threads>>>(g, w, w_gas, fluxR, fluxZ, D);

    // Update quantities a half time step

    _update_mid_quants<<<blocks,threads>>>(g, q_mids, q, dt, fluxR, fluxZ);
    _set_boundariesRP<<<blocks,threads>>>(g, q_mids, _boundary, _floor);
    _calc_primRP<<<blocks,threads>>>(g, q_mids, w);

    // Compute fluxes with Van Leer

    _calc_diff_flux_vl<<<blocks,threads>>>(g, w, w_gas, fluxR, fluxZ, D);

    _update_quants<<<blocks,threads>>>(g, q, dt, fluxR, fluxZ, _floor);

}
void VL_Diff_Advect::operator() (Grid& g, Field3D<Quants>& q, const Field<Quants>& w_gas, const Field3D<double>& D, double dt, CudaArray<double>& h_w, double R_cav, CudaArray<int>& coagbool) {

    Field3D<Quants> q_mids = Field3D<Quants>(g.NR+2*g.Nghost, g.Nphi+2*g.Nghost, q.Nd);
    Field3D<Quants> w = Field3D<Quants>(g.NR+2*g.Nghost, g.Nphi+2*g.Nghost, q.Nd);

    Field3D<Quants> fluxR = Field3D<Quants>(g.NR+2*g.Nghost, g.Nphi+2*g.Nghost, q.Nd);
    Field3D<Quants> fluxZ = Field3D<Quants>(g.NR+2*g.Nghost, g.Nphi+2*g.Nghost, q.Nd);

    dim3 threads(16,8,4) ;
    dim3 blocks((g.NR + 2*g.Nghost+15)/16,(g.Nphi + 2*g.Nghost+7)/8, (q.Nd+3)/4) ;
    //dim3 blocks(4,4,4) ;

    _set_boundariesRP<<<blocks,threads>>>(g, q, _boundary, _floor);
    _calc_primRP<<<blocks,threads>>>(g, q, w);

    // Calc donor cell flux

    _calc_diff_flux<<<blocks,threads>>>(g, w, w_gas, fluxR, fluxZ, D);

    // Update quantities a half time step

    _update_mid_quants<<<blocks,threads>>>(g, q_mids, q, dt, fluxR, fluxZ);
    _set_boundariesRP<<<blocks,threads>>>(g, q_mids, _boundary, _floor);
    _calc_primRP<<<blocks,threads>>>(g, q_mids, w);

    // Compute fluxes with Van Leer

    _calc_diff_flux_vl<<<blocks,threads>>>(g, w, w_gas, fluxR, fluxZ, D);

    _update_quants<<<blocks,threads>>>(g, q, dt, fluxR, fluxZ, _floor, h_w.get(), coagbool.get());

}

__global__
void _compute_CFL_diff(GridRef g, Field3DConstRef<Quants> q, FieldRef<double> CFL_grid, Field3DConstRef<double> D,
                        double CFL_adv, double CFL_diff, double floor) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost; j+=jstride) {
            double CFL_k = 1e308;
            for (int k=0; k<q.Nd; k++) {

                // if (q(i,j,k).rho <= 1.e-40) { continue; }

                double dtR = abs(g.dRe(i) *  q(i,j,k).rho/q(i,j,k).mom_R);
                double dtZ = abs(g.dZe(i,j) *  q(i,j,k).rho/q(i,j,k).mom_Z);

                double CFL_RZmin = min(dtR, dtZ);

                double dtR_diff = abs(g.dRe(i)*g.dRe(i) / D(i,j,k));
                double dtZ_diff = abs(g.dZe(i,j)*g.dZe(i,j) / D(i,j,k));

                double CFL_RZmin_diff = min(dtR_diff, dtZ_diff);

                double CFL_advdiffmin; 
                
                if (CFL_RZmin < CFL_RZmin_diff) {
                    CFL_advdiffmin = CFL_adv * CFL_RZmin;
                }
                else {
                    CFL_advdiffmin = CFL_diff * CFL_RZmin_diff;
                }

                CFL_k = min(CFL_k, CFL_advdiffmin);
                
            }
            CFL_grid(i,j) = CFL_k;
        }
    } 
}

double VL_Diff_Advect::get_CFL_limit(const Grid& g, const Field3D<Quants>& q, const Field3D<double>& D) {

    dim3 threads(32,32) ;
    dim3 blocks((g.NR + 2*g.Nghost+31)/32,(g.Nphi + 2*g.Nghost+31)/32) ;
    //dim3 blocks(48,48) ;

    Field<double> CFL_grid = create_field<double>(g);
    set_all(g, CFL_grid, 1e300);

    _compute_CFL_diff<<<blocks,threads>>>(g, q, CFL_grid, D, _CFL_adv, _CFL_diff, _floor);
    check_CUDA_errors("_compute_CFL_diff") ;
    Reduction::scan_R_min(g, CFL_grid);

    double dt = CFL_grid(g.NR+g.Nghost-1,g.Nghost) ;
    for (int j=g.Nghost; j < g.Nphi+g.Nghost; j++) {
        dt = std::min(dt, CFL_grid(g.NR+g.Nghost-1, j)) ;
    }

    return dt;
}
double VL_Diff_Advect::get_CFL_limit_debug(const Grid& g, const Field3D<Quants>& q, const Field3D<double>& D) {

    dim3 threads(32,32) ;
    dim3 blocks((g.NR + 2*g.Nghost+31)/32,(g.Nphi + 2*g.Nghost+31)/32) ;
    //dim3 blocks(48,48) ;

    Field<double> CFL_grid = create_field<double>(g);
    set_all(g, CFL_grid, 1e300);

    _compute_CFL_diff<<<blocks,threads>>>(g, q, CFL_grid, D, _CFL_adv, _CFL_diff, _floor);
    check_CUDA_errors("_compute_CFL_diff") ;

    double dt = 1e300 ;
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