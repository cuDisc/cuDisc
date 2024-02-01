
#ifndef _CUDISC_HEADERS_FLD_DEVICE_H_
#define _CUDISC_HEADERS_FLD_DEVICE_H_

/* FLD_device.h
 *
 * Contains re-usable but internal routines for the FLD modules. 
*/

#include "field.h"
#include "grid.h"
#include "planck.h"


/* gradJ
 *
 * Computes |\grad J| / J for the flux limiter
*/
template<typename FieldType, typename... Indices>
 __device__
double gradJ(GridRef& g, FieldType& J, int i, int j, Indices... k) {
    
    double J0 = J(i,j,k...) ;
    double Jtot = J0*J0 ; int n = 1;
    double grad_J[2] = {0, 0} ;

    // Clip i/j to make sure we get a sensible flux-limiter at the boundaries.
    i = min(max(i, g.Nghost), g.NR   + g.Nghost-1) ;
    j = min(max(j, g.Nghost), g.Nphi + g.Nghost-1) ;

    if (i > g.Nghost) {
        grad_J[0] += 0.5 * (J(i-1,j,k...) - J0) / (g.Rc(i-1) - g.Rc(i)) ;
        Jtot += J(i-1,j,k...)*J(i-1,j,k...) ;
        n++ ;
    }
    if (i < g.NR + g.Nghost-1) {
        grad_J[0] += 0.5 * (J(i+1,j,k...) - J0) / (g.Rc(i+1) - g.Rc(i)) ;
        Jtot += J(i+1,j,k...)*J(i+1,j,k...) ;
        n++ ;
    }
    if (i == g.Nghost || i ==  g.NR + g.Nghost-1)
        grad_J[0] *= 2 ;
    
    if (j > g.Nghost) {
        grad_J[1] += 
            0.5 * (J(i,j-1,k...) - J0) / ((g.Zc(i,j-1) - g.Zc(i,j)) * g.face_normal_Z(i,j).Z) ;
        Jtot += J(i,j-1,k...)*J(i,j-1,k...) ;
        n++ ;
    }
    if (j < g.Nphi + g.Nghost-1) {
        grad_J[1] += 
            0.5 * (J(i,j+1,k...) - J0) / ((g.Zc(i,j+1) - g.Zc(i,j)) * g.face_normal_Z(i,j+1).Z) ;
        Jtot += J(i,j+1,k...)*J(i,j+1,k...);
        n++ ;
    }
    if (j == g.Nghost || j ==  g.Nphi + g.Nghost-1)
        grad_J[1] *= 2 ;

    Jtot /= n ;
    
    return sqrt((grad_J[0]*grad_J[0] + grad_J[1]*grad_J[1]) / (Jtot+1e-300)) ;
}


/* diffusion_coeff
 *
 * Computes the flux-limited diffusion coefficient D = \lambda(J) / (k*rho), 
 * where \lambda(J) is the flux limiter.
 */
template<typename FieldType, typename... Indices>
inline __device__ 
double diffusion_coeff(GridRef& g, FieldType& J, double krho,
                       int i, int j, Indices... k) {

    double gJ = gradJ(g, J, i, j, k...) ;

    if (gJ < 2*krho)
        return 2 / (3*krho + sqrt(9*krho*krho + 10*gJ*gJ)) ;
    else 
        return 10 / (9*krho + 10*gJ + sqrt(81*krho*krho + 180*gJ*krho)) ;
}


/* Kernels for computing the flux-limited diffusion co-efficients in the
 * grey or multi-band approximations,
 *
 * Computes the flux-limited diffusion coefficient D = \lambda(J) / (k*rho), 
 * where \lambda(J) is the flux limiter.
 */
__global__ void compute_diffusion_coeff(GridRef g, FieldConstRef<double> J,
                                        FieldConstRef<double> rho,  
                                        FieldConstRef<double> kappa_R,
                                        FieldRef<double> D) ;
                                        
__global__ void compute_diffusion_coeff(GridRef g, Field3DConstRef<double> J,
                                        FieldConstRef<double> rho,  
                                        Field3DConstRef<double> kappa_ext,
                                        Field3DRef<double> D) ;


#endif //_CUDISC_HEADERS_FLD_DEVICE_H_