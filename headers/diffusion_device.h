
#ifndef  _CUDISC_DIFFUSION_DEVICE_H_
#define  _CUDISC_DIFFUSION_DEVICE_H_

#include "field.h"
#include "flags.h"
#include "grid.h"

/* Device functions for computing the diffusion coefficients on the 
 * non-orthogonal mesh. The scheme is based on:
 *    http://dx.doi.org/10.1016/j.jcp.2012.06.042
 *
 * Note: Variadatic templates are being used to support both Field and Field3D
 * types. The number of variadic arguments (k) is always zero or one and must 
 * match the field type.
*/


// Solve w0 * a + w1 * b = v0 for w0, w1
inline  __device__
vec decompose_vector(vec v0, vec a, vec b) {
    double det = 1 / (a.R*b.Z - b.R*a.Z) ;

    return { (+b.Z*v0.R - b.R*v0.Z) * det, 
             (-a.Z*v0.R + a.R*v0.Z) * det } ;

}

// Weighting for different components in the gradients

template<typename FieldType, typename... Indices>
__device__
double weights_Z(GridRef g, FieldType D, int i, int j, Indices... k) {
    if (j == g.Nghost)
        return 0 ;
    if (j == g.Nphi + 2*g.Nghost)
        return 1 ;

    double mu1 = (g.Ze(i,j) - g.Zc(i,j-1)) / D(i,j-1,k...) ;
    double mu2 = (g.Zc(i,j) - g.Ze(i,j)) / D(i,j,k...) ;
    
    return mu1/(mu1 + mu2) ;
}

template<typename FieldType, typename... Indices>
__device__ 
double weights_R(GridRef g,  FieldType D, int i, int j, Indices... k) {
    if (i == 0)
        return 0 ;
    if (i == g.NR + 2*g.Nghost)
        return 1 ;

    double mu1 = (g.Re(i) - g.Rc(i-1)) / D(i-1,j,k...) ;
    double mu2 = (g.Rc(i) - g.Re(i)) / D(i,j,k...) ;
    
    return mu1/(mu1 + mu2) ;
}


struct fluxes {
   vec R, Z, w ;
} ;

// Flux-coefficients for the left and upper faces.
template<typename FieldType, typename... Indices>
__device__ 
fluxes left_side_flux(GridRef g, FieldType D, int i, int j, Indices... k) {

    double w_I = weights_R(g, D, i, j, k...) ;
    double w_J = weights_Z(g, D, i, j+1, k...) ;

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
    flux.R.R =    w_I *D(i,j,k...) * w_12.R ;
    flux.R.Z = (1-w_J)*D(i,j,k...) * w_12.Z ;
    
    w_12 = decompose_vector(g.face_normal_Z(i,j+1), dy1, dy2) ;
    flux.Z.R =    w_I *D(i,j,k...) * w_12.R ;
    flux.Z.Z = (1-w_J)*D(i,j,k...) * w_12.Z ;

    return flux ;
}

// Flux-coefficients for the right and lower faces.
template<typename FieldType, typename... Indices>
__device__ 
fluxes right_side_flux(GridRef g, FieldType D, int i, int j, Indices... k) {

    double w_I = weights_R(g, D, i+1, j, k...) ;
    double w_J = weights_Z(g, D, i, j, k...) ;

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
    flux.R.R = (1-w_I)*D(i,j,k...) * w_12.R ;
    flux.R.Z =    w_J *D(i,j,k...) * w_12.Z ;

    w_12 = decompose_vector(g.face_normal_Z(i,j)*-1, dy1, dy2) ;
    flux.Z.R = (1-w_I)*D(i,j,k...) * w_12.R ;
    flux.Z.Z =    w_J *D(i,j,k...) * w_12.Z ;

    return flux ;
}

// Scheme based on: http://dx.doi.org/10.1016/j.jcp.2012.06.042
template<typename FieldType, typename... Indices>
__device__ 
void compute_diffusion_matrix(GridRef& g, FieldType D, 
                              double (&Dij)[3][3], int boundary, 
                              int i, int j, Indices... k) {
                                
    for (int l=0; l<3; l++)
        for (int m=0; m<3; m++) 
            Dij[l][m] = 0 ;

    fluxes fi = left_side_flux(g, D, i, j, k...) ;

    if (i > g.Nghost) {
        Dij[0][1] -= (1 - fi.w.R) * fi.R.R * g.area_R(i,j) ;
        Dij[1][1] += (1 - fi.w.R) * fi.R.R * g.area_R(i,j) ;

        Dij[1][2] -= (1 - fi.w.R) * fi.R.Z * g.area_R(i,j) ;
        Dij[1][1] += (1 - fi.w.R) * fi.R.Z * g.area_R(i,j) ;

        fluxes fj = right_side_flux(g, D, i-1, j, k...) ;

        Dij[0][1] -= fj.w.R * fj.R.R * g.area_R(i,j) ;
        Dij[1][1] += fj.w.R * fj.R.R * g.area_R(i,j) ;
        
        Dij[0][1] -= fj.w.R * fj.R.Z * g.area_R(i,j) ;
        Dij[0][0] += fj.w.R * fj.R.Z * g.area_R(i,j) ;
        
    } else {
        Dij[0][1] -= fi.R.R * g.area_R(i,j) ;
        Dij[1][1] += fi.R.R * g.area_R(i,j) ;
    }

    if (j < g.Nphi + g.Nghost - 1) {
        Dij[0][1] -= fi.w.Z * fi.Z.R * g.area_Z(i, j+1) ;
        Dij[1][1] += fi.w.Z * fi.Z.R * g.area_Z(i, j+1) ;

        Dij[1][2] -= fi.w.Z * fi.Z.Z * g.area_Z(i, j+1) ;
        Dij[1][1] += fi.w.Z * fi.Z.Z * g.area_Z(i, j+1) ;

        fluxes fj = right_side_flux(g, D, i, j+1, k...) ;

        Dij[1][2] -= (1-fj.w.Z) * fj.Z.Z * g.area_Z(i, j+1) ;
        Dij[1][1] += (1-fj.w.Z) * fj.Z.Z * g.area_Z(i, j+1) ;

        Dij[1][2] -= (1-fj.w.Z) * fj.Z.R * g.area_Z(i, j+1) ;
        Dij[2][2] += (1-fj.w.Z) * fj.Z.R * g.area_Z(i, j+1) ;
    } else {
        Dij[1][2] -= fi.Z.Z * g.area_Z(i, j+1) ;
        Dij[1][1] += fi.Z.Z * g.area_Z(i, j+1) ;
    }

    fi = right_side_flux(g, D, i, j, k...) ;

    if (i < g.NR + g.Nghost - 1) {
        Dij[2][1] -= fi.w.R * fi.R.R * g.area_R(i+1,j) ;
        Dij[1][1] += fi.w.R * fi.R.R * g.area_R(i+1,j) ;

        Dij[1][0] -= fi.w.R * fi.R.Z * g.area_R(i+1,j) ;
        Dij[1][1] += fi.w.R * fi.R.Z * g.area_R(i+1,j) ;

        fluxes fj = left_side_flux(g, D, i+1, j, k...) ;

        Dij[2][1] -= (1-fj.w.R) * fj.R.R * g.area_R(i+1,j) ;
        Dij[1][1] += (1-fj.w.R) * fj.R.R * g.area_R(i+1,j) ;

        Dij[2][1] -= (1-fj.w.R) * fj.R.Z * g.area_R(i+1,j) ;
        Dij[2][2] += (1-fj.w.R) * fj.R.Z * g.area_R(i+1,j) ;
    } else {
        Dij[2][1] -= fi.R.R * g.area_R(i+1,j) ;
        Dij[1][1] += fi.R.R * g.area_R(i+1,j) ;
    }

    if (j > g.Nghost) {
        Dij[2][1] -= (1-fi.w.Z) * fi.Z.R * g.area_Z(i,j) ;
        Dij[1][1] += (1-fi.w.Z) * fi.Z.R * g.area_Z(i,j) ;

        Dij[1][0] -= (1-fi.w.Z) * fi.Z.Z * g.area_Z(i,j) ;
        Dij[1][1] += (1-fi.w.Z) * fi.Z.Z * g.area_Z(i,j) ;

        fluxes fj = left_side_flux(g, D, i, j-1, k...) ;

        Dij[1][0] -= fj.w.Z * fj.Z.Z * g.area_Z(i,j) ;
        Dij[1][1] += fj.w.Z * fj.Z.Z * g.area_Z(i,j) ;

        Dij[1][0] -= fj.w.Z * fj.Z.R * g.area_Z(i,j) ;
        Dij[0][0] += fj.w.Z * fj.Z.R * g.area_Z(i,j) ;
    } else {
        Dij[1][0] -= fi.Z.Z * g.area_Z(i,j) ;
        Dij[1][1] += fi.Z.Z * g.area_Z(i,j) ;
    }

    /////////////////////////////////
    // Apply closed boundaries:
    //   - subtract the boundary terms from the matrix
    //
    // Note that there are no corner cases because of the above if statements.

    if (!(boundary & BoundaryFlags::open_R_inner))
        if (i == g.Nghost) {
            Dij[1][0] += Dij[0][0] ;
            Dij[1][1] += Dij[0][1] ;
        }
    if (!(boundary & BoundaryFlags::open_R_outer)) 
        if (i == g.NR + g.Nghost - 1) {
            Dij[1][1] += Dij[2][1] ;
            Dij[1][2] += Dij[2][2] ;
        }
    
    if (!(boundary & BoundaryFlags::open_Z_inner))
        if (j == g.Nghost) {
            Dij[0][1] += Dij[0][0] ;
            Dij[1][1] += Dij[1][0] ;
        }
    
    if (!(boundary & BoundaryFlags::open_Z_outer))
        if (j == g.Nphi + g.Nghost - 1) {
            Dij[1][1] += Dij[1][2] ;
            Dij[2][1] += Dij[2][2] ;
        }

}


/* _compute_CFL_limit_diffusion
 *
 * Get the CFL limit for each cell for a diffusion process.
 */
__global__ 
void _compute_CFL_limit_diffusion(GridRef g, Field3DConstRef<double>D, 
                                  FieldRef<double> CFL) ;

#endif// _CUDISC_DIFFUSION_DEVICE_H_