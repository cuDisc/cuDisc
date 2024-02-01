
#include <iostream>
#include <cuda_runtime.h>

#include "advection.h"
#include "field.h"
#include "grid.h"
#include "reductions.h"
#include "scan.h"
#include "timing.h"
#include "utils.h"


/////////////////////////////////////////////////////////////////////////////////////////
///// Advection using van-Leer integration
/////////////////////////////////////////////////////////////////////////////////////////
__device__ inline
double _vanleer_slope(double dQF, double dQB, double cF, double cB) {

    if (dQF*dQB > 0) {
        double v = dQB/dQF ;
        return dQB * (cF*v + cB) / (v*v + (cF + cB - 2)*v + 1) ;
    } else {
        return 0 ;
    }
}

__device__ inline 
double _vanleer_slope_R(GridRef& g, Field3DConstRef<double>& Qty, int i, int j, int k) {
    double cF = (g.Rc(i+1) - g.Rc(i)) / (g.Re(i+1)-g.Rc(i)) ;
    double cB = (g.Rc(i-1) - g.Rc(i)) / (g.Re(i-1)-g.Rc(i)) ;

    double dQF = (Qty(i+1, j, k) - Qty(i, j, k)) / (g.Rc(i+1) - g.Rc(i)) ;
    double dQB = (Qty(i-1, j, k) - Qty(i, j, k)) / (g.Rc(i-1) - g.Rc(i)) ;

    return _vanleer_slope(dQF, dQB, cF, cB) ;
}

__device__ inline 
double _vanleer_slope_Z(GridRef& g, Field3DConstRef<double>& Qty, int i, int j, int k) {
    double cF = (g.Zc(i,j+1) - g.Zc(i,j)) / (g.Ze(i,j+1)-g.Zc(i,j)) ;
    double cB = (g.Zc(i,j-1) - g.Zc(i,j)) / (g.Ze(i,j-1)-g.Zc(i,j)) ;

    double dQF = (Qty(i,j+1, k) - Qty(i, j  , k)) / (g.Ze(i,j+1)-g.Zc(i,j)) ;
    double dQB = (Qty(i,j  , k) - Qty(i, j-1, k)) / (g.Ze(i,j-1)-g.Zc(i,j)) ;

    return _vanleer_slope(dQF, dQB, cF, cB) ;
}

__global__ 
void _compute_fluxes_donor(GridRef g, Field3DConstRef<double> v_R, Field3DConstRef<double> v_phi,
                           Field3DConstRef<double> Qty, Field3DRef<double> flux_R, Field3DRef<double> flux_Z) {

    int k = threadIdx.x + blockIdx.x*blockDim.x ; 
    int j = threadIdx.y + blockIdx.y*blockDim.y ;
    int i = threadIdx.z + blockIdx.z*blockDim.z ;

    int is, js ;
    if (i > 0 && i < g.NR + 2*g.Nghost && 
        j > 0 && j < g.Nphi + 2*g.Nghost &&
        k < Qty.Nd) {
        
        if (v_R(i, j, k) < 0)
            is = i ;
        else 
            is = i-1 ;
        
        if (v_phi(i, j, k) < 0)
            js = j ;
        else
            js = j-1 ;

        flux_R(i,j,k) = v_R(i, j, k) * g.area_R(i, j) * Qty(is, j, k);
        flux_Z(i,j,k) = v_phi(i, j, k) * g.area_Z(i, j) * Qty(i, js, k);
    }
} 


__global__ 
void _compute_fluxes_vanleer(GridRef g, Field3DConstRef<double> v_R, Field3DConstRef<double> v_phi,
                             Field3DConstRef<double> Qty, Field3DRef<double> flux_R, Field3DRef<double> flux_Z) {

    int k = threadIdx.x + blockIdx.x*blockDim.x ; 
    int j = threadIdx.y + blockIdx.y*blockDim.y ;
    int i = threadIdx.z + blockIdx.z*blockDim.z ;

    double Q ;
    int is, js ;
    if (i > 1 && i < g.NR + 2*g.Nghost - 1 && 
        j > 1 && j < g.Nphi + 2*g.Nghost - 1 &&
        k < Qty.Nd ) {
        
        if (v_R(i, j, k) < 0)
            is = i ;
        else 
            is = i-1 ;
        
        if (v_phi(i, j, k) < 0)
            js = j ;
        else
            js = j-1 ;

        Q = Qty(is, j, k) + _vanleer_slope_R(g, Qty, is, j, k) * (g.Re(i) - g.Rc(is)) ;
        flux_R(i, j, k) = v_R(i, j, k) * g.area_R(i, j) * Q ; 

        Q = Qty(i, js, k) + _vanleer_slope_Z(g, Qty, i, js, k) * (g.Ze(i,j) - g.Zc(i,js)) ;
        flux_Z(i, j, k) = v_phi(i, j, k) * g.area_Z(i, j) * Q ;
    }
} 
                               
__global__ 
void _update_conserved(GridRef g, Field3DConstRef<double>  Qty,
                       Field3DConstRef<double>  flux_R, Field3DConstRef<double>  flux_Z,
                       double dt, Field3DRef<double>  Qty_new) {
    
    int k = threadIdx.x + blockIdx.x*blockDim.x ; 
    int j = threadIdx.y + blockIdx.y*blockDim.y ;
    int i = threadIdx.z + blockIdx.z*blockDim.z ;

    if (i > 0 && i < g.NR + 2*g.Nghost - 1 && 
        j > 0 && j < g.Nphi + 2*g.Nghost - 1 &&
        k < Qty.Nd) {
        
        double dF = flux_R(i+1,j,k) - flux_R(i,j,k) + flux_Z(i,j+1,k) - flux_Z(i,j,k) ;

        Qty_new(i,j,k) = Qty(i,j,k) - dt * dF / g.volume(i,j) ;
    }
}

void VanLeerAdvection::operator()(const Grid& g, Field<double>& Qty,
                                  const Field<double>& v_R, const Field<double>& v_phi, double dt) {

    CodeTiming::BlockTimer timing_block = 
        timer->StartNewTimer("VanLeerAdvection::operator()") ;
                            
    // Temporary work-space for fluxes and the intermediate step:
    Field<double> fR = create_field<double>(g,Staggering::R) ;
    Field<double> fZ = create_field<double>(g,Staggering::phi) ;

    Field<double> Qty_mid = create_field<double>(g) ;

    dim3 threads(1,32,32) ;
    dim3 blocks(1,(g.Nphi+2*g.Nghost + 31)/32, (g.NR+2*g.Nghost + 31)/32) ;

    using Ref = Field3DRef<double> ;
    using ConstRef = Field3DConstRef<double> ;

    // Half step with Donor cell method
    _compute_fluxes_donor<<<blocks, threads>>>(
       g, ConstRef(v_R), ConstRef(v_phi), ConstRef(Qty), Ref(fR), Ref(fZ)
    ) ;
    check_CUDA_errors("_compute_fluxes_donor") ;

    _update_conserved<<<blocks, threads>>>(
        g, ConstRef(Qty), ConstRef(fR), ConstRef(fZ), dt/2, Ref(Qty_mid)
    ) ;        
    check_CUDA_errors("_update_conserved") ;

    // Full step with van-Leer
    _compute_fluxes_vanleer<<<blocks, threads>>>( 
        g, ConstRef(v_R), ConstRef(v_phi), ConstRef(Qty_mid), Ref(fR), Ref(fZ)
    ) ;
    check_CUDA_errors("_compute_fluxes_vanleer") ;

    _update_conserved<<<blocks, threads>>>(g, ConstRef(Qty), ConstRef(fR), ConstRef(fZ), dt/2, Ref(Qty)) ;        
    check_CUDA_errors("_update_conserved") ;
}

void VanLeerAdvection::operator()(const Grid& g, Field3D<double>& Qty,
                                  const Field3D<double>& v_R, const Field3D<double>& v_phi, double dt) {

    CodeTiming::BlockTimer timing_block = 
        timer->StartNewTimer("VanLeerAdvection::operator()") ;
                            
    // Temporary work-space for fluxes and the intermediate step:
    Field3D<double> fR = create_field3D<double>(g, Qty.Nd, Staggering::R) ;
    Field3D<double> fZ = create_field3D<double>(g, Qty.Nd, Staggering::phi) ;

    Field3D<double> Qty_mid = create_field3D<double>(g, Qty.Nd) ;

    // Setup threads
    dim3 threads(1,32,1) ;
    while (threads.x < Qty.Nd && threads.x < 32)
       threads.x *= 2 ;
    threads.z = 32/threads.x ;

    dim3 blocks(
        (Qty.Nd + threads.x-1)/threads.x,
        (g.Nphi+2*g.Nghost + threads.y-1)/threads.y, 
        (g.NR+2*g.Nghost + threads.z-1)/threads.z
    ) ;

    using Ref = Field3DRef<double> ;
    using ConstRef = Field3DConstRef<double> ;

    // Half step with Donor cell method
    _compute_fluxes_donor<<<blocks, threads>>>(
       g, ConstRef(v_R), ConstRef(v_phi), ConstRef(Qty), Ref(fR), Ref(fZ)
    ) ;
    check_CUDA_errors("_compute_fluxes_donor") ;

    _update_conserved<<<blocks, threads>>>(
        g, ConstRef(Qty), ConstRef(fR), ConstRef(fZ), dt/2, Ref(Qty_mid)
    ) ;        
    check_CUDA_errors("_update_conserved") ;

    // Full step with van-Leer
    _compute_fluxes_vanleer<<<blocks, threads>>>( 
        g, ConstRef(v_R), ConstRef(v_phi), ConstRef(Qty_mid), Ref(fR), Ref(fZ)
    ) ;
    check_CUDA_errors("_compute_fluxes_vanleer") ;

    _update_conserved<<<blocks, threads>>>(g, ConstRef(Qty), ConstRef(fR), ConstRef(fZ), dt/2, Ref(Qty)) ;        
    check_CUDA_errors("_update_conserved") ;
}

/////////////////////////////////////////////////////////////////////////////////////////
///// Computation of the CFL limit
/////////////////////////////////////////////////////////////////////////////////////////

__global__ 
void _compute_CFL_limit_VL(GridRef g, Field3DConstRef<double> v_R, Field3DConstRef<double> v_phi, FieldRef<double>  CFL) {

    int j = threadIdx.x + blockIdx.x*blockDim.x ;
    int i = threadIdx.y + blockIdx.y*blockDim.y ;

    double dx, CFL1, CFL2 ;
    if (i > 0 && i < g.NR + 2*g.Nghost && 
        j > 0 && j < g.Nphi + 2*g.Nghost) {
        
        double CFL_ij = min2<double>::identity() ;
        for (int k=0; k < v_R.Nd; k++) {
            dx = min(g.volume(i,j), g.volume(i-1,j)) /  g.area_R(i,j) ;
            CFL1 = abs(dx/v_R(i,j, k)) ;

            dx = min(g.volume(i,j), g.volume(i,j-1)) / g.area_Z(i,j) ;
            CFL2 = abs(dx/v_phi(i,j, k)) ;
            
            CFL_ij = min(CFL_ij, min(CFL1, CFL2)) ;
        }
        CFL(i,j) = CFL_ij ;
    } 
    else if (i < g.NR + 2*g.Nghost && j < g.Nphi + 2*g.Nghost) {
        CFL(i, j) = min2<double>::identity() ;
    }
} 

double VanLeerAdvection::get_CFL_limit(const Grid& g, 
                                       const Field<double>& v_R, 
                                       const Field<double>& v_phi) {

    CodeTiming::BlockTimer timing_block = 
        timer->StartNewTimer("VanLeerAdvection::get_CFL_limit") ;

    // Temporary work-space for CFL condition 
    Field<double> CFL = create_field<double>(g) ;

    dim3 threads(32,32,1) ;
    dim3 blocks((g.Nphi+2*g.Nghost + 31)/32, (g.NR+2*g.Nghost + 31)/32, 1) ;

    using ConstRef = Field3DConstRef<double> ;

    // Get CFL condition for each cell
    _compute_CFL_limit_VL<<<blocks, threads>>>(
        g, ConstRef(v_R), ConstRef(v_phi), CFL
    ) ;
    check_CUDA_errors("_compute_CFL_limit_VL") ;

    // Compute the minimum CFL for each cell
    Reduction::scan_R_min(g, CFL) ;

    // Get the global minimum
    int NR = g.NR + 2*g.Nghost ;
    int NZ = g.Nphi + 2*g.Nghost ;

    double dt = CFL(NR-1, 0) ;
    for (int j=0; j < NZ; j++) {
        dt = std::min(dt, CFL(NR-1, j)) ;
    }

    return _CFL* dt ;
}

double VanLeerAdvection::get_CFL_limit(const Grid& g, 
                                       const Field3D<double>& v_R, 
                                       const Field3D<double>& v_phi) {

    CodeTiming::BlockTimer timing_block = 
        timer->StartNewTimer("VanLeerAdvection::get_CFL_limit") ;

    // Temporary work-space for CFL condition 
    Field<double> CFL = create_field<double>(g) ;

    dim3 threads(32,32,1) ;
    dim3 blocks((g.Nphi+2*g.Nghost + 31)/32, (g.NR+2*g.Nghost + 31)/32, 1) ;

    using ConstRef = Field3DConstRef<double> ;

    // Get CFL condition for each cell
    _compute_CFL_limit_VL<<<blocks, threads>>>(
        g, ConstRef(v_R), ConstRef(v_phi), CFL
    ) ;
    check_CUDA_errors("_compute_CFL_limit_VL") ;

    // Compute the minimum CFL for each cell
    Reduction::scan_R_min(g, CFL) ;

    // Get the global minimum
    int NR = g.NR + 2*g.Nghost ;
    int NZ = g.Nphi + 2*g.Nghost ;

    double dt = CFL(NR-1, 0) ;
    for (int j=0; j < NZ; j++) {
        dt = std::min(dt, CFL(NR-1, j)) ;
    }
    
    return _CFL* dt ;
}
