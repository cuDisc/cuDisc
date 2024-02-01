
#include <cassert>
#include <iostream>
#include <fstream>
#include <stdexcept>

#include <cuda_runtime.h>

#include "constants.h"
#include "diffusion_device.h"
#include "grid.h"
#include "advection.h"
#include "reductions.h"
#include "utils.h"
#include "super_stepping.h"
#include "timing.h"



__global__ void compute_Drho_gas(GridRef g, FieldConstRef<double> D,
                                 FieldConstRef<double> rho,  
                                 FieldRef<double> Drho) {
                                             
    int j = threadIdx.x + blockIdx.x*blockDim.x ;
    int i = threadIdx.y + blockIdx.y*blockDim.y ;

    if (i < g.NR + 2*g.Nghost && j < g.Nphi + 2*g.Nghost) {
        Drho(i,j) = D(i,j)*rho(i,j) ;
    }
}

__device__ double _compute_diffusion_rate(GridRef g, double Dij[3][3], 
                                          FieldConstRef<double> rho_g, 
                                          Field3DConstRef<double> rho_d,
                                          int i, int j, int k) {

    double drho = 0 ;

    if (i >= g.Nghost && i < g.NR + g.Nghost  &&
        j >= g.Nghost && j < g.Nphi + g.Nghost) {

        if (k < rho_d.Nd) {

            if (i > g.Nghost) {
                if (j > g.Nghost) 
                    drho += Dij[0][0] * rho_d(i-1,j-1,k)/rho_g(i-1,j-1) ;

                drho += Dij[0][1] * rho_d(i-1,j,k)/rho_g(i-1,j) ;

                if (j < g.Nphi + g.Nghost - 1) 
                    drho += Dij[0][2] * rho_d(i-1,j+1,k)/rho_g(i-1,j+1) ;
            }            


            if (j > g.Nghost) 
                drho += Dij[1][0] * rho_d(i,j-1,k)/rho_g(i,j-1) ;

            drho += Dij[1][1] * rho_d(i,j,k)/rho_g(i,j) ;

            if (j < g.Nphi + g.Nghost - 1) 
                drho += Dij[1][2] * rho_d(i,j+1,k)/rho_g(i,j+1) ;

            if (i < g.NR + g.Nghost - 1) {
                if (j > g.Nghost) 
                    drho += Dij[2][0] * rho_d(i+1,j-1,k)/rho_g(i+1,j-1) ;

                drho += Dij[2][1] * rho_d(i+1,j,k)/rho_g(i+1,j) ;

                if (j < g.Nphi + g.Nghost - 1) 
                    drho += Dij[2][2] * rho_d(i+1,j+1,k)/rho_g(i+1,j+1) ;
            }
        }
        drho = - drho/g.volume(i,j) ;
    }

    return drho ;

}

__global__ void compute_diffusive_flux_update(GridRef g, FieldConstRef<double> D, 
                                       FieldConstRef<double> rho_g, 
                                       Field3DConstRef<double> rho_d,
                                       Field3DRef<double> rho_new,
                                       double dt) {

    int j = threadIdx.x + blockIdx.x*blockDim.x ;
    int i = threadIdx.y + blockIdx.y*blockDim.y ;

    if (i >= g.Nghost && i < g.NR + g.Nghost  &&
        j >= g.Nghost && j < g.Nphi + g.Nghost) {
        
        // Temporary storage for diffusion matrix elements
        double Dij[3][3] ;
        compute_diffusion_matrix(g, D, Dij, BoundaryFlags::all_closed, i, j) ;

        for (int k=0; k < rho_d.Nd; k++) {
            rho_new(i,j,k) = rho_d(i,j,k) + dt * _compute_diffusion_rate(g, Dij, rho_g, rho_d, i, j, k) ;
        }
    }
}


void Diffusion::operator()(const Grid& g, Field<double>& rho_dust,
                           const Field<double>& rho_gas, const Field<double>& D, 
                           double dt) {

    CodeTiming::BlockTimer timing_block = 
        timer->StartNewTimer("Diffusion::operator()") ;

    dim3 threads(32,32,1) ;
    dim3 blocks((g.Nphi + 2*g.Nghost + 31)/32,(g.NR + 2*g.Nghost + 31)/32,1) ;

    Field<double> Drho = create_field<double>(g) ;
    compute_Drho_gas<<<blocks, threads>>>(g, D, rho_gas, Drho) ;
    check_CUDA_errors("compute_Drho_gas") ;           

    dim3 threads2(16,16,1) ;
    dim3 blocks2((g.Nphi + 2*g.Nghost + 15)/16,(g.NR + 2*g.Nghost + 15)/16,1) ;

    Field<double> rho_new = create_field<double>(g) ;
    compute_diffusive_flux_update<<<blocks2, threads2>>>(g, Drho, rho_gas, 
        Field3DConstRef<double>(rho_dust), Field3DRef<double>(rho_new), dt) ;
 
    check_CUDA_errors("compute_diffusive_flux_update") ;    

    // Copy back new density
    copy_field(g, rho_new, rho_dust) ;
}


void Diffusion::operator()(const Grid& g, Field3D<double>& rho_dust,
                           const Field<double>& rho_gas, const Field<double>& D, 
                           double dt) {

    CodeTiming::BlockTimer timing_block = 
    timer->StartNewTimer("Diffusion::operator()") ;

    dim3 threads(32,32,1) ;
    dim3 blocks((g.Nphi + 2*g.Nghost + 31)/32,(g.NR + 2*g.Nghost + 31)/32,1) ;

    Field<double> Drho = create_field<double>(g) ;
    compute_Drho_gas<<<blocks, threads>>>(g, D, rho_gas, Drho) ;
    check_CUDA_errors("compute_Drho_gas") ;           

    dim3 threads2(16,16,1) ;
    dim3 blocks2((g.Nphi + 2*g.Nghost + 15)/16,(g.NR + 2*g.Nghost + 15)/16,1) ;

    Field3D<double> rho_new = create_field3D<double>(g,rho_dust.Nd) ;
    compute_diffusive_flux_update<<<blocks2, threads2>>>(g, Drho, rho_gas, 
                                                  rho_dust, rho_new, dt) ;

    check_CUDA_errors("compute_diffusive_flux_update") ;    

    // Copy back new density
    copy_field(g, rho_new, rho_dust) ;
}

__global__ void compute_diffusion_rate(GridRef g, FieldConstRef<double> D, 
                                       FieldConstRef<double> rho_g, 
                                       Field3DConstRef<double> rho_d,
                                       Field3DRef<double> drho_dt) {

    int j = threadIdx.x + blockIdx.x*blockDim.x ;
    int i = threadIdx.y + blockIdx.y*blockDim.y ;

    if (i >= g.Nghost && i < g.NR + g.Nghost  &&
        j >= g.Nghost && j < g.Nphi + g.Nghost) {

        double Dij[3][3] ;
        compute_diffusion_matrix(g, D, Dij, BoundaryFlags::all_closed, i, j) ;

        for (int k=0; k < rho_d.Nd; k++) {
            drho_dt(i,j,k) = _compute_diffusion_rate(g, Dij, rho_g, rho_d, i, j, k) ;
        }
    }
}

int Diffusion::update_sts(const Grid& g, Field<double>& rho_dust,
                         const Field<double>& rho_gas, const Field<double>& D, 
                         double dt) {

     CodeTiming::BlockTimer timing_block = 
        timer->StartNewTimer("Diffusion::operator()") ;
 
    // Create the super-stepper
    double dt_explicit = get_CFL_limit(g, D) ;
    
    int steps = SuperStepping::num_steps_required(dt, dt_explicit) ;
    SuperStepping sts(g, steps) ;

    dim3 threads(32,32,1) ;
    dim3 blocks((g.Nphi + 2*g.Nghost + 31)/32,(g.NR + 2*g.Nghost + 31)/32,1) ;

    Field<double> Drho = create_field<double>(g) ;
    compute_Drho_gas<<<blocks, threads>>>(g, D, rho_gas, Drho) ;
    check_CUDA_errors("compute_Drho_gas") ;           

    threads = dim3(16,16,1) ;
    blocks  = dim3((g.Nphi + 2*g.Nghost + 15)/16,(g.NR + 2*g.Nghost + 15)/16,1) ;

    auto rate = [blocks, threads, &g, &rho_gas, &Drho](
        const Field<double>& rho_d, Field<double>& drho_dt) {
        
        compute_diffusion_rate<<<blocks, threads>>>(g, Drho, rho_gas, 
            Field3DConstRef<double>(rho_d), Field3DRef<double>(drho_dt)) ;

        check_CUDA_errors("compute_diffusion_rate") ;    
    } ;

    // Update using super-time stepping
    sts(rate, rho_dust, dt) ;

    return steps ;
}


int Diffusion::update_sts(const Grid& g, Field3D<double>& rho_dust,
                           const Field<double>& rho_gas, const Field<double>& D, 
                           double dt) {

    CodeTiming::BlockTimer timing_block = 
        timer->StartNewTimer("Diffusion::update_sts") ;

    // Create the super-stepper
    double dt_explicit = get_CFL_limit(g, D) ;
    
    int steps = SuperStepping::num_steps_required(dt, dt_explicit) ;
    SuperStepping sts(g, steps) ;

    // Compute the diffusion coefficient
    dim3 threads(32,32,1) ;
    dim3 blocks((g.Nphi + 2*g.Nghost + 31)/32,(g.NR + 2*g.Nghost + 31)/32,1) ;

    Field<double> Drho = create_field<double>(g) ;
    compute_Drho_gas<<<blocks, threads>>>(g, D, rho_gas, Drho) ;
    check_CUDA_errors("compute_Drho_gas") ;   

    // Create wrapper to the diffusion rate
    threads = dim3(16,16,1) ;
    blocks = dim3((g.Nphi + 2*g.Nghost + 15)/16,(g.NR + 2*g.Nghost + 15)/16,1) ;

    auto rate = [blocks, threads, &g, &rho_gas, &Drho](
        const Field3D<double>& rho_d, Field3D<double>& drho_dt) {
        
        compute_diffusion_rate<<<blocks, threads>>>(
            g, Drho, rho_gas, rho_d, drho_dt) ;

        check_CUDA_errors("compute_diffusion_rate") ;    
    } ;

    // Update using super-time stepping
    sts(rate, rho_dust, dt) ;

    return steps ;
}




__global__ 
void _compute_CFL_limit_diffusion(GridRef g, Field3DConstRef<double>D, FieldRef<double> CFL) {

    int j = threadIdx.x + blockIdx.x*blockDim.x ;
    int i = threadIdx.y + blockIdx.y*blockDim.y ;

    double dx, CFL1, CFL2 ;
    if (i < g.NR + 2*g.Nghost && j < g.Nphi + 2*g.Nghost) {
        for (int k=0; k < D.Nd; k++) {
            dx = g.Re(i+1)-g.Re(i) ;
            CFL1 = abs(dx*dx/D(i,j,k)) ;

            dx = g.Ze(i,j+1)-g.Ze(i,j) ;
            CFL2 = abs(dx*dx/D(i,j,k)) ;
            
            CFL(i,j) = 0.25 * min(CFL1, CFL2) ;
        }
    } 
} 

double Diffusion::get_CFL_limit(const Grid& g, const Field<double>& D) {

    CodeTiming::BlockTimer timing_block = 
        timer->StartNewTimer("Diffusion::get_CFL_limit") ;

    // Temporary work-space for CFL condition 
    Field<double> CFL = create_field<double>(g) ;

    dim3 threads(32,32,1) ;
    dim3 blocks((g.Nphi+2*g.Nghost + 31)/32, (g.NR+2*g.Nghost + 31)/32, 1) ;

    // Get CFL condition for each cell
    _compute_CFL_limit_diffusion<<<blocks, threads>>>(
        g, Field3DConstRef<double>(D), CFL
    ) ;
    check_CUDA_errors("_compute_CFL_limit_diffusion") ;

    // Compute the minimum CFL for each cell
    Reduction::scan_R_min(g, CFL) ;

    // Get the global minimum
    int NR = g.NR + 2*g.Nghost ;
    int NZ = g.Nphi + 2*g.Nghost ;

    double dt = CFL(NR-1, 0) ;
    for (int j=0; j < NZ; j++) {
        dt = std::min(dt, CFL(NR-1, j)) ;
    }


    return _CFL * dt ;
}

