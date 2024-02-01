
#include <cassert>
#include <iostream>
#include <fstream>
#include <stdexcept>

#include <cuda_runtime.h>

#include "constants.h"
#include "diffusion_device.h"
#include "FLD.h"
#include "FLD_device.h"
#include "grid.h"
#include "planck.h"
#include "reductions.h"
#include "super_stepping.h"
#include "utils.h"
#include "timing.h"

__global__ void compute_FLD_rate(GridRef g,
                                 Field3DConstRef<double> J,
                                 Field3DConstRef<double> D, 
                                 double T_ext,
                                 PlanckIntegralRef planck, const double* wle,
                                 int boundary, double c_factor,
                                 Field3DRef<double> rate) {

    int j = threadIdx.x + blockIdx.x*blockDim.x ;
    int i = threadIdx.y + blockIdx.y*blockDim.y ;

    int b = threadIdx.z + blockIdx.z*blockDim.z ;

    
    int num_bands = J.Nd ;
    
    if (i >= g.Nghost && i < g.NR + g.Nghost &&
        j >= g.Nghost && j < g.Nphi + g.Nghost) {

        /////// Radiation density equation
        //    dJ/dt = c*grad[D grad(J)]
        for (; b < num_bands; b+=blockDim.z*gridDim.z) {

        // Compute diffusion matrix elements
        double Dij[3][3] ;
        compute_diffusion_matrix(g, D, Dij, boundary, i, j, b) ;


        // Compute the diffusion rate
        double diff = 0 ;
        if (i > g.Nghost) {
            if (j > g.Nghost) 
                diff += Dij[0][0] * J(i-1,j-1,b) ;

            diff += Dij[0][1] * J(i-1,j,b);

            if (j < g.Nphi + g.Nghost - 1) 
                diff += Dij[0][2] * J(i-1,j+1,b) ;
        }

        if (j > g.Nghost) 
            diff += Dij[1][0] * J(i,j-1,b) ;

        diff += Dij[1][1] * J(i,j,b) ;

        if (j < g.Nphi + g.Nghost - 1) 
            diff += Dij[1][2] * J(i,j+1,b) ;

        if (i < g.NR + g.Nghost - 1) {
            if (j > g.Nghost) 
                diff += Dij[2][0] * J(i+1,j-1,b) ;

            diff += Dij[2][1] * J(i+1,j,b) ;

            if (j < g.Nphi + g.Nghost - 1) 
                diff += Dij[2][2] * J(i+1,j+1,b);
        }


        //////////////////////////////////////
        // Handle the boundaries. For:
        //   - open boundaries:   add the external flux to the rhs
        //   - closed boundaries: Handled by compute_diffusion_matrix.
        //
        // Note that there are no corner cases

        double J_ext ;
        if (T_ext > 0)
            J_ext = 4*sigma_SB*pow(T_ext,4) *
                planck_factor(planck, T_ext, b, num_bands, wle) ;
        else
            J_ext = 0 ;

        if (i == g.Nghost) {
            if (boundary & BoundaryFlags::open_R_inner)
                diff += J_ext * (Dij[0][0] + Dij[0][1]) ;
        }
        if (i == g.NR + g.Nghost - 1) {
            if (boundary & BoundaryFlags::open_R_outer)
                diff += J_ext * (Dij[2][1] + Dij[2][2]) ;
        }
        if (j == g.Nghost) {
            if (boundary & BoundaryFlags::open_Z_inner)
                diff += J_ext * (Dij[0][0] + Dij[1][0]) ;
        }
        if (j == g.Nphi + g.Nghost - 1) {
            if (boundary & BoundaryFlags::open_Z_outer)
                diff += J_ext * (Dij[1][2] + Dij[2][2]) ;
        }

        rate(i, j, b) = - diff * c_light * c_factor / g.volume(i, j) ;

        if (false && i<= g.Nghost + 1 && j <= g.Nghost + 1) {
            printf("% d %d: (%4.3e %4.3e %4.3e, %4.3e %4.3e %4.3e, %4.3e %4.3e %4.3e), %g %g\n",
                  i, j, 
                  Dij[0][0],  Dij[1][0], Dij[2][0],
                  Dij[0][1],  Dij[1][1], Dij[2][1],
                  Dij[0][2],  Dij[1][2], Dij[2][2],
                  rate(i,j,b), D(i,j, b)) ;
            printf("% d %d: %g\n", i,j,J(i,j,b)) ; 

        }

    }} // End loop over bands
}


__global__ void apply_FLD_source_terms(GridRef g, double dt, double Cv, 
                                       FieldRef<double> T, Field3DRef<double> J,
                                       FieldConstRef<double> rho, 
                                       Field3DConstRef<double> opacity, 
                                       FieldConstRef<double> heat, 
                                       Field3DConstRef<double> scattering,
                                       PlanckIntegralRef planck, const double* wle,
                                       double c_factor) {

    int j = threadIdx.x + blockIdx.x*blockDim.x ;
    int i = threadIdx.y + blockIdx.y*blockDim.y ;
    
    int num_bands = J.Nd ;
    
    if (i >= g.Nghost && i < g.NR + g.Nghost &&
        j >= g.Nghost && j < g.Nphi + g.Nghost) {
    
        // Solves the following equaions:
        //
        ////// Gas temperature equation
        //    (rho*Cv) (T1 - T0) / dt = - rho kp [4*sigma*T1**4 - J1] + heat 
        //
        ////// Radiation density equation
        //    (J1-J0) / (c dt) = rho kp [4*sigma*T1**4 - J1] + scattering 

        // Step 1: Write J1 in terms of T1**4 and sum:
        //  kp*J1 = kp*[(J0 + c*dt*scattering) + kp*rho*c*dt*4*sigma * T1**4] / (1 + rho*kp*c*dt)
        double Etot = 0, kP = 0 ;
        double cdt = c_light*c_factor*dt ;
        for (int b=0; b < num_bands; b++) {
            double f = 1/(1 + rho(i,j)*opacity(i,j,b) * cdt) ;

            Etot += opacity(i,j,b) * (J(i,j,b) + cdt*scattering(i,j,b)) * f ;
            kP += opacity(i,j,b) * planck_factor(planck, T(i,j), b, num_bands, wle) * f;
        }
        
        // Step 2: Iterate to solve for the new temperature using a linearization.
        double CvT0 = Cv*T(i,j) + dt * (Etot + heat(i,j) / rho(i,j)) ;

        double T1 = T(i,j) ;
        for (int n=0; n < 20; n++) {
            double f = dt * 4*sigma_SB * T1*T1*T1 * kP ;
            T1 = (CvT0 + 3*f*T1) / (Cv + 4*f) ;
        }
        
        // Step 3: Compute the new J and store T
        Etot = 4*sigma_SB*T1*T1*T1*T1 ;
        double E0 = 0 ;
        double T0 = T(i, j) ;
        for (int b=0; b < num_bands; b++) {
            E0 += J(i,j,b) ;
            double f = 1/(1 + rho(i,j)*opacity(i,j,b) * cdt) ;
            double fp = planck_factor(planck, T(i,j), b, num_bands, wle) ;

            J(i,j,b) = (J(i,j,b) + cdt*scattering(i,j,b) + cdt*rho(i,j)*opacity(i,j,b)*Etot*fp) * f ;
        }
        T(i, j) = T1 ;

        
        if (false && i <= g.Nghost + 1 && j == 10*g.Nghost + 1) {
            printf("% d %d: %g %g, %g %g, %g\n", i, j, T0, E0, T(i, j), J(i,j,0), rho(i,j)*opacity(i,j,0) * cdt) ;
            printf("%d %d: %g %g %g\n", i,j, rho(i,j)*Cv*T0 + E0/(c_light*c_factor) + heat(i,j)*dt + scattering(i,j,0)*dt,
                    rho(i,j)*Cv*T1 + J(i,j,0) /(c_light*c_factor),  heat(i,j)*dt + scattering(i,j,0)*dt) ;


        }

    }

}




__global__ void fill_boundaries_FLD(GridRef g, FieldRef<double> T, Field3DRef<double> J) {

    int j = threadIdx.x + blockIdx.x*blockDim.x ;
    int i = threadIdx.y + blockIdx.y*blockDim.y ;                                       

    int num_bands = J.Nd ;
    
    if (i >= g.Nghost && i < g.NR + g.Nghost && 
        j >= g.Nghost && j < g.Nphi + g.Nghost) {
        

        // Reflecting
        if (j < 2*g.Nghost) {
            T(i,2*g.Nghost-1-j) = T(i,j) ;
            for(int b=0; b < num_bands; b++)
                J(i,2*g.Nghost-1-j, b) = J(i,j, b) ;

            // Corners
            if (i == g.Nghost)
                for (int ib=0; ib < g.Nghost; ib++) {
                    T(ib,2*g.Nghost-1-j) = T(i,j) ;
                    for(int b=0; b < num_bands; b++)
                        J(ib,2*g.Nghost-1-j,b) = J(i,j,b) ;
                }
            if (i == g.NR + g.Nghost-1)
                for (int ib=0; ib < g.Nghost; ib++) {
                    T(g.NR + g.Nghost + ib,2*g.Nghost-1-j) = T(i,j) ;
                    for(int b=0; b < num_bands; b++)
                        J(g.NR + g.Nghost + ib,2*g.Nghost-1-j, b) = J(i,j,b) ;
                }
        }
        // Copy
        if (j == g.Nphi + g.Nghost-1) {
            for (int jb=0; jb < g.Nghost; jb++) {
                T(i,g.Nphi + g.Nghost + jb) = T(i,j) ;
                for(int b=0; b < num_bands; b++)
                    J(i,g.Nphi + g.Nghost + jb,b) = J(i,j,b) ;

                // Corners
                if (i == g.Nghost)
                    for (int ib=0; ib < g.Nghost; ib++) {
                        T(ib,g.Nphi + g.Nghost + jb) = T(i,j) ;
                        for(int b=0; b < num_bands; b++)
                            J(ib,g.Nphi + g.Nghost + jb, b) = J(i,j,b) ;
                    }
                if (i == g.NR + g.Nghost-1)
                    for (int ib=0; ib < g.Nghost; ib++) {
                        T(g.NR + g.Nghost + ib,g.Nphi + g.Nghost + jb) = T(i,j) ;
                        for(int b=0; b < num_bands; b++)
                            J(g.NR + g.Nghost + ib,g.Nphi + g.Nghost + jb, b) = J(i,j,b) ;
                    }
            }
        }
        if (i == g.Nghost)
            for (int ib=0; ib < g.Nghost; ib++) {
                T(ib,j) = T(i,j) ;
                for(int b=0; b < num_bands; b++)
                    J(ib,j, b) = J(i,j, b) ;
            }
        if (i == g.NR + g.Nghost-1)
            for (int ib=0; ib < g.Nghost; ib++) {
                T(g.NR + g.Nghost + ib,j) = T(i,j) ;
                for(int b=0; b < num_bands; b++)
                    J(g.NR + g.Nghost + ib,j, b) = J(i,j, b) ;
            }
    }
}

template<typename FieldType>
double get_CFL_limit_FLD(const Grid& g, const FieldType& D) {

    CodeTiming::BlockTimer timing_block = 
        timer->StartNewTimer("get_CFL_limit_FLD") ;

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


    return dt / c_light;
}




int FLD_SuperStepping::operator()(
                    const Grid& g, double dt, double Cv, 
                    const Field<double>& rho, 
                    const Field<double>& kappa_P, const Field<double>& kappa_R,
                    const Field<double>& heat, 
                    Field<double>& T, Field<double>& J) {

    CodeTiming::BlockTimer timing_block = 
        timer->StartNewTimer("FLD_SuperStepping::operator()") ;

    dim3 threads(32,32,1) ;
    dim3 blocks((g.Nphi + 2*g.Nghost+31)/32,(g.NR + 2*g.Nghost+31)/32,1) ;                 

    // Set up storage for the diffusion matrix and scattering (0)
    Field<double> D = create_field<double>(g) ;
    compute_diffusion_coeff<<<blocks, threads>>>(g, J, rho, kappa_R, D) ;
    check_CUDA_errors("compute_diffusion_coeff") ;     

    Field<double> scattering = create_field<double>(g) ;
    set_all(g, scattering, 0) ;

    // Create the super-stepper
    //  Include a factor of 4 safety factor
    double dt_explicit = 0.25 * get_CFL_limit_FLD(g, D) / _c_fac ;  
    
    int steps = SuperStepping::num_steps_required(dt, dt_explicit) ;
    SuperStepping sts(g, steps) ; 
    

    // Cache critical variables.
    double c_factor = _c_fac ;
    double T_ext = _T_ext ;
    double boundary = _boundary ;
    PlanckInegral planck ;


    // Wrapper for the diffusion function for Super Time Stepping
    auto do_diffusion = 
        [&g, &rho, &kappa_R, &D, 
            &planck,T_ext, c_factor, boundary](const Field<double>& J, 
                                               Field<double>& rate) {

        // Compute the diffusion coefficient.
        dim3 threads(32,32,1) ;
        dim3 blocks((g.Nphi + 2*g.Nghost+31)/32,(g.NR + 2*g.Nghost+31)/32,1) ;  

        //compute_diffusion_coeff<<<blocks, threads>>>(g, J, rho, kappa_R, D) ;
        
        // Compute the diffusion rate
        threads = dim3(16,16,1) ;
        blocks = dim3((g.Nphi + 2*g.Nghost+15)/16,(g.NR + 2*g.Nghost+15)/16,1) ;  

        compute_FLD_rate<<<blocks, threads>>>(
            g, Field3DConstRef<double>(J), Field3DConstRef<double>(D), T_ext,
            planck, NULL, boundary, c_factor, 
            Field3DRef<double>(rate)) ; 

        check_CUDA_errors("compute_FLD_rate") ;
    } ;
    

    // Wrapper for the source terms for Super Time Stepping
    auto apply_source = 
        [&g, Cv, &rho, &kappa_P, &heat, &scattering, 
            &planck, c_factor, &T](Field<double>& J, double dt) {

        dim3 threads(16,16,1) ;
        dim3 blocks((g.Nphi + 2*g.Nghost+15)/16,(g.NR + 2*g.Nghost+15)/16,1) ;  

        apply_FLD_source_terms<<<blocks, threads>>>(
            g, dt, Cv, T, Field3DRef<double>(J),
            rho, Field3DConstRef<double>(kappa_P), 
            heat, Field3DConstRef<double>(scattering), 
            planck, NULL, c_factor) ;
        
        check_CUDA_errors("apply_FLD_source_terms") ;

        fill_boundaries_FLD<<<blocks, threads>>>(g, T, Field3DRef<double>(J)) ;
        check_CUDA_errors("fill_boundaries_FLD") ;
    } ;
        

    apply_source(J, dt/2) ;
    sts(do_diffusion, J, dt) ;
    apply_source(J, dt/2) ;

    //throw 1 ;
    return steps ;
}


int FLD_SuperStepping::solve_multi_band(
                        const Grid& g, double dt, double Cv, 
                        const Field<double>& rho, const Field3D<double>& kappa_abs, 
                        const Field<double>& heat, const CudaArray<double>& wle,
                        Field<double>& T, Field3D<double>& J) {


    Field3D<double> zeros = create_field3D<double>(g, J.Nd) ;
    set_all(g, zeros, 0) ;

    return solve_multi_band(g, dt, Cv, rho, kappa_abs, zeros, heat, zeros, wle, T, J) ;
}




int FLD_SuperStepping::solve_multi_band(
                        const Grid& g, double dt, double Cv, 
                        const Field<double>& rho, 
                        const Field3D<double>& kappa_abs, const Field3D<double>& kappa_ext,
                        const Field<double>& heat, const Field3D<double>& scattering, 
                        const CudaArray<double>& wle,
                        Field<double>& T, Field3D<double>& J) {

    CodeTiming::BlockTimer timing_block = 
        timer->StartNewTimer("FLD_SuperStepping::solve_multi_band()") ;
        
    dim3 threads(32,32,1) ;
    dim3 blocks((g.Nphi + 2*g.Nghost+31)/32,(g.NR + 2*g.Nghost+31)/32,1) ;                 

    // Set up storage for the diffusion matrix and scattering (0)
    Field3D<double> D = create_field3D<double>(g, J.Nd) ;
    compute_diffusion_coeff<<<blocks, threads>>>(g, J, rho, kappa_ext, D) ;
    check_CUDA_errors("compute_diffusion_coeff") ;     


    // Create the super-stepper
    //  Include a factor of 4 safety factor
    double dt_explicit = 0.25 * get_CFL_limit_FLD(g, D) / _c_fac ;  
    
    int steps = SuperStepping::num_steps_required(dt, dt_explicit) ;
    SuperStepping sts(g, steps) ; 
    

    // Cache critical variables.
    double c_factor = _c_fac ;
    double T_ext = _T_ext ;
    double boundary = _boundary ;
    PlanckInegral planck ;


    // Wrapper for the diffusion function for Super Time Stepping
    auto do_diffusion = 
        [&g, &rho, &kappa_abs, &kappa_ext, &D, 
            &planck, &wle, T_ext, c_factor, boundary](const Field3D<double>& J, 
                                                    Field3D<double>& rate) {

        // Compute the diffusion coefficient.
        dim3 threads(32,32,1) ;
        dim3 blocks((g.Nphi + 2*g.Nghost+31)/32,(g.NR + 2*g.Nghost+31)/32,1) ;  

        //compute_diffusion_coeff<<<blocks, threads>>>(g, J, rho, kappa_ext, D) ;
        
        // Compute the diffusion rate
        threads = dim3(16,16,1) ;
        blocks = dim3((g.Nphi + 2*g.Nghost+15)/16,(g.NR + 2*g.Nghost+15)/16,1) ;  

        compute_FLD_rate<<<blocks, threads>>>(
            g, J, D, T_ext,
            planck, wle.get(), boundary, c_factor, 
            rate) ; 

        check_CUDA_errors("compute_FLD_rate") ;
    } ;


    // Wrapper for the source terms for Super Time Stepping
    auto apply_source = 
        [&g, Cv, &rho, &kappa_abs, &heat, &scattering, 
            &planck, &wle, c_factor, &T](Field3D<double>& J, double dt) {

        dim3 threads(16,16,1) ;
        dim3 blocks((g.Nphi + 2*g.Nghost+15)/16,(g.NR + 2*g.Nghost+15)/16,1) ;  

        apply_FLD_source_terms<<<blocks, threads>>>(
            g, dt, Cv, T, J,
            rho, kappa_abs,  
            heat, scattering,
            planck, wle.get(), c_factor) ;
        
        check_CUDA_errors("apply_FLD_source_terms") ;

        fill_boundaries_FLD<<<blocks, threads>>>(g, T, Field3DRef<double>(J)) ;
        check_CUDA_errors("fill_boundaries_FLD") ;
    } ;
        

    apply_source(J, dt/2) ;
    sts(do_diffusion, J, dt) ;
    apply_source(J, dt/2) ;

    //throw 1 ;
    return steps ;
}