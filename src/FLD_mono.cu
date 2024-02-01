
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
#include "matrix_types.h"
#include "pcg_solver.h"
#include "utils.h"
#include "timing.h"



CSR_SpMatrix _create_FLD_mono_matrix(const Grid& g) {

    int nx = g.NR ;
    int ny = g.Nphi ;

    assert(nx > 1 && ny > 1) ;

    int non_zero = 
        + 4*nx*ny // Radiation-matter coupling
        + 8*(nx-2)*(ny-2) + 5*2*(nx + ny - 4) + 3*4 ; // FLD

    auto ngb = [nx, ny](int i, int j) {
        int edges = 
            (i == 0) + (i == nx - 1)  + (j == 0) + (j == ny - 1) ;

        switch (edges) {
            case 0:
                return 8;
            case 1:
                return 5;
            case 2:
                return 3;
            default:
                throw std::invalid_argument("Should never occur") ;
                return -1 ;
        }
    } ;

    CSR_SpMatrix mat(2*nx*ny, 2*nx*ny, non_zero) ;

    mat.csr_offset[0] = 0 ;
    for(int i=0; i < nx; i++) 
        for (int j=0; j < ny; j++) {
            int k = 2*(i*ny + j) ;
            mat.csr_offset[k+1] = mat.csr_offset[k] + 2 ;
            mat.csr_offset[k+2] = mat.csr_offset[k+1] + 2 + ngb(i,j) ;      
        }
       
    assert (mat.csr_offset[2*nx*ny] == mat.csr_offset[0] + non_zero) ;

    return mat ;
}


__global__ void compute_diffusion_coeff(GridRef g, FieldConstRef<double> J,
                                        FieldConstRef<double> rho,  
                                        FieldConstRef<double> kappa_R,
                                        FieldRef<double> D) {
                                             
    int j = threadIdx.x + blockIdx.x*blockDim.x ;
    int i = threadIdx.y + blockIdx.y*blockDim.y ;

    if (i < g.NR + 2*g.Nghost && j < g.Nphi + 2*g.Nghost) {

        double krho = rho(i,j)*kappa_R(i,j) ;               
        D(i,j) = diffusion_coeff(g, J, krho, i, j) ;
    }
}


// Scheme based on: http://dx.doi.org/10.1016/j.jcp.2012.06.042
__global__ void create_FLD_mono_system(GridRef g, double dt, double Cv, 
                                       FieldConstRef<double> rho, 
                                       FieldConstRef<double> T, FieldConstRef<double> J,
                                       FieldConstRef<double> D, 
                                       FieldConstRef<double> kappa_P, 
                                       FieldConstRef<double> heat, double T_ext,
                                       CSR_SpMatrixRef mat, DnVecRef rhs,
                                       int boundary) {

    int j = threadIdx.x + blockIdx.x*blockDim.x + g.Nghost ;
    int i = threadIdx.y + blockIdx.y*blockDim.y + g.Nghost ;
    
    int nx = g.NR  ;
    int ny = g.Nphi ;
    
    if (i < nx+g.Nghost && j < ny+g.Nghost) {
        
        double kappa = kappa_P(i,j) ;
        double T0 = T(i,j) ;
        double d = rho(i,j) ;
        double k0 = 16*sigma_SB*T0*T0*T0 ;
        double vol = g.volume(i,j) ;
        
        ////// Gas temperature equation
        //    (Cv0/k0) (e1 - e0) / dt = - rho kp [e1 - J1] + heat + 0.75 k0*T0
        int idx = 2*((i-g.Nghost)*ny + j-g.Nghost) ;
        int start = mat.csr_offset[idx] ;
        mat.col_index[start] = idx ;


        // Time dependent terms
        if (dt > 0) {
            mat.data[start] = vol*Cv*d/(k0*dt) ;
            rhs.data[idx]   = vol*Cv*d*T0/(4*dt) ;
        }
        else {
            mat.data[start] = 0 ;
            rhs.data[idx]   = 0 ;
        }

        // Radiation-matter coupling
        mat.data[start] += vol*d*kappa ;
        rhs.data[idx]   += vol*heat(i,j) ;

        start += 1 ;
        mat.col_index[start] = idx+1 ;
        mat.data[start] = -vol*d*kappa ;


        /////// Radiation density equation
        //    (J1-J0) / (c dt) = rho kp [e1 - J1] - 0.75 k0*T0 + grad[D grad(J1)]
        idx = 2*((i-g.Nghost)*ny + j-g.Nghost) + 1 ;
        rhs.data[idx]  = 0 ;

        // Compute diffusion matrix elements
        double Dij[3][3] ;
        compute_diffusion_matrix(g, D, Dij, boundary, i, j) ;


        //////////////////////////////////////
        // Handle the boundaries. For:
        //   - open boundaries:   add the external flux to the rhs
        //   - closed boundaries: done by compute_diffusion_matrix
        //
        // Note that there are no corner cases because of the above if statements.

        double J_ext = 4*sigma_SB*pow(T_ext,4) ;

        if (i == g.Nghost) {
            if (boundary & BoundaryFlags::open_R_inner)
                rhs.data[idx] -= J_ext * (Dij[0][0] + Dij[0][1]) ;
        }
        if (i == nx + g.Nghost - 1) {
            if (boundary & BoundaryFlags::open_R_outer)
                rhs.data[idx] -= J_ext * (Dij[2][1] + Dij[2][2]) ;
        }
        if (j == g.Nghost) {
            if (boundary & BoundaryFlags::open_Z_inner)
                rhs.data[idx] -= J_ext * (Dij[0][0] + Dij[1][0]) ;
        }
        if (j == ny + g.Nghost - 1) {
            if (boundary & BoundaryFlags::open_Z_outer)
                rhs.data[idx] -= J_ext * (Dij[1][2] + Dij[2][2]) ;
        }

        ////////////////////////////////////////
        // Write the results to the matrix array:
        start = mat.csr_offset[idx] ;

        if (i > g.Nghost) {
            if (j > g.Nghost) {
                mat.col_index[start] = idx - 2*(ny+1) ;
                mat.data[start] = Dij[0][0] ;
                start++ ;
            }
            mat.col_index[start] = idx - 2*ny ;
            mat.data[start] = Dij[0][1] ;
            start++ ;
            if (j < ny + g.Nghost - 1) {
                mat.col_index[start] = idx - 2*(ny-1) ;
                mat.data[start] = Dij[0][2] ;
                start++ ;
            }
        }
        if (j > g.Nghost) {
            mat.col_index[start] = idx - 2 ;
            mat.data[start] = Dij[1][0] ;
            start++ ;
        }

        // Add the time-dependent terms
        // Radiation-matter coupling
        kappa = kappa_P(i,j) ;

        mat.col_index[start] = idx - 1 ;
        mat.data[start] = -vol*d*kappa ;
        start++ ; 

        mat.col_index[start] = idx ;
        mat.data[start] = vol*d*kappa ;

        // Time dependent terms
        if (dt > 0) {
            mat.data[start] += vol/(c_light*dt) ;
            rhs.data[idx]  += J(i,j) * vol/(c_light*dt) ;
        }

        // Add the central matrix element:
        mat.data[start] += Dij[1][1] ;
        start++;

        if (j  < ny + g.Nghost - 1) {
            mat.col_index[start] = idx + 2 ;
            mat.data[start] = Dij[1][2] ;
            start++ ;
        }


        if (i < nx + g.Nghost - 1) {
            if (j > g.Nghost) {
                mat.col_index[start] = idx + 2*(ny-1) ;
                mat.data[start] = Dij[2][0] ;
                start++ ;
            }
            mat.col_index[start] = idx + 2*ny ;
            mat.data[start] = Dij[2][1] ;
            start++ ;
            if (j < ny + g.Nghost - 1) {
                mat.col_index[start] = idx + 2*(ny+1) ;
                mat.data[start] = Dij[2][2] ;
                start++ ;
            }
        }
    }   
}

__global__ void copy_initial_values(GridRef g, FieldConstRef<double> T, 
                                   FieldConstRef<double> J, DnVecRef x) {

    int j = threadIdx.x + blockIdx.x*blockDim.x + g.Nghost ;
    int i = threadIdx.y + blockIdx.y*blockDim.y + g.Nghost ;
    
    int nx = g.NR ;
    int ny = g.Nphi ;

    if (i-g.Nghost < nx && j-g.Nghost < ny) {
        int idx = 2*((i-g.Nghost)*ny + j-g.Nghost) ;
        double T0 = T(i,j) ;
        x.data[idx] = 4*sigma_SB*T0*T0*T0*T0 ;
        x.data[idx+1] = J(i,j) ;
    }

}


__global__ void copy_final_values(GridRef g, FieldRef<double> T, 
                                FieldRef<double> J, DnVecConstRef x) {

    int j = threadIdx.x + blockIdx.x*blockDim.x + g.Nghost ;
    int i = threadIdx.y + blockIdx.y*blockDim.y + g.Nghost ;

    int nx = g.NR ;
    int ny = g.Nphi ;

    if (i-g.Nghost < nx && j-g.Nghost < ny) {
        int idx = 2*((i-g.Nghost)*ny + j-g.Nghost) ;

        T(i,j) = pow(x.data[idx]/(4*sigma_SB), 0.25) ;
        J(i,j) = x.data[idx+1] ;
        
        // Reflecting
        if (j < 2*g.Nghost) {
            T(i,2*g.Nghost-1-j) = T(i,j) ;
            J(i,2*g.Nghost-1-j) = J(i,j) ;

            // Corners
            if (i == g.Nghost)
                for (int ib=0; ib < g.Nghost; ib++) {
                    T(ib,2*g.Nghost-1-j) = T(i,j) ;
                    J(ib,2*g.Nghost-1-j) = J(i,j) ;
                }
            if (i == nx + g.Nghost-1)
                for (int ib=0; ib < g.Nghost; ib++) {
                    T(nx + g.Nghost + ib,2*g.Nghost-1-j) = T(i,j) ;
                    J(nx + g.Nghost + ib,2*g.Nghost-1-j) = J(i,j) ;
                }
        }
        // Copy
        if (j == ny + g.Nghost-1) {
            for (int jb=0; jb < g.Nghost; jb++) {
                T(i,ny + g.Nghost + jb) = T(i,j) ;
                J(i,ny + g.Nghost + jb) = J(i,j) ;

                // Corners
                if (i == g.Nghost)
                    for (int ib=0; ib < g.Nghost; ib++) {
                        T(ib,ny + g.Nghost + jb) = T(i,j) ;
                        J(ib,ny + g.Nghost + jb) = J(i,j) ;
                    }
                if (i == nx + g.Nghost-1)
                    for (int ib=0; ib < g.Nghost; ib++) {
                        T(nx + g.Nghost + ib,ny + g.Nghost + jb) = T(i,j) ;
                        J(nx + g.Nghost + ib,ny + g.Nghost + jb) = J(i,j) ;
                    }
            }
        }
        if (i == g.Nghost)
            for (int ib=0; ib < g.Nghost; ib++) {
                T(ib,j) = T(i,j) ;
                J(ib,j) = J(i,j) ;
            }
        if (i == nx + g.Nghost-1)
            for (int ib=0; ib < g.Nghost; ib++) {
                T(nx + g.Nghost + ib,j) = T(i,j) ;
                J(nx + g.Nghost + ib,j) = J(i,j) ;
            }
    }

}

void FLD_Solver::operator()(const Grid& g, double dt, double Cv, 
                            const Field<double>& rho, 
                            const Field<double>& kappa_P, 
                            const Field<double>& kappa_R,
                            const Field<double>& heat,
                            Field<double>& T, Field<double>& J) {

    CodeTiming::BlockTimer timing_block = 
        timer->StartNewTimer("FLD_Solver::operator()") ;
    CodeTiming::BlockTimer timing_subblock = 
        timer->StartNewTimer("FLD_Solver::operator()::create_system") ;   

    dim3 threads(32,32,1) ;
    dim3 blocks((g.Nphi + 2*g.Nghost+31)/32,(g.NR + 2*g.Nghost+31)/32,1) ;                 

    CSR_SpMatrix FLD_mat = _create_FLD_mono_matrix(g) ;

    DnVec rhs(FLD_mat.rows) ;
    DnVec sol(FLD_mat.rows) ;
    copy_initial_values<<<blocks, threads>>>(g, T, J, sol) ; 

    Field<double> D = create_field<double>(g) ;
    compute_diffusion_coeff<<<blocks, threads>>>(g, J, rho, kappa_R, D) ;
    check_CUDA_errors("compute_diffusion_coeff") ;           

    dim3 threads2(16,16,1) ;
    dim3 blocks2((g.Nphi + 2*g.Nghost+15)/16,(g.NR + 2*g.Nghost+15)/16,1) ;

    create_FLD_mono_system<<<blocks2, threads2>>>(g, dt, Cv, rho, T, J, D, kappa_P, 
                                                  heat, _T_ext, FLD_mat, rhs, _boundary) ;
    check_CUDA_errors("create_FLD_mono_system") ;    
    
    timing_subblock.StartNewBlock("FLD_Solver::operator()::solve") ;     

    Jacobi_Precond jacobi(FLD_mat) ;
    jacobi.transform(FLD_mat, sol, rhs) ;

    /*
        std::ofstream m("FLD_mono_mat.mm") ;
        write_MM_format(FLD_mat, m) ;
        std::ofstream v("FLD_mono_rhs.mm") ;
        write_MM_format(rhs, v) ;
    */

    PCG_Solver pcg(std::unique_ptr<CheckConvergence>(new CheckTemperatureResidual(_tol, 1, _tol)), _max_iter) ;

    // Solve the linear system
    if (_ILU_order < 0) {
        NoPrecond precond ;
        //BlockJacobi_precond precond(FLD_mat) ;
        bool success = 
            pcg.solve_non_symmetric(FLD_mat, rhs, sol, precond) ;
        
        if (not success) {
            std::cout << "Non-preconditioned solve failed, falling back to ILU(0)." << std::endl;
            
            copy_initial_values<<<blocks, threads>>>(g, T, J, sol) ; 
            jacobi.transform_guess(sol) ;

            ILU_precond precond(FLD_mat, 0) ;
            pcg.solve_non_symmetric(FLD_mat, rhs, sol, precond) ;
        }
    }
    else {
        ILU_precond precond(FLD_mat, _ILU_order) ;
        pcg.solve_non_symmetric(FLD_mat, rhs, sol, precond) ;
    }
    //GMRES_Solver gmres(1e-12, 100, 50) ;
    //gmres(FLD_mat, rhs, sol) ;
   
    jacobi.invert(sol) ;

    copy_final_values<<<blocks, threads>>>(g, T, J, sol) ; 
    check_CUDA_errors("copy_final_values") ;           

    timing_subblock.EndTiming() ;
}