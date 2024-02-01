
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
#include "planck.h"
#include "utils.h"
#include "timing.h"


CSR_SpMatrix _create_FLD_multi_matrix(const Grid& g, int num_bands) {
    int nx = g.NR ;
    int ny = g.Nphi ;

    assert(nx > 1 && ny > 1) ;

    int non_zero = 
        + nx*ny*(1 + 3*num_bands) // Radiation-matter coupling
        + num_bands*(8*(nx-2)*(ny-2) + 5*2*(nx + ny - 4) + 3*4) ; // FLD

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

    CSR_SpMatrix mat((1+num_bands)*nx*ny, (1 + num_bands)*nx*ny, non_zero) ;

    mat.csr_offset[0] = 0 ;
    for(int i=0; i < nx; i++) 
        for (int j=0; j < ny; j++) {
            int k = (1+num_bands)*(i*ny + j) ;
            mat.csr_offset[k+1] = mat.csr_offset[k] + 1 + num_bands ;

            for (int b=0; b < num_bands; b++)
                mat.csr_offset[k+b+2] = mat.csr_offset[k+b+1] + 2 + ngb(i,j) ;    
        }
       
    assert (mat.csr_offset[(1+num_bands)*nx*ny] == mat.csr_offset[0] + non_zero) ;

    return mat ;
}


__global__ void compute_diffusion_coeff(GridRef g, Field3DConstRef<double> J,
                                        FieldConstRef<double> rho,  
                                        Field3DConstRef<double> kappa_ext,
                                        Field3DRef<double> D) {
                                             
    int j = threadIdx.x + blockIdx.x*blockDim.x ;
    int i = threadIdx.y + blockIdx.y*blockDim.y ;

    if (i < g.NR + 2*g.Nghost && j < g.Nphi + 2*g.Nghost) {
        double rho_ij = rho(i,j) ;
        for (int k=0; k < J.Nd; k++) {
            double krho = rho_ij*kappa_ext(i,j,k) ;          
            D(i,j,k) = diffusion_coeff(g, J, krho, i, j, k) ;
        }
    }
}

__global__ void compute_diffusion_coeff(GridRef g, Field3DConstRef<double> J,
                                        Field3DConstRef<double> rhokabs,  
                                        Field3DConstRef<double> rhoksca,
                                        Field3DRef<double> D) {
                                             
    int j = threadIdx.x + blockIdx.x*blockDim.x ;
    int i = threadIdx.y + blockIdx.y*blockDim.y ;

    if (i < g.NR + 2*g.Nghost && j < g.Nphi + 2*g.Nghost) {
        for (int k=0; k < J.Nd; k++) {
            double krho = rhokabs(i,j,k) + rhoksca(i,j,k) ;          
            D(i,j,k) = diffusion_coeff(g, J, krho, i, j, k) ;
        }
    }
}

__global__ void create_FLD_multi_system(GridRef g, double dt, double Cv, 
                                        FieldConstRef<double> rho, 
                                        FieldConstRef<double> T, Field3DConstRef<double> J,
                                        Field3DConstRef<double> D, 
                                        Field3DConstRef<double> opacity, 
                                        FieldConstRef<double> heat, 
                                        Field3DConstRef<double> scattering,
                                        double T_ext,
                                        PlanckIntegralRef planck, const double* wle,
                                        CSR_SpMatrixRef mat, DnVecRef rhs,
                                        int boundary) {

    int j = threadIdx.x + blockIdx.x*blockDim.x + g.Nghost ;
    int i = threadIdx.y + blockIdx.y*blockDim.y + g.Nghost ;
    
    int nx = g.NR ;
    int ny = g.Nphi ;
    int num_bands = J.Nd ;
    
    if (i < nx+g.Nghost && j < ny+g.Nghost) {
        
        double T0 = T[T.index(i,j)] ;
        double d = rho[rho.index(i,j)] ;
        double k0 = 16*sigma_SB*T0*T0*T0 ;
        double vol = g.volume(i,j) ;
        
        ////// Gas temperature equation
        //    (Cv0/k0) (e1 - e0) / dt = - rho kp [e1 - J1] + heat + 0.75 k0*T0
        int idx = (1+num_bands)*((i-g.Nghost)*ny + j-g.Nghost) ;
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
        rhs.data[idx]   += vol*heat(i,j) ;
        for (int b=0; b < num_bands; b++) {
            double kappa = opacity(i,j,b) ;
            
            mat.data[start] += vol*d*kappa *
                planck_factor(planck, T0, b, num_bands, wle) ;
            
            mat.col_index[start+b+1] = idx+b+1 ;
            mat.data[start+b+1] = -vol*d*kappa ;
        }


        /////// Radiation density equation
        //    (J1-J0) / (c dt) = rho kp [e1 - J1] - 0.75 k0*T0 + grad[D grad(J1)]
        for (int b=0; b < num_bands; b++) {

        idx = ((i-g.Nghost)*ny + j-g.Nghost) * (1 + num_bands) + 1 + b ;
        rhs.data[idx]  = 0 ;


        // Compute diffusion matrix elements
        double Dij[3][3] ;
        compute_diffusion_matrix(g, D, Dij, boundary, i, j, b) ;


        //////////////////////////////////////
        // Handle the boundaries. For:
        //   - open boundaries:   add the external flux to the rhs
        //   - closed boundaries: subtract the boundary terms from the matrix
        //
        // Note that there are no corner cases because of the above if statements.

        double J_ext ;
        if (T_ext > 0)
            J_ext = 4*sigma_SB*pow(T_ext,4) *
                planck_factor(planck, T_ext, b, num_bands, wle) ;
        else
            J_ext = 0 ;

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
                mat.col_index[start] = idx - (1+num_bands)*(ny+1) ;
                mat.data[start] = Dij[0][0] ;
                start++ ;
            }
            mat.col_index[start] = idx - (1+num_bands)*ny ;
            mat.data[start] = Dij[0][1] ;
            start++ ;
            if (j < ny + g.Nghost- 1) {
                mat.col_index[start] = idx - (1+num_bands)*(ny-1) ;
                mat.data[start] = Dij[0][2] ;
                start++ ;
            }
        }
        if (j > g.Nghost) {
            mat.col_index[start] = idx - (1+num_bands) ;
            mat.data[start] = Dij[1][0] ;
            start++ ;
        }

        // Add the time-dependent terms
        // Radiation-matter coupling
        double kappa = opacity(i,j,b) ;

        mat.col_index[start] = idx - (b + 1) ;
        mat.data[start] = -vol*d*kappa *
            planck_factor(planck, T0, b, num_bands, wle) ;
        start++ ; 

        mat.col_index[start] = idx ;
        mat.data[start] = vol*d*kappa ;

        // Diffusive flux production due to scattering of stellar
        // radiation:
        rhs.data[idx] += vol * scattering(i,j,b) ;

        // Time dependent terms
        if (dt > 0) {
            mat.data[start] += vol/(c_light*dt) ;
            rhs.data[idx]  += J(i,j,b) * vol/(c_light*dt) ;
        }

        // Add the central matrix element:
        mat.data[start] += Dij[1][1] ;
        start++;

        if (j < ny + g.Nghost - 1) {
            mat.col_index[start] = idx + (1+num_bands) ;
            mat.data[start] = Dij[1][2] ;
            start++ ;
        }


        if (i < nx + g.Nghost - 1) {
            if (j > g.Nghost) {
                mat.col_index[start] = idx + (1+num_bands)*(ny-1) ;
                mat.data[start] = Dij[2][0] ;
                start++ ;
            }
            mat.col_index[start] = idx + (1+num_bands)*ny ;
            mat.data[start] = Dij[2][1] ;
            start++ ;
            if (j < ny + g.Nghost - 1) {
                mat.col_index[start] = idx + (1+num_bands)*(ny+1) ;
                mat.data[start] = Dij[2][2] ;
                start++ ;
            }
        }
    }} // End loop over bands
}

__global__ void create_FLD_multi_system(GridRef g, double dt, double Cv, 
                                        FieldConstRef<double> rho, Field3DConstRef<double> rhok_abs, 
                                        FieldConstRef<double> T, Field3DConstRef<double> J,
                                        Field3DConstRef<double> D, 
                                        FieldConstRef<double> heat, 
                                        Field3DConstRef<double> scattering,
                                        double T_ext,
                                        PlanckIntegralRef planck, const double* wle,
                                        CSR_SpMatrixRef mat, DnVecRef rhs,
                                        int boundary) {

    int j = threadIdx.x + blockIdx.x*blockDim.x + g.Nghost ;
    int i = threadIdx.y + blockIdx.y*blockDim.y + g.Nghost ;
    
    int nx = g.NR ;
    int ny = g.Nphi ;
    int num_bands = J.Nd ;
    
    if (i < nx+g.Nghost && j < ny+g.Nghost) {
        
        double T0 = T[T.index(i,j)] ;
        double d = rho[rho.index(i,j)] ;
        double k0 = 16*sigma_SB*T0*T0*T0 ;
        double vol = g.volume(i,j) ;
        
        ////// Gas temperature equation
        //    (Cv0/k0) (e1 - e0) / dt = - rho kp [e1 - J1] + heat + 0.75 k0*T0
        int idx = (1+num_bands)*((i-g.Nghost)*ny + j-g.Nghost) ;
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
        rhs.data[idx]   += vol*heat(i,j) ;
        for (int b=0; b < num_bands; b++) {
            double rhokappa = rhok_abs(i,j,b) ;
            
            mat.data[start] += vol*rhokappa *
                planck_factor(planck, T0, b, num_bands, wle) ;
            
            mat.col_index[start+b+1] = idx+b+1 ;
            mat.data[start+b+1] = -vol*rhokappa ;
        }


        /////// Radiation density equation
        //    (J1-J0) / (c dt) = rho kp [e1 - J1] - 0.75 k0*T0 + grad[D grad(J1)]
        for (int b=0; b < num_bands; b++) {

        idx = ((i-g.Nghost)*ny + j-g.Nghost) * (1 + num_bands) + 1 + b ;
        rhs.data[idx]  = 0 ;


        // Compute diffusion matrix elements
        double Dij[3][3] ;
        compute_diffusion_matrix(g, D, Dij, boundary, i, j, b) ;


        //////////////////////////////////////
        // Handle the boundaries. For:
        //   - open boundaries:   add the external flux to the rhs
        //   - closed boundaries: subtract the boundary terms from the matrix
        //
        // Note that there are no corner cases because of the above if statements.

        double J_ext ;
        if (T_ext > 0)
            J_ext = 4*sigma_SB*pow(T_ext,4) *
                planck_factor(planck, T_ext, b, num_bands, wle) ;
        else
            J_ext = 0 ;

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
                mat.col_index[start] = idx - (1+num_bands)*(ny+1) ;
                mat.data[start] = Dij[0][0] ;
                start++ ;
            }
            mat.col_index[start] = idx - (1+num_bands)*ny ;
            mat.data[start] = Dij[0][1] ;
            start++ ;
            if (j < ny + g.Nghost- 1) {
                mat.col_index[start] = idx - (1+num_bands)*(ny-1) ;
                mat.data[start] = Dij[0][2] ;
                start++ ;
            }
        }
        if (j > g.Nghost) {
            mat.col_index[start] = idx - (1+num_bands) ;
            mat.data[start] = Dij[1][0] ;
            start++ ;
        }

        // Add the time-dependent terms
        // Radiation-matter coupling
        double rhokappa = rhok_abs(i,j,b) ;

        mat.col_index[start] = idx - (b + 1) ;
        mat.data[start] = -vol*rhokappa *
            planck_factor(planck, T0, b, num_bands, wle) ;
        start++ ; 

        mat.col_index[start] = idx ;
        mat.data[start] = vol*rhokappa ;

        // Diffusive flux production due to scattering of stellar
        // radiation:
        rhs.data[idx] += vol * scattering(i,j,b) ;

        // Time dependent terms
        if (dt > 0) {
            mat.data[start] += vol/(c_light*dt) ;
            rhs.data[idx]  += J(i,j,b) * vol/(c_light*dt) ;
        }

        // Add the central matrix element:
        mat.data[start] += Dij[1][1] ;
        start++;

        if (j < ny + g.Nghost - 1) {
            mat.col_index[start] = idx + (1+num_bands) ;
            mat.data[start] = Dij[1][2] ;
            start++ ;
        }


        if (i < nx + g.Nghost - 1) {
            if (j > g.Nghost) {
                mat.col_index[start] = idx + (1+num_bands)*(ny-1) ;
                mat.data[start] = Dij[2][0] ;
                start++ ;
            }
            mat.col_index[start] = idx + (1+num_bands)*ny ;
            mat.data[start] = Dij[2][1] ;
            start++ ;
            if (j < ny + g.Nghost - 1) {
                mat.col_index[start] = idx + (1+num_bands)*(ny+1) ;
                mat.data[start] = Dij[2][2] ;
                start++ ;
            }
        }
    }} // End loop over bands
}

__global__ void copy_initial_values(GridRef g, FieldConstRef<double> T, 
                                    Field3DConstRef<double> J, DnVecRef x) {

    int j = threadIdx.x + blockIdx.x*blockDim.x + g.Nghost ;
    int i = threadIdx.y + blockIdx.y*blockDim.y + g.Nghost ;

    int nx = g.NR ;
    int ny = g.Nphi ;
    int num_bands = J.Nd ;

    if (i < nx + g.Nghost  && j < ny + g.Nghost) {
        int idx = ((i-g.Nghost)*ny + j-g.Nghost)*(1+num_bands) ;
        double T0 = T(i,j) ;
        x.data[idx] = 4*sigma_SB*T0*T0*T0*T0 ;
        for(int b=0; b < num_bands; b++)
            x.data[idx+1+b] = J(i,j, b) ;
    }

}

__global__ void copy_final_values(GridRef g, FieldRef<double> T, 
                                  Field3DRef<double> J, DnVecConstRef x) {

    int j = threadIdx.x + blockIdx.x*blockDim.x + g.Nghost ;
    int i = threadIdx.y + blockIdx.y*blockDim.y + g.Nghost ;

    int nx = g.NR ;
    int ny = g.Nphi ;
    int num_bands = J.Nd ;


    if (i < nx + g.Nghost && j < ny + g.Nghost) {
        int idx = ((i-g.Nghost)*ny + j-g.Nghost)*(1 + num_bands) ;
        
        T(i,j) = pow(max(x.data[idx]/(4*sigma_SB), 1.), 0.25) ;
        for(int b=0; b < num_bands; b++)
            J(i,j,b) = x.data[idx+1+b] ;

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
            if (i == nx + g.Nghost-1)
                for (int ib=0; ib < g.Nghost; ib++) {
                    T(nx + g.Nghost + ib,2*g.Nghost-1-j) = T(i,j) ;
                    for(int b=0; b < num_bands; b++)
                        J(nx + g.Nghost + ib,2*g.Nghost-1-j, b) = J(i,j,b) ;
                }
        }
        // Copy
        if (j == ny + g.Nghost-1) {
            for (int jb=0; jb < g.Nghost; jb++) {
                T(i,ny + g.Nghost + jb) = T(i,j) ;
                for(int b=0; b < num_bands; b++)
                    J(i,ny + g.Nghost + jb,b) = J(i,j,b) ;

                // Corners
                if (i == g.Nghost)
                    for (int ib=0; ib < g.Nghost; ib++) {
                        T(ib,ny + g.Nghost + jb) = T(i,j) ;
                        for(int b=0; b < num_bands; b++)
                            J(ib,ny + g.Nghost + jb, b) = J(i,j,b) ;
                    }
                if (i == nx + g.Nghost-1)
                    for (int ib=0; ib < g.Nghost; ib++) {
                        T(nx + g.Nghost + ib,ny + g.Nghost + jb) = T(i,j) ;
                        for(int b=0; b < num_bands; b++)
                            J(nx + g.Nghost + ib,ny + g.Nghost + jb, b) = J(i,j,b) ;
                    }
            }
        }
        if (i == g.Nghost)
            for (int ib=0; ib < g.Nghost; ib++) {
                T(ib,j) = T(i,j) ;
                for(int b=0; b < num_bands; b++)
                    J(ib,j, b) = J(i,j, b) ;
            }
        if (i == nx + g.Nghost-1)
            for (int ib=0; ib < g.Nghost; ib++) {
                T(nx + g.Nghost + ib,j) = T(i,j) ;
                for(int b=0; b < num_bands; b++)
                    J(nx + g.Nghost + ib,j, b) = J(i,j, b) ;
            }
    }

}




// Scheme based on: http://dx.doi.org/10.1016/j.jcp.2012.06.042
void FLD_Solver::solve_multi_band(const Grid& g, double dt, double Cv, 
                                  const Field<double>& rho, 
                                  const Field3D<double>& opacity, 
                                  const Field<double>& heat, 
                                  const CudaArray<double>& wle,
                                  Field<double>& T, Field3D<double>& J) {


    Field3D<double> scattering = create_field3D<double>(g, J.Nd) ;
    set_all(g, scattering, 0) ;

    solve_multi_band(
        g, dt, Cv, rho, opacity, opacity, heat, scattering, wle, T, J) ;
}



void FLD_Solver::solve_multi_band(const Grid& g, double dt, double Cv, 
                                 const Field<double>& rho, 
                                 const Field3D<double>&kappa_abs,
                                 const Field3D<double>&kappa_ext,
                                 const Field<double>& heat, 
                                 const Field3D<double>& scattering,
                                 const CudaArray<double>& wle,
                                 Field<double>& T, Field3D<double>& J) {

    CodeTiming::BlockTimer timing_block = 
        timer->StartNewTimer("FLD_Solver::solve_multi_band") ;
    CodeTiming::BlockTimer timing_subblock = 
        timer->StartNewTimer("FLD_Solver::solve_multi_band::create_system") ;   

    dim3 threads(32,32,1) ;
    dim3 blocks((g.Nphi + 2*g.Nghost+31)/32,(g.NR + 2*g.Nghost+31)/32,1) ;
          
    CSR_SpMatrix FLD_mat = _create_FLD_multi_matrix(g, J.Nd) ;

    DnVec rhs(FLD_mat.rows) ;
    DnVec sol(FLD_mat.rows) ;
    copy_initial_values<<<blocks, threads>>>(g, T, J, sol) ; 

    Field3D<double> D = create_field3D<double>(g, J.Nd) ;
    compute_diffusion_coeff<<<blocks, threads>>>(g, J, rho, kappa_ext, D) ;
    //compute_diffusion_coeff<<<blocks, threads>>>(g, J, rho, kappa_abs, kappa_ext, D) ;
    check_CUDA_errors("compute_diffusion_coeff") ;           

    dim3 threads2(16,16,1) ;
    dim3 blocks2((g.Nphi + 2*g.Nghost+15)/16,(g.NR + 2*g.Nghost+15)/16,1) ;

    PlanckInegral planck ;
    create_FLD_multi_system<<<blocks2, threads2>>>(g, dt, Cv, rho, T, J, D, 
                                                   kappa_abs, 
                                                   heat, scattering,
                                                   _T_ext, planck, wle.get(),  
                                                   FLD_mat, rhs, _boundary) ;
    check_CUDA_errors("create_FLD_multi_system") ;    

    timing_subblock.StartNewBlock("FLD_Solver::solve_multi_band::solve") ;      

    Jacobi_Precond jacobi(FLD_mat) ;
    jacobi.transform(FLD_mat, sol, rhs) ;

    /*
    std::ofstream m("FLD_multi_mat.mm") ;
    write_MM_format(FLD_mat, m) ;
    std::ofstream v("FLD_multi_rhs.mm") ;
    write_MM_format(rhs, v) ;
    */


    // Solve the linear system
    //BlockJacobi_precond pc(FLD_mat, 1+J.Nd) ;
    PCG_Solver pcg(std::unique_ptr<CheckConvergence>(new CheckTemperatureResidual(_tol, J.Nd, _tol)), _max_iter) ;

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

    jacobi.invert(sol) ;

    copy_final_values<<<blocks, threads>>>(g, T, J, sol) ; 
    check_CUDA_errors("copy_final_values") ;     

    timing_subblock.EndTiming() ;
}

void FLD_Solver::solve_multi_band(const Grid& g, double dt, double Cv, 
                                 const Field3D<double>& rhokappa_abs,
                                 const Field3D<double>& rhokappa_sca,
                                 const Field<double>& rho, 
                                 const Field<double>& heat, 
                                 const Field3D<double>& scattering,
                                 const CudaArray<double>& wle,
                                 Field<double>& T, Field3D<double>& J) {

    CodeTiming::BlockTimer timing_block = 
        timer->StartNewTimer("FLD_Solver::solve_multi_band") ;
    CodeTiming::BlockTimer timing_subblock = 
        timer->StartNewTimer("FLD_Solver::solve_multi_band::create_system") ;   

    dim3 threads(32,32,1) ;
    dim3 blocks((g.Nphi + 2*g.Nghost+31)/32,(g.NR + 2*g.Nghost+31)/32,1) ;
          
    CSR_SpMatrix FLD_mat = _create_FLD_multi_matrix(g, J.Nd) ;

    DnVec rhs(FLD_mat.rows) ;
    DnVec sol(FLD_mat.rows) ;
    copy_initial_values<<<blocks, threads>>>(g, T, J, sol) ; 

    Field3D<double> D = create_field3D<double>(g, J.Nd) ;
    compute_diffusion_coeff<<<blocks, threads>>>(g, J, rhokappa_abs, rhokappa_sca, D) ;
    //compute_diffusion_coeff<<<blocks, threads>>>(g, J, rho, kappa_abs, kappa_ext, D) ;
    check_CUDA_errors("compute_diffusion_coeff") ;           

    dim3 threads2(16,16,1) ;
    dim3 blocks2((g.Nphi + 2*g.Nghost+15)/16,(g.NR + 2*g.Nghost+15)/16,1) ;

    PlanckInegral planck ;
    create_FLD_multi_system<<<blocks2, threads2>>>(g, dt, Cv, rho, rhokappa_abs, T, J, D, 
                                                   heat, scattering,
                                                   _T_ext, planck, wle.get(),  
                                                   FLD_mat, rhs, _boundary) ;
    check_CUDA_errors("create_FLD_multi_system") ;    

    timing_subblock.StartNewBlock("FLD_Solver::solve_multi_band::solve") ;      

    Jacobi_Precond jacobi(FLD_mat) ;
    jacobi.transform(FLD_mat, sol, rhs) ;

    /*
    std::ofstream m("FLD_multi_mat.mm") ;
    write_MM_format(FLD_mat, m) ;
    std::ofstream v("FLD_multi_rhs.mm") ;
    write_MM_format(rhs, v) ;
    */


    // Solve the linear system
    //BlockJacobi_precond pc(FLD_mat, 1+J.Nd) ;
    PCG_Solver pcg(std::unique_ptr<CheckConvergence>(new CheckTemperatureResidual(_tol, J.Nd, _tol)), _max_iter) ;

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

    jacobi.invert(sol) ;

    copy_final_values<<<blocks, threads>>>(g, T, J, sol) ; 
    check_CUDA_errors("copy_final_values") ;     

    timing_subblock.EndTiming() ;
}