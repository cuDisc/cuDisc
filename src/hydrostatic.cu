
#include <iostream>
#include <stdexcept>
#include <string>

#include "grid.h"
#include "hydrostatic.h"
#include "reductions.h"
#include "star.h"
#include "utils.h"
#include "timing.h"
#include "dustdynamics.h"
#include "icevapour.h"

__global__ void setup_hydrostatic_maxtrix_device(double GM, GridRef g, 
                                                 FieldConstRef<double> cs2, FieldRef<double> out) {

    int j = threadIdx.x + blockIdx.x*blockDim.x ;
    int i = threadIdx.y + blockIdx.y*blockDim.y ;

    double val ;
    if (i < g.NR + 2*g.Nghost && j <  g.Nphi + 2*g.Nghost) {
        if (j > g.Nghost) {
            if (i < g.NR + 2*g.Nghost && j < g.Nphi + 2*g.Nghost) {
                double Rm = sqrt(g.Rc(i)*g.Rc(i) + g.Zc(i,j-1)*g.Zc(i,j-1)) ;
                double Rp = sqrt(g.Rc(i)*g.Rc(i) + g.Zc(i,j)*g.Zc(i,j)) ;

                double av_cs = 0.5 * (1/cs2[cs2.index(i,j-1)] + 1/cs2[cs2.index(i,j)]) ;
                val = exp(GM * av_cs * (1/Rp - 1/Rm)) ;
            }
        } else {
            val = 1 ;
        }

        // Prevent negative densities
        out[out.index(i,j)] = max(val, 0.0) ;
    }
}

/* setup_hydrostatic_maxtrix
 * 
 * Setups up a matrix of the recurrance relation linking the density between 
 * cells. I.e. the solution of:
 *    (log P_+ - log P_0) / dz = - a_grav / cs^2
 * where 
 *    a_grav = (GM * z) / (R^2 + z^2)^(3/2).
 *  
 */
void setup_hydrostatic_matrix(const Star& star, const Grid& g, const Field<double>& cs2, 
                               Field<double>& rho) {
    dim3 threads(32, 32, 1) ;
    dim3 blocks((g.Nphi+2*g.Nghost+31)/32, (g.NR+2*g.Nghost+31)/32 ) ;

    setup_hydrostatic_maxtrix_device<<<blocks, threads>>>(star.GM, g, cs2, rho) ;
    check_CUDA_errors("setup_hydrostatic_maxtrix") ;
}

__global__ void convert_pressure_to_density_device(GridRef g, FieldConstRef<double> cs2, 
                                                   FieldRef<double> rho) {
                                            
    int j = threadIdx.x + blockIdx.x*blockDim.x ;
    int i = threadIdx.y + blockIdx.y*blockDim.y ;

    if (i < g.NR + 2*g.Nghost && j < g.Nphi + 2*g.Nghost) 
        rho[rho.index(i, j)] /= cs2[cs2.index(i, j)] ;
}


void convert_pressure_to_density(const Grid& g, const Field<double>& cs2,
                                 Field<double>& rho) {

    dim3 threads(32, 32, 1) ;
    dim3 blocks((g.Nphi+2*g.Nghost+31)/32, (g.NR+2*g.Nghost+31)/32 ) ;

    convert_pressure_to_density_device<<<blocks, threads>>>(g, cs2, rho) ;
    check_CUDA_errors("convert_pressure_to_density_device") ;
}


__global__ void normalize_density_device(GridRef g, FieldRef<double> rho, 
                                         const double* Sigma, const double *norm) {

    int j = threadIdx.x + blockIdx.x*blockDim.x ;
    int i = threadIdx.y + blockIdx.y*blockDim.y ;

    if (i < g.NR + 2*g.Nghost && j < g.Nphi + 2*g.Nghost) {
        double Area = 0.5*(g.Re(i+1)*g.Re(i+1) - g.Re(i)*g.Re(i)) ;  
        rho(i, j) *= 0.5 * Sigma[i] * Area / norm[i] ; 
        rho(i, j) = max(rho(i, j), 1e-200);
    }
    __syncthreads() ;

    // Finally set the density in the ghost cell.
    if (i < g.NR + 2*g.Nghost && j < g.Nghost)
        rho(i, j) = rho(i, 2*g.Nghost-j-1) ;
}

void normalize_density(const Grid& g, Field<double>& rho, 
                       const CudaArray<double>& Sigma, const CudaArray<double>& norm) {
  
    dim3 threads(32, 32, 1) ;
    dim3 blocks((g.Nphi+2*g.Nghost+31)/32, (g.NR+2*g.Nghost+31)/32 ) ;

    normalize_density_device<<<blocks, threads>>>(g, rho, Sigma.get(), norm.get()) ;
    check_CUDA_errors("normalize_density") ;
}


void compute_hydrostatic_equilibrium(const Star& star, const Grid& g, Field<double>& rho, 
                                     const Field<double>& cs2, const CudaArray<double>& Sigma) {

    if (g.Nghost > 64) {
        std::string msg = 
            "compute_hydrostatic_equilibrium only works for Nghost <= 16" ;
        throw std::invalid_argument(msg);
    }

    CodeTiming::BlockTimer timing_block = 
        timer->StartNewTimer("compute_hydrostatic_equilibrium") ;

    // Step 1: Setup the finite difference factors for hydrostatic equilibrium
    setup_hydrostatic_matrix(star, g, cs2, rho) ;
    
    // Step 2: Solve the relation using parallel scan
    Reduction::scan_Z_mul(g, rho) ;
    convert_pressure_to_density(g, cs2, rho) ;

    // Step 3 compute the normalizations:
    zero_midplane_boundary(g, rho) ;

    CudaArray<double> norm = make_CudaArray<double>(g.NR + 2*g.Nghost) ;
    Reduction::volume_integrate_Z(g, rho, norm) ;

    // Step 4: Multiply rho by normalization 
    normalize_density(g, rho, Sigma, norm) ;    
}

__global__ void _rho_from_wg(GridRef g, FieldRef<double> rho, FieldRef<Prims> w_g, double gasfloor) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) { 
            rho(i,j) = w_g(i,j).rho;
        }
    }

}
__global__ void _wg_from_rho(GridRef g, FieldRef<double> rho, FieldRef<Prims> w_g, double gasfloor) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) { 
            w_g(i,j).rho = max(rho(i,j),gasfloor);
        }
    }

}

__global__ void _wg_from_rho(GridRef g, FieldRef<double> rho, FieldRef<Prims> w_g, double gasfloor, MoleculeRef mol) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) { 
            mol.vap(i,j) *= (rho(i,j) + gasfloor) / w_g(i,j).rho;
            w_g(i,j).rho = rho(i,j) + gasfloor;
        }
    }

}

// __global__ void _check_dust(GridRef g, FieldRef<double> rho, FieldRef<Quants> w_g, Field3DRef<Quants> q_d) {

//     int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
//     int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
//     int istride = gridDim.x * blockDim.x ;
//     int jstride = gridDim.y * blockDim.y ;

//     for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
//         for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) { 
//             if (w_g(i,j).rho/rho(i,j) > 100. || w_g(i,j).rho/rho(i,j) < 0.01) {
//                 for (int k=0; k<q_d.Nd; k++) {
//                     q_d(i,j,k).rho = 1.e-40;
//                     q_d(i,j,k).mom_R = 0.;
//                     q_d(i,j,k).amom_phi = 0.;
//                     q_d(i,j,k).mom_Z = 0.;
//                 }
//             }
//         }
//     }

// }

void compute_hydrostatic_equilibrium(const Star& star, const Grid& g, Field<Prims>& w_g, 
                                     const Field<double>& cs2, const CudaArray<double>& Sigma, double gasfloor) {
    
    Field<double> rho = create_field<double>(g);
    
    dim3 threads(32,32,1);
    dim3 blocks((g.NR+2*g.Nghost+31)/32, (g.Nphi+2*g.Nghost+31)/32 );

    _rho_from_wg<<<blocks,threads>>>(g, rho, w_g, gasfloor);

    if (g.Nghost > 64) {
        std::string msg = 
            "compute_hydrostatic_equilibrium only works for Nghost <= 16" ;
        throw std::invalid_argument(msg);
    }

    CodeTiming::BlockTimer timing_block = 
        timer->StartNewTimer("compute_hydrostatic_equilibrium") ;

    // Step 1: Setup the finite difference factors for hydrostatic equilibrium
    setup_hydrostatic_matrix(star, g, cs2, rho) ;
    
    // Step 2: Solve the relation using parallel scan
    Reduction::scan_Z_mul(g, rho) ;
    convert_pressure_to_density(g, cs2, rho) ;

    // Step 3 compute the normalizations:
    zero_midplane_boundary(g, rho) ;

    CudaArray<double> norm = make_CudaArray<double>(g.NR + 2*g.Nghost) ;
    Reduction::volume_integrate_Z(g, rho, norm) ;

    // Step 4: Multiply rho by normalization 
    normalize_density(g, rho, Sigma, norm) ;

    _wg_from_rho<<<blocks,threads>>>(g, rho, w_g, gasfloor);  
}

void compute_hydrostatic_equilibrium(const Star& star, const Grid& g, Field<Prims>& w_g, 
                                     const Field<double>& cs2, const CudaArray<double>& Sigma, Field3D<Prims>& q_d, double gasfloor) {
    
    Field<double> rho = create_field<double>(g);
    
    dim3 threads(32,32,1);
    dim3 blocks((g.NR+2*g.Nghost+31)/32, (g.Nphi+2*g.Nghost+31)/32 );

    _rho_from_wg<<<blocks,threads>>>(g, rho, w_g, gasfloor);

    if (g.Nghost > 64) {
        std::string msg = 
            "compute_hydrostatic_equilibrium only works for Nghost <= 16" ;
        throw std::invalid_argument(msg);
    }

    CodeTiming::BlockTimer timing_block = 
        timer->StartNewTimer("compute_hydrostatic_equilibrium") ;

    // Step 1: Setup the finite difference factors for hydrostatic equilibrium
    setup_hydrostatic_matrix(star, g, cs2, rho) ;
    
    // Step 2: Solve the relation using parallel scan
    Reduction::scan_Z_mul(g, rho) ;
    convert_pressure_to_density(g, cs2, rho) ;

    // Step 3 compute the normalizations:
    zero_midplane_boundary(g, rho) ;

    CudaArray<double> norm = make_CudaArray<double>(g.NR + 2*g.Nghost) ;
    Reduction::volume_integrate_Z(g, rho, norm) ;

    // Step 4: Multiply rho by normalization 
    normalize_density(g, rho, Sigma, norm) ;

    // _check_dust<<<blocks,threads>>>(g, rho, w_g, q_d);

    _wg_from_rho<<<blocks,threads>>>(g, rho, w_g, gasfloor);  
}


__global__
void _calc_Sigvap_new(GridRef g, FieldRef<double> Sig0, FieldRef<double> Sig1, FieldRef<double> Sigmol0, FieldRef<double> Sigmol1) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost; j+=jstride) {
            int j_r = g.Nphi+2*g.Nghost-1-j;

            int j_interp = g.Nphi+g.Nghost-2;
            for (int jint=g.Nghost; jint<g.Nphi+g.Nghost; jint++) {
                if (Sig0(i,jint) > Sig1(i,j)) {
                    j_interp = jint-1;
                    break;
                }
            }

            if (j_interp == g.Nghost-1) {
                Sigmol1(i,j) = exp(log(Sigmol0(i,j_interp)) + (log(Sig1(i,j))-log(Sig0(i,j_interp))) * (log(Sigmol0(i,j_interp+1))-log(Sigmol0(i,j_interp))) / (log(Sig0(i,j_interp+1))-log(Sig0(i,j_interp))));
            }
            else if (j_interp == g.Nphi+g.Nghost-2) {
                Sigmol1(i,j) = exp(log(Sigmol0(i,j_interp+1)) + (log(Sig1(i,j))-log(Sig0(i,j_interp+1))) * (log(Sigmol0(i,j_interp+1))-log(Sigmol0(i,j_interp))) / (log(Sig0(i,j_interp+1))-log(Sig0(i,j_interp))));
            }
            else {
                Sigmol1(i,j) = exp(log(Sigmol0(i,j_interp)) + (log(Sig1(i,j))-log(Sig0(i,j_interp))) * (log(Sigmol0(i,j_interp+1))-log(Sigmol0(i,j_interp))) / (log(Sig0(i,j_interp+1))-log(Sig0(i,j_interp))));
            }
        }
    }
}

__global__
void _calc_Sigma(GridRef g, FieldRef<double> rho, FieldRef<double> Sig, double gasfloor) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost; j+=jstride) {

            Sig(i,g.Nphi+2*g.Nghost-1-j) = rho(i,j)*g.dZe(i,j);

        }
    }
}

__global__
void _calc_Sigma(GridRef g, FieldRef<Prims> rho, FieldRef<double> Sig, double gasfloor) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost; j+=jstride) {

            Sig(i,g.Nphi+2*g.Nghost-1-j) = rho(i,j).rho*g.dZe(i,j);

        }
    }
}

__global__
void _calc_rho_vap(GridRef g, FieldRef<Prims> w_g, FieldRef<double> Sig, FieldRef<double> vap, double gas_floor, double floor) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int istride = gridDim.x * blockDim.x ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {
        for (int j=g.Nghost; j<g.Nphi+g.Nghost; j++) {

            if (w_g(i,j).rho <= gas_floor) {
                vap(i,j) = floor*w_g(i,j).rho;
            }
            else {
                vap(i,j) = max((Sig(i,g.Nphi+2*g.Nghost-1-j)-Sig(i,g.Nphi+2*g.Nghost-1-(j+1)))/g.dZe(i,j),floor*w_g(i,j).rho);
            }

        }
    }
}

__global__
void _renormalise_vap(GridRef g, FieldRef<double> Sig, FieldRef<double> vap) {
    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int istride = gridDim.x * blockDim.x ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {
        double Sig_temp=0;
        for (int j=g.Nghost; j<g.Nphi+g.Nghost; j++) {
            Sig_temp += vap(i,j)*g.dZe(i,j);
        }
        for (int j=g.Nghost; j<g.Nphi+g.Nghost; j++) {
            vap(i,j) *= Sig(i,g.Nphi+2*g.Nghost-1)/Sig_temp;
        }
    }
}

void compute_hydrostatic_equilibrium(const Star& star, const Grid& g, Field<Prims>& w_g, 
                                     const Field<double>& cs2, const CudaArray<double>& Sigma, Molecule& mol, double gasfloor, double floor) {
    
    Field<double> rho = create_field<double>(g);
    
    dim3 threads(32,32,1);
    dim3 blocks((g.NR+2*g.Nghost+31)/32, (g.Nphi+2*g.Nghost+31)/32 );

    _rho_from_wg<<<blocks,threads>>>(g, rho, w_g, gasfloor);

    Field<double> SigmaZ0 = create_field<double>(g);
    set_all(g, SigmaZ0, 0.);
    _calc_Sigma<<<blocks,threads>>>(g, rho, SigmaZ0, gasfloor);
    Reduction::scan_Z_sum(g, SigmaZ0);


    if (g.Nghost > 64) {
        std::string msg = 
            "compute_hydrostatic_equilibrium only works for Nghost <= 16" ;
        throw std::invalid_argument(msg);
    }

    CodeTiming::BlockTimer timing_block = 
        timer->StartNewTimer("compute_hydrostatic_equilibrium") ;

    // Step 1: Setup the finite difference factors for hydrostatic equilibrium
    setup_hydrostatic_matrix(star, g, cs2, rho) ;
    
    // Step 2: Solve the relation using parallel scan
    Reduction::scan_Z_mul(g, rho) ;
    convert_pressure_to_density(g, cs2, rho) ;

    // Step 3 compute the normalizations:
    zero_midplane_boundary(g, rho) ;

    CudaArray<double> norm = make_CudaArray<double>(g.NR + 2*g.Nghost) ;
    Reduction::volume_integrate_Z(g, rho, norm) ;

    // Step 4: Multiply rho by normalization 
    normalize_density(g, rho, Sigma, norm) ;
    _wg_from_rho<<<blocks,threads>>>(g, rho, w_g, gasfloor);  

    Field<double> SigmaZ = create_field<double>(g);
    set_all(g, SigmaZ, 0.);
    _calc_Sigma<<<blocks,threads>>>(g, w_g, SigmaZ, gasfloor);
    Reduction::scan_Z_sum(g, SigmaZ);

    Field<double> SigmaVapZ0 = create_field<double>(g);
    Field<double> SigmaVapZ1 = create_field<double>(g);
    set_all(g, SigmaVapZ0, 0.);
    set_all(g, SigmaVapZ1, 0.);
    _calc_Sigma<<<blocks,threads>>>(g, mol.vap, SigmaVapZ0, gasfloor);
    Reduction::scan_Z_sum(g, SigmaVapZ0);

    _calc_Sigvap_new<<<blocks,threads>>>(g, SigmaZ0, SigmaZ, SigmaVapZ0, SigmaVapZ1);

    _calc_rho_vap<<<1,1024>>>(g, w_g, SigmaVapZ1, mol.vap, gasfloor, floor);
    _renormalise_vap<<<1,1024>>>(g, SigmaVapZ0, mol.vap);
}