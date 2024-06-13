#include <iostream>
#include "cuda_runtime.h"

#include "grid.h"
#include "dustdynamics1D.h"
#include "constants.h"
#include "van_leer.h"

__global__ 
void _calc_eta(GridRef g, FieldConstRef<Prims1D> W_g, double GMstar, FieldConstRef<double> cs, double* eta) {
    
    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int istride = gridDim.x * blockDim.x ;

    int j = g.Nghost;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {

        double P1 = W_g(i+1,g.Nghost).Sig / sqrt(2.*M_PI) * cs(i+1,j) * sqrt(GMstar/(g.Rc(i+1)*g.Rc(i+1)*g.Rc(i+1)));
        double P0 = W_g(i-1,g.Nghost).Sig / sqrt(2.*M_PI) * cs(i-1,j) * sqrt(GMstar/(g.Rc(i-1)*g.Rc(i-1)*g.Rc(i-1)));

        double vk = sqrt(GMstar/g.Rc(i));
        eta[i] = - cs(i,j)*cs(i,j) / (vk*vk) * log(P1/P0) / log(g.Rc(i+1)/g.Rc(i-1));

    }
}

__global__
void _calc_dust_vel(GridRef g, Field3DRef<Prims1D> W_d, FieldConstRef<Prims1D> W_g, double* eta, FieldConstRef<double> cs, double GMstar, RealType rho_m, const RealType* a, double floor) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int kstride = gridDim.z * blockDim.z ;

    int j = g.Nghost;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {
        for (int k=kidx; k<W_d.Nd; k+=kstride) {

            if (W_d(i,j,k).Sig < 1.1*W_g(i,j).Sig*floor) {
                W_d(i,j,k).v_R = 0.;
            }
            else {
                double vk = sqrt(GMstar/g.Rc(i));

                double St = M_PI/2. * (rho_m*a[k])/W_g(i,g.Nghost).Sig;
                
                W_d(i,j,k).v_R = (W_g(i,j).v_R - St*eta[i]*vk)/(St*St+1);
            }
        }
    }

}


void calculate_dust_vel(Grid& g, Field3D<Prims1D>& W_d, const Field<Prims1D>& W_g,
                        CudaArray<double>& nu, FieldConstRef<double>& cs, Star& star, SizeGrid& sizes, double floor) {

    size_t threads = 64 ;
    size_t blocks = (g.NR + 2*g.Nghost+63)/64;

    dim3 threads2D(32,1,32) ;
    dim3 blocks2D((g.NR + 2*g.Nghost+31)/32,1,(W_d.Nd+31)/32) ;


    CudaArray<double> eta = make_CudaArray<double>(g.NR+2*g.Nghost);
    CudaArray<double> v_gas = make_CudaArray<double>(g.NR+2*g.Nghost);
    
    _calc_eta<<<blocks,threads>>>(g, W_g, star.GM, cs, eta.get());
    check_CUDA_errors("_calc_eta");
    cudaDeviceSynchronize();
    // for (int i=0; i<g.NR+2*g.Nghost; i++) {
    //     std::cout << eta[i] << "\n";
    // }
    // _calc_v_gas<<<blocks,threads>>>(g, W_g, nu.get(), v_gas.get(), star.GM, gasfloor);
    _calc_dust_vel<<<blocks2D, threads2D>>>(g, W_d, W_g, eta.get(), cs, star.GM, sizes.solid_density(), sizes.grain_sizes(), floor);
    check_CUDA_errors("_calc_dust_vel");

}

__device__
double vl(GridRef& g, Field3DConstRef<Prims1D>& Qty, int i, int k, int qidx) {
    return vl_R(g, Qty, i, g.Nghost, k, qidx) ;
}

__device__
double compute_diff_flux(GridRef& g, Field3DConstRef<Prims1D>& W_d, FieldConstRef<Prims1D> W_g, Field3DRef<double>& D, int i, int k, double gas_floor, int bound) {

    if ((i < g.Nghost+1 && !(bound & BoundaryFlags::set_ext_R_inner)) || (i > g.NR+g.Nghost-1 && !(bound & BoundaryFlags::set_ext_R_outer))) { return 0.; }
    else if (W_g(i,g.Nghost).Sig < 1.1*gas_floor) {
        return 0 ;
    }

    double dRc = g.dRc(i-1);
    double dRe = g.Re(i)-g.Rc(i-1);

    double D_e = D(i-1,g.Nghost,k) + dRe*(D(i,g.Nghost,k)-D(i-1,g.Nghost,k)) / dRc;
    double ddtg = (W_d(i,g.Nghost,k).Sig/W_g(i,g.Nghost).Sig - W_d(i-1,g.Nghost,k).Sig/W_g(i-1,g.Nghost).Sig) / dRc;

    double flux = - D_e * ddtg;

    return flux;
}

__device__
void construct_diff_fluxes(GridRef& g, double v_l, double v_r, double v_av, 
                  double sig_l, double sig_r, Field3DRef<double>& flux, double diff_flux, int i, int k) {

    if (v_l < 0 && v_r > 0) {
        flux(i,g.Nghost,k) = diff_flux; 
    }
    
    if (v_av > 0.) {  
        flux(i,g.Nghost,k) = sig_l*v_l + diff_flux;
    }

    if (v_av < 0.) {   
        flux(i,g.Nghost,k) = sig_r*v_r + diff_flux;
    }

    if (v_av == 0.) {
        flux(i,g.Nghost,k) = 0.5*(sig_l*v_l + sig_r*v_r + 2*diff_flux);
    }
}

__device__
void dust_diff_flux(GridRef& g, Field3DConstRef<Prims1D>& W_d, int i, int k, Field3DRef<double>& flux, double diff_flux) {

    int j = g.Nghost;

    double sig_l = W_d(i-1,j,k).Sig;
    double sig_r = W_d(i,j,k).Sig;

    double v_l = W_d(i-1,j,k).v_R;
    double v_r = W_d(i,j,k).v_R;
    double rhorat, v_av;

    rhorat = std::pow(sig_r, 0.5)/std::pow(sig_l, 0.5);
    v_av = (v_l + rhorat * v_r) / (1. + rhorat);

    // Construct fluxes depending on sign of interface velocities

    construct_diff_fluxes(g, v_l, v_r, v_av, sig_l, sig_r, flux, diff_flux, i, k);
}

__device__
void dust_diff_flux_vl(GridRef& g, Field3DConstRef<Prims1D>& W_d, int i, int k, Field3DRef<double> flux, double diff_flux) {

    int j = g.Nghost;

    double dR_l = g.Re(i)-g.Rc(i-1);
    double dR_r = g.Re(i)-g.Rc(i);

    double sig_l = W_d(i-1,j,k).Sig + vl(g,W_d,i-1,k,0)*dR_l;
    double sig_r = W_d(i,j,k).Sig + vl(g,W_d,i,k,0)*dR_r;

    double v_l = W_d(i-1,j,k).v_R + vl(g,W_d,i-1,k,1)*dR_l;
    double v_r = W_d(i,j,k).v_R + vl(g,W_d,i,k,1)*dR_r;

    double rhorat, v_av;

    rhorat = std::pow(sig_r, 0.5)/std::pow(sig_l, 0.5);
    v_av = (v_l + rhorat * v_r) / (1. + rhorat);

    // Construct fluxes depending on sign of interface velocities

    construct_diff_fluxes(g, v_l, v_r, v_av, sig_l, sig_r, flux, diff_flux, i, k);
}

__global__ void _calc_diff_flux(GridRef g, Field3DConstRef<Prims1D> W_d, FieldConstRef<Prims1D> W_g,
                         Field3DRef<double> flux, Field3DRef<double> D, double gas_floor, int bound) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int kstride = gridDim.z * blockDim.z ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost+1; i+=istride) {
        for (int k=kidx; k<W_d.Nd; k+=kstride) {

            double diff_flux = compute_diff_flux(g, W_d, W_g, D, i, k, gas_floor, bound);

            dust_diff_flux(g, W_d, i, k, flux, diff_flux);

        }
    }
}

__global__ void _calc_diff_flux_vl(GridRef g, Field3DConstRef<Prims1D> W_d, FieldConstRef<Prims1D> W_g, 
                                    Field3DRef<double> flux, Field3DRef<double> D, double gas_floor, int bound) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int kstride = gridDim.z * blockDim.z ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost+1; i+=istride) {
        for (int k=kidx; k<W_d.Nd; k+=kstride) {

            double diff_flux = compute_diff_flux(g, W_d, W_g, D, i, k, gas_floor, bound);

            dust_diff_flux_vl(g, W_d, i, k, flux, diff_flux);

        }
    }
}

__global__
void _set_bounds_d(GridRef g, Field3DRef<Prims1D> W_d, int bound, double floor) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int kstride = gridDim.z * blockDim.z ;

    int j = g.Nghost;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int k=kidx; k<W_d.Nd; k+=kstride) {

            if (i < g.Nghost) {
                if (bound & BoundaryFlags::open_R_inner) {  //outflow
                    if (W_d(g.Nghost,j,k).v_R < 0.) {
                        W_d(i,j,k) = W_d(g.Nghost,j,k) ;
                    }
                    else {
                        W_d(i,j,k) = W_d(2*g.Nghost-1-i,j,k) ;
                        W_d(i,j,k).v_R *= -1 ;
                    }
                }
                else if (bound & BoundaryFlags::set_ext_R_inner) {} //set externally (e.g. inflow)
                else {  //reflecting
                    W_d(i,j,k) = W_d(2*g.Nghost-1-i,j,k) ;
                    W_d(i,j,k).v_R *= -1 ;
                }
            }        

            if (i>=g.NR+g.Nghost) {
                if (bound & BoundaryFlags::open_R_outer) {
                    if (W_d(g.NR+g.Nghost-1,j,k).v_R > 0.) {
                        W_d(i,j,k) = W_d(g.NR+g.Nghost-1,j,k);
                    }
                    else {
                        W_d(i,j,k) = W_d(g.NR+g.Nghost-1,j,k);
                        W_d(i,j,k).v_R *= -1 ;
                    }
                }
                else if (bound & BoundaryFlags::set_ext_R_outer) {} //set externally (e.g. inflow)
                else {
                    W_d(i,j,k) = W_d(g.NR+g.Nghost-1,j,k);
                    W_d(i,j,k).v_R *= -1 ;
                }
            }    
        
        }
    }
}


__global__ void _set_boundary_flux(GridRef g, int bound, Field3DRef<double> flux) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int kstride = gridDim.z * blockDim.z ;

    int j = g.Nghost;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int k=kidx; k<flux.Nd; k+=kstride) {

            if (i <= g.Nghost) {
                if (bound & BoundaryFlags::open_R_inner) {  //outflow
                    if (flux(i,j,k) > 0) // prevent inflow
                        flux(i,j,k) = 0.;
                }
                else if (bound & BoundaryFlags::set_ext_R_inner) {} //set externally (e.g. inflow)
                else {  //reflecting
                    flux(i,j,k) = 0.;
                }
            }     

            if (i>=g.NR+g.Nghost) {
                if (bound & BoundaryFlags::open_R_outer) {
                    if (flux(i,j,k) < 0) // prevent inflow
                        flux(i,j,k) = 0.;
                }
                else if (bound & BoundaryFlags::set_ext_R_outer) {} //set externally (e.g. inflow)
                else {
                    flux(i,j,k) = 0.;
                }
            }    
        }
    }
}

__global__ void _update_mid_Sig(GridRef g, Field3DRef<Prims1D> W_d_mid, Field3DRef<Prims1D> W_d, FieldConstRef<Prims1D> W_g, double dt, Field3DRef<double> flux, double floor) {
    
    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int kstride = gridDim.z * blockDim.z ;

    int j = g.Nghost;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {
        for (int k=kidx; k<W_d.Nd; k+=kstride) {

            double dRf = g.Re(i) * flux(i,j,k) - g.Re(i+1) * flux(i+1,j,k);
            double dV = 0.5 * (g.Re(i+1)*g.Re(i+1) - g.Re(i)*g.Re(i));

            W_d_mid(i,j,k).Sig = W_d(i,j,k).Sig + (0.5*dt/dV)*dRf;

            if (W_d_mid(i,j,k).Sig < 1.1*floor*W_g(i,g.Nghost).Sig) { W_d(i,j,k).Sig = floor*W_g(i,g.Nghost).Sig; } 

        }
    }

}

__global__ void _update_Sig(GridRef g, Field3DRef<Prims1D> W_d, FieldConstRef<Prims1D> W_g, double dt, Field3DRef<double> flux, double floor) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int kstride = gridDim.z * blockDim.z ;

    int j = g.Nghost;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {
        for (int k=kidx; k<W_d.Nd; k+=kstride) {

            double dRf = g.Re(i) * flux(i,j,k) - g.Re(i+1) * flux(i+1,j,k);
            double dV = 0.5 * (g.Re(i+1)*g.Re(i+1) - g.Re(i)*g.Re(i));

            W_d(i,j,k).Sig = W_d(i,j,k).Sig + (dt/dV)*dRf;
            
            if (W_d(i,j,k).Sig < 1.1*floor*W_g(i,g.Nghost).Sig) { W_d(i,j,k).Sig = floor*W_g(i,g.Nghost).Sig; } 

        }
    }
}

__global__ void copy_boundaries(GridRef g, Field3DRef<Prims1D> W_d, Field3DRef<Prims1D> W_d_mid, int bound) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int kstride = gridDim.z * blockDim.z ;

    int j = g.Nghost;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int k=kidx; k<W_d.Nd; k+=kstride) {
            if (i < g.Nghost) {
                if (bound & BoundaryFlags::set_ext_R_inner) {
                    W_d_mid(i,j,k) = W_d(i,j,k);
                }
            }     

            if (i >= g.NR+g.Nghost) {
                if (bound & BoundaryFlags::set_ext_R_outer) {
                    W_d_mid(i,j,k) = W_d(i,j,k);
                }
            }    
        }
    }
}


void DustDyn1D::operator() (Grid& g, Field3D<Prims1D>& W_d, const Field<Prims1D>& W_g, double dt) {

    dim3 threads(32,1,16) ;
    dim3 blocks((g.NR + 2*g.Nghost+31)/32,1,(W_d.Nd+31)/32) ;

    // calculate advection-diffusion

    Field3D<Prims1D> W_d_mid = Field3D<Prims1D>(g.NR+2*g.Nghost,1+2*g.Nghost,W_d.Nd);

    if (_boundary & BoundaryFlags::set_ext_R_inner || _boundary & BoundaryFlags::set_ext_R_outer) {
        copy_boundaries<<<blocks,threads>>>(g, W_d, W_d_mid, _boundary);
    }

    Field3D<double> flux = Field3D<double>(g.NR+2*g.Nghost,1+2*g.Nghost,W_d.Nd);

    _set_bounds_d<<<blocks,threads>>>(g, W_d, _boundary, _floor);
    check_CUDA_errors("_set_bounds_d");

    // Calc donor cell flux

    _calc_diff_flux<<<blocks,threads>>>(g, W_d, W_g, flux, _D, _gas_floor, _boundary);
    check_CUDA_errors("_set_bounds_d");

    // Update quantities a half time step
    _set_boundary_flux<<<blocks,threads>>>(g, _boundary, flux);
    check_CUDA_errors("_set_boundary_flux");
    _update_mid_Sig<<<blocks,threads>>>(g, W_d_mid, W_d, W_g, dt, flux, _floor);
    check_CUDA_errors("_update_mid_Sig");
    cudaDeviceSynchronize();
    calculate_dust_vel(g, W_d_mid, W_g, _nu, _cs, _star, _sizes, _floor);

    _set_bounds_d<<<blocks,threads>>>(g, W_d_mid, _boundary, _floor);
    check_CUDA_errors("_set_bounds_d");

    // Compute fluxes with Van Leer

    _calc_diff_flux_vl<<<blocks,threads>>>(g, W_d_mid, W_g, flux, _D, _gas_floor, _boundary);
    check_CUDA_errors("_calc_diff_flux_vl");

    _set_boundary_flux<<<blocks,threads>>>(g, _boundary, flux);
    check_CUDA_errors("_set_boundary_flux");
    _update_Sig<<<blocks,threads>>>(g, W_d, W_g, dt, flux, _floor);
    check_CUDA_errors("_update_Sig");
    cudaDeviceSynchronize();
    calculate_dust_vel(g, W_d, W_g, _nu, _cs, _star, _sizes, _floor);
}

double DustDyn1D::get_CFL_limit(const Grid& g, const Field3D<Prims1D>& W_dust, const Field<Prims1D>& W_gas) {
    double CFL_min = 1e308;
    for (int i=g.Nghost; i<g.NR+g.Nghost; i++) {
        double CFL_k = 1e308;
        for (int k=0; k<W_dust.Nd; k++) {

            if (W_dust(i,g.Nghost,k).Sig < 10.*W_gas(i,g.Nghost).Sig*_floor) { continue; }

            double dtR = abs(g.dRe(i)/W_dust(i,g.Nghost,k).v_R);
            CFL_k = min(CFL_k, _CFL_adv*dtR);
            
            if (_D(i,g.Nghost,k) != 0) {
                dtR = abs(g.dRe(i)*g.dRe(i) * W_gas(i,g.Nghost).Sig / _D(i,g.Nghost,k));
                CFL_k = min(CFL_k, _CFL_diff*dtR);
            }
        }
        CFL_min = min(CFL_min, CFL_k);
    } 
    return CFL_min;
}
