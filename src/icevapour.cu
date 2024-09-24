#include <iostream>
#include <algorithm>
#include <cassert>

#include "icevapour.h"
#include "dustdynamics.h"
#include "constants.h"

struct ChemRate {
    double rate;
    double jac;
} ;

/*  Implicit scheme

Solve system:

    drho_vap/dt = -sum(R_a,n) rho_vap + sum(R_d,n rho_ice,n)

    drho_ice,n/dt = R_a,n rho_vap - R_d,n rho_ice,n

*/

__host__ __device__
double nu_i(MoleculeRef mol, double N_s) {

    double nu_0 = std::sqrt(2*N_s*k_B / (m_H * M_PI*M_PI));

    return nu_0 * mol.T_bind/(mol.m_mol/m_H);
}

__host__ __device__
ChemRate R_d_jac(MoleculeRef mol, Field3DRef<double> ice_grain, double N_s, FieldConstRef<double>& T, Field3DRef<Ice>& ice, 
                    Field3DRef<Prims>& W, const RealType* a, const RealType* m, int i, int j, int k) {

    // Scaled 0th order rate 

    double R = nu_i(mol, N_s) * std::exp(-mol.T_bind/T(i,j));

    double mass_per_layer = 4.*M_PI * ice(i,j,k).a * ice(i,j,k).a * N_s * W(i,j,k).rho / m[k] * mol.m_mol;
    // double mass_per_layer = 4.*M_PI * a[k] * a[k] * N_s * W(i,j,k).rho / m[k] * mol.m_mol;
    double num_layers = ice_grain(i,j,k) / max(mass_per_layer,1e-100); 

    ChemRate Rd;
    Rd.rate  = R / (1+num_layers);
    Rd.jac = -R * num_layers / ((1+num_layers)*(1+num_layers));

    return Rd;
}

__host__ __device__
ChemRate R_ph_jac(MoleculeRef mol, Field3DRef<double> ice_grain, double N_s, Field3DConstRef<double> J, FieldRef<Prims> Wg, 
                    Field3DRef<Prims>& W, const RealType* m, Field3DRef<Ice>& ice, int Jbin_idx, int i, int j, int k) {

    double gamma_UV = 0.;
    
    for (int l=0; l<Jbin_idx; l++) {
        gamma_UV += max(J(i,j,l)/(4.*M_PI), 0.);
    }

    double eta_CR = 1.e-17, Y = 2.7e-3;
    double n_H = 2. * Wg(i,j).rho / (2.8*m_H);
    double sum_mfp = 0.;

    for (int l=0; l<ice.Nd; l++) {
        sum_mfp += M_PI * ice(i,j,l).a * ice(i,j,l).a * W(i,j,l).rho / m[l];   
    }

    double gamma_CR = 0.15*eta_CR*n_H / max(sum_mfp,1e-100);

    double mass_per_layer = 4.*M_PI * ice(i,j,k).a * ice(i,j,k).a * N_s * W(i,j,k).rho / m[k] * mol.m_mol;
    double num_layers = ice_grain(i,j,k) / max(mass_per_layer,1e-100); 

    ChemRate Rd;
    Rd.rate  = (gamma_UV+gamma_CR) * Y / (4.*N_s*(1+num_layers));
    Rd.jac = 0.;

    return Rd;
}

__host__ __device__
ChemRate R_a_jac(MoleculeRef mol, FieldConstRef<double> T, Field3DRef<Prims>& W, Field3DRef<Ice>& ice, const RealType* m, const RealType* a, int i, int j, int k) {
    
    double v_th = std::sqrt(8.*k_B*T(i,j)/(M_PI*mol.m_mol));

    double R = M_PI * ice(i,j,k).a * ice(i,j,k).a * v_th * W(i,j,k).rho / m[k];

    ChemRate Ra;
    Ra.rate = R;
    Ra.jac = 0.;

    return Ra;
}

__global__ void _update_sizegrid(GridRef g, Field3DRef<Ice> ice, Field3DRef<Prims> W, Field3DRef<double> rho_ice, const RealType* m, RealType rho_ms, RealType rho_mi) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.z * blockDim.z ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost; j+=jstride) {
            for (int k=kidx; k<W.Nd; k+=kstride) {
                double rho_1 = (rho_ice(i,j,k)/(W(i,j,k)[0] * rho_mi) + 1./rho_ms);
                ice(i,j,k).a = pow((3.*m[k]/(4.*M_PI)) * rho_1, 1./3.);
                ice(i,j,k).rho = (rho_ice(i,j,k) + W(i,j,k)[0]) / (W(i,j,k)[0] * rho_1);
            } 
        }
    }

}

__global__ void _update_sizegrid(GridRef g, Field3DRef<Ice> ice, Field3DRef<Quants> W, Field3DRef<Quants> rhoice, const RealType* m, RealType rho_ms, RealType rho_mi) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.z * blockDim.z ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost; j+=jstride) {
            for (int k=kidx; k<W.Nd; k+=kstride) {
                double rho_1 = (rhoice(i,j,k).rho/(W(i,j,k)[0] * rho_mi) + 1./rho_ms);
                ice(i,j,k).a = pow((3.*m[k]/(4.*M_PI)) * rho_1, 1./3.);
                ice(i,j,k).rho = (rhoice(i,j,k).rho + W(i,j,k)[0]) / (W(i,j,k)[0] * rho_1);
            } 
        }
    }

}


__global__ void _implicit_update(GridRef g, Field3DRef<Prims> W, FieldRef<Prims> Wg, FieldConstRef<double> T, Field3DConstRef<double> J, Field3DRef<Ice> ice, const RealType* a, 
                                    const RealType* m, double N_s, MoleculeRef mol, Field3DRef<double> rhos, int Jbin_idx, double dt) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost; j+=jstride) {

            double A = 0, B = 0;
            int ndust = mol.ice.Nd;

            for (int k=0; k < ndust; k++) {
                
                ChemRate R_a = R_a_jac(mol, T, W, ice, m,a,i,j,k);
                ChemRate R_d = R_d_jac(mol, rhos, N_s, T, ice, W, a, m, i,j,k);
                ChemRate R_phd = R_ph_jac(mol, rhos, N_s, J, Wg, W, m, ice, Jbin_idx, i,j,k);

                A += (R_d.rate+R_phd.rate) * dt * mol.ice(i,j,k) / (1. + (R_d.rate+R_phd.rate) * dt);
                B += R_a.rate * dt / (1. + (R_d.rate+R_phd.rate) * dt);
            }

            rhos(i,j,ndust) = (mol.vap(i,j) + A) / (1. + B);

            for (int k=0; k < ndust; k++) {

                ChemRate R_a = R_a_jac(mol, T, W, ice, m,a, i,j,k);
                ChemRate R_d = R_d_jac(mol, rhos, N_s, T, ice, W, a, m, i,j,k);
                ChemRate R_phd = R_ph_jac(mol, rhos, N_s, J, Wg, W, m, ice, Jbin_idx, i,j,k);

                rhos(i,j,k) = (mol.ice(i,j,k) + R_a.rate * dt * rhos(i,j,ndust))  / (1. + (R_d.rate+R_phd.rate) * dt);
            }

        }
    }

}

__global__ void copy_initial_values(GridRef g, Field3DRef<double> rhos, MoleculeRef mol, Field3DRef<Prims> w_nof, Field3DRef<Prims> w, FieldRef<Prims> wg, double floor) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    int n_grains = mol.ice.Nd;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost; j+=jstride) {
            
            rhos(i,j,n_grains) = mol.vap(i,j);
            for (int k=0; k<n_grains; k++) {
                rhos(i,j,k) = mol.ice(i,j,k); 
                w_nof(i,j,k).rho = max(w(i,j,k).rho - 1.1*floor*wg(i,j).rho, 0.);
            }
        }
    }

}
__global__ void _copy_rhos(GridRef g, Field3DRef<double> rhos, Field3DRef<double> rhos_0) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.z * blockDim.z ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost; j+=jstride) {
            for (int k=kidx; k<rhos.Nd+1; k+=kstride) {
            
                rhos_0(i,j,k) = rhos(i,j,k);
            }
        }
    }

}

__global__ void copy_final_values(GridRef g, Field3DRef<double> rhos, MoleculeRef mol, double floor, FieldRef<Prims> wg) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    int n_grains = mol.ice.Nd;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost; j+=jstride) {
            
            mol.vap(i,j) = rhos(i,j,n_grains)+floor*1e-100*wg(i,j).rho;

            for (int k=0; k<n_grains; k++) {
                mol.ice(i,j,k) = rhos(i,j,k)+floor*1e-100*wg(i,j).rho;
            }
        }
    }

}

__global__ void get_tol(Field3DRef<double> rhos, Field3DRef<double> rhos_0, GridRef g, int ngrains, double* err) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.z * blockDim.z ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost; j+=jstride) {
            for (int k=kidx; k<ngrains+1; k+=kstride) {
                atomicAdd(&err[0], abs(rhos(i,j,k) - rhos_0(i,j,k)) / (rhos_0(i,j,k) + 1e-100) / ((ngrains + 1) * g.NR * g.Nphi));
            }
        }
    }
}

void IceVapChem::imp_update(double dt) {

    dim3 threads(32,16,1) ;
    dim3 blocks((_g.NR + 2*_g.Nghost+31)/32,(_g.Nphi + 2*_g.Nghost+15)/16,1) ;
          
    dim3 threads2(16,16,4) ;
    dim3 blocks2((_g.NR + 2*_g.Nghost+15)/16,(_g.Nphi + 2*_g.Nghost+15)/16,(_W.Nd+1 + 3)/4) ;

    dim3 threads3(16,16,4) ;
    dim3 blocks3((_g.NR + 2*_g.Nghost+15)/16,(_g.Nphi + 2*_g.Nghost+15)/16, (_W.Nd + 3)/4);
          
    Field3D<double> rhos = Field3D<double>(_g.NR + 2*_g.Nghost, _g.Nphi + 2*_g.Nghost, _W.Nd+1);
    Field3D<double> rhos_0 = Field3D<double>(_g.NR + 2*_g.Nghost, _g.Nphi + 2*_g.Nghost, _W.Nd+1);
    Field3D<Prims> W_nofloor = Field3D<Prims>(_g.NR + 2*_g.Nghost, _g.Nphi + 2*_g.Nghost, _W.Nd);

    copy_initial_values<<<blocks, threads>>>(_g, rhos, _mol, W_nofloor, _W, _Wg, _floor) ; 

    int it = 0;
    CudaArray<double> err = make_CudaArray<double>(1);
    err[0]= 1;

    while (err[0] > 1e-3) {
        err[0] = 0;

        _copy_rhos<<<blocks2,threads2>>>(_g, rhos, rhos_0);

        _implicit_update<<<blocks,threads>>>(_g, W_nofloor, _Wg, _T, _J, _sizes.ice, _sizes.grain_sizes(), _sizes.grain_masses(), N_s, _mol, rhos, _Jbin_idx, dt);

        get_tol<<<blocks2,threads2>>>(rhos, rhos_0, _g, _W.Nd, err.get());

        it++;
        _update_sizegrid<<<blocks3,threads3>>>(_g, _sizes.ice, _W, rhos, _sizes.grain_masses(), _sizes.solid_density(), _sizes.ice_density());
        cudaDeviceSynchronize();
    }
    // std::cout << "Ice-vap iterations: " << it << "\n";

    copy_final_values<<<blocks,threads>>>(_g, rhos, _mol, _floor, _Wg);
}

void update_sizegrid(Grid& g, SizeGridIce& sizes, Field3D<Quants>& Qd, Field3D<Quants>& ice) {

    dim3 threads3(16,16,4) ;
    dim3 blocks3((g.NR + 2*g.Nghost+15)/16,(g.Nphi + 2*g.Nghost+15)/16, (Qd.Nd + 3)/4);

    _update_sizegrid<<<blocks3,threads3>>>(g, sizes.ice, Qd, ice, sizes.grain_masses(), sizes.solid_density(), sizes.ice_density());
}

void update_sizegrid(Grid& g, SizeGridIce& sizes, Field3D<Prims>& Qd, Field3D<double>& ice) {

    dim3 threads3(16,16,4) ;
    dim3 blocks3((g.NR + 2*g.Nghost+15)/16,(g.Nphi + 2*g.Nghost+15)/16, (Qd.Nd + 3)/4);

    _update_sizegrid<<<blocks3,threads3>>>(g, sizes.ice, Qd, ice, sizes.grain_masses(), sizes.solid_density(), sizes.ice_density());
}
