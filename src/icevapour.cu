#include <iostream>
#include <algorithm>
#include <cassert>
#include <type_traits>

#include "icevapour.h"
#include "dustdynamics.h"
#include "constants.h"
#include "reductions.h"

struct ChemRate {
    double rate;
    double jac;
} ;

/*  Implicit scheme

Solve system:

    drho_vap/dt = -sum(R_a,n) rho_vap + sum(R_d,n rho_grains,n)

    drho_grains,n/dt = R_a,n rho_vap - R_d,n rho_grains,n

*/

__host__ __device__
double nu_i(MoleculeRef mol, double N_s) {

    double nu_0 = std::sqrt(2*N_s*k_B / (m_H * M_PI*M_PI));

    return nu_0 * std::sqrt(mol.T_bind/(mol.m_mol/m_H));
}

template<typename Te>
__host__ __device__
ChemRate R_d_jac(MoleculeRef mol, Field3DRef<double> ice_grain, double N_s, FieldConstRef<double>& T, Field3DRef<Grain>& grains, 
                    Field3DRef<Te>& W, const RealType* m, int i, int j, int k) {

    ChemRate Rd;
    
    // Scaled 0th order rate 
    if (W(i,j,k)[0] == 0.) {
        Rd.rate = 0.;
        Rd.jac = 0.;
    }
    else {
        double R = nu_i(mol, N_s) * std::exp(-mol.T_bind/T(i,j));

        double mass_per_layer = 4.*M_PI * grains(i,j,k).a * grains(i,j,k).a * N_s * W(i,j,k)[0] / m[k] * mol.m_mol;
        // double mass_per_layer = 4.*M_PI * a[k] * a[k] * N_s * W(i,j,k).rho / m[k] * mol.m_mol;
        double num_layers = ice_grain(i,j,k) / max(mass_per_layer,1e-100); 

        Rd.rate = R * (-expm1(-num_layers))/ max(num_layers,1e-100);//mass_per_layer * R / max(ice_grain(i,j,k),1e-100);// (1+num_layers);
        // if (i==50 && j==170 && k<10) {printf("%d %g %g\n", k, num_layers, (1.-std::exp(-num_layers))/ max(num_layers,1e-100));}
        Rd.jac = -R * num_layers / ((1+num_layers)*(1+num_layers));
    }

    return Rd;
}

__host__ __device__
ChemRate R_d_ph_jac(MoleculeRef mol, Field3DRef<double> ice_grain, double N_s, FieldConstRef<double>& T, Field3DRef<Grain>& grains, 
                    Field3DRef<Prims>& W, const RealType* m, Field3DConstRef<double> J, FieldRef<Prims> Wg, 
                    FieldRef<double>& F_UV, int Jbin_idx, double* lam_bins, double area_tot, int i, int j, int k) {

    ChemRate Rd;
    
    // Scaled 0th order rate 
    if (W(i,j,k)[0] == 0.) {
        Rd.rate = 0.;
        Rd.jac = 0.;
    }
    else {
        double R = nu_i(mol, N_s) * std::exp(-mol.T_bind/T(i,j));

        double temp = 4.*M_PI * grains(i,j,k).a * grains(i,j,k).a * N_s * W(i,j,k)[0] / m[k] * mol.m_mol;
        double num_layers = ice_grain(i,j,k) / max(temp,1e-100); 

        double gamma_UV = 0.;
        
        for (int l=0; l<Jbin_idx; l++) {
            double E_phot = 6.6260755e-27 * c_light/(lam_bins[l]/1.e4);
            gamma_UV += max(J(i,j,l)/(E_phot), 0.);
        }
        gamma_UV += F_UV(i,j);
    
        double eta_CR = 1.e-17, Y = 2.7e-3;
        double n_H = 2. * Wg(i,j)[0]/ (2.8*m_H);

        double gamma_CR = 0.15*eta_CR*n_H / max(area_tot,1e-100);
        temp = (gamma_UV + gamma_CR) * Y / (4. * N_s);

        Rd.rate = (R+temp) * -expm1(-num_layers) / max(num_layers, 1e-100);
        Rd.jac = 0.;//-R * num_layers / ((1+num_layers)*(1+num_layers));
    }

    return Rd;
}

__host__ __device__
ChemRate R_ph_jac(MoleculeRef mol, Field3DRef<double> ice_grain, double N_s, Field3DConstRef<double> J, FieldRef<Prims> Wg, 
                    Field3DRef<Prims>& W, const RealType* m, Field3DRef<Grain>& grains, int Jbin_idx, double* lam_bins, int i, int j, int k) {
    
    ChemRate Rd;

    if (W(i,j,k)[0] == 0.) {
        Rd.rate = 0.;
        Rd.jac = 0.;
    }
    else {
        double gamma_UV = 0.;
        
        for (int l=0; l<Jbin_idx; l++) {
            double nu = c_light/(lam_bins[l]/1.e4);
            double E_phot = 6.6260755e-27 * nu;
            gamma_UV += max(J(i,j,l)/(E_phot), 0.);
        }
    
        double eta_CR = 1.e-17, Y = 2.7e-3;
        double n_H = 2. * Wg(i,j).rho / (2.8*m_H);
        double sum_mfp = 0.;

        for (int l=0; l<grains.Nd; l++) {
            sum_mfp += M_PI * grains(i,j,l).a * grains(i,j,l).a * W(i,j,l).rho / m[l];   
        }

        double gamma_CR = 0.15*eta_CR*n_H / max(sum_mfp,1e-100);

        double mass_per_layer = 4.*M_PI * grains(i,j,k).a * grains(i,j,k).a * N_s * W(i,j,k).rho / m[k] * mol.m_mol;
        double num_layers = ice_grain(i,j,k) / max(mass_per_layer,1e-100); 

        
        Rd.rate  = (gamma_UV+gamma_CR) * Y * (-expm1(-num_layers))/ max(4.*N_s*num_layers,1e-100);
        Rd.jac = 0.;
    }

    return Rd;
}


__host__ __device__
ChemRate R_a_jac(MoleculeRef mol, FieldConstRef<double> T, Field3DRef<Prims>& W, Field3DRef<Grain>& grains, const RealType* m,int i, int j, int k) {
    
    double v_th = std::sqrt(8.*k_B*T(i,j)/(M_PI*mol.m_mol));

    double R = M_PI * grains(i,j,k).a * grains(i,j,k).a * v_th * W(i,j,k)[0] / m[k];
    
    ChemRate Ra;
    Ra.rate = R;
    Ra.jac = 0.;

    return Ra;
}

__host__ __device__
ChemRate R_a_jac(GridRef g, MoleculeRef mol, FieldConstRef<double> T, Field3DRef<Prims1D>& W, FieldRef<Prims1D>& W_g, Field3DRef<Grain>& grains, const RealType* m, double mu, double alpha, double GMstar, int i, int j, int k) {
    
    double v_th = std::sqrt(8.*k_B*T(i,j)/(M_PI*mol.m_mol));

    // double St = M_PI/2. * grains(i,j,k).a*grains(i,j,k).rho / W_g(i,j).Sig;
    double H = std::sqrt(k_B*T(i,j)*g.Rc(i)*g.Rc(i)*g.Rc(i)/(mu*m_H*GMstar));//*min(1.,sqrt(alpha/(min(St,0.5)*(1.+St*St))));
    double R = M_PI * grains(i,j,k).a * grains(i,j,k).a * v_th * W(i,j,k)[0] / m[k] / (std::sqrt(2.*M_PI)*H);

    ChemRate Ra;
    Ra.rate = R;
    Ra.jac = 0.;

    return Ra;
}

__global__ void _update_sizegrid(GridRef g, Field3DRef<Grain> grains, DensityView rho_dust, DensityView rho_ice, const RealType* m, RealType rho_ms, RealType rho_mi) {

    int iidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int istride = gridDim.z * blockDim.z ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.x * blockDim.x ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost; j+=jstride) {
            for (int k=kidx; k<rho_dust.Nd; k+=kstride) {
                double rho_d = max(rho_dust(i,j,k),1e-200), rho_i = max(rho_ice(i,j,k), 0.);
                // Ice without a refractory core (e.g. ice moved into a bin whose dust was
                // clipped to zero during coagulation) would give unbounded grain sizes, and
                // NaN coagulation rates; treat such bins as bare grains.
                if (rho_d == 1e-200) rho_i = 0;
                double rho_1 = (rho_i/(rho_d * rho_mi) + 1./rho_ms);
                grains(i,j,k).a = pow((3.*m[k]/(4.*M_PI)) * rho_1, 1./3.);
                grains(i,j,k).rho = (rho_i + rho_d) / (rho_d * rho_1);
            } 
        }
    }

}


/* The implicit ice-vapour update (2D) is split into per-species and per-cell
 * steps, so the per-species work runs in parallel with the species index
 * (contiguous in memory) on threadIdx.x. The sums over species are done with
 * Reduction::sum_species:
 *   1. area_tot = sum_k pi a_k^2 n_k
 *   2. adsorption / desorption rates R_a, R_d of each species, and
 *        A = sum_k R_d dt rho0_k / (1 + R_d dt),  B = sum_k R_a dt / (1 + R_d dt)
 *   3. vapour: rho_v = (rho0_v + A) / (1 + B)
 *   4. ice:    rho_k = (rho0_k + R_a dt rho_v) / (1 + R_d dt)
 */

// Grain surface area of each species, pi a^2 n
__global__ void _area_terms(GridRef g, Field3DRef<Prims> W, Field3DRef<Grain> grains, const RealType* m, Field3DRef<double> area) {

    int iidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int istride = gridDim.z * blockDim.z ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.x * blockDim.x ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride)
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost; j+=jstride)
            for (int k=kidx; k<area.Nd; k+=kstride)
                area(i,j,k) = M_PI * grains(i,j,k).a * grains(i,j,k).a * W(i,j,k)[0]/ m[k];
}

// Adsorption and desorption rates of each species, and their contributions to A and B
__global__ void _rates(GridRef g, Field3DRef<Prims> W, FieldRef<Prims> Wg, FieldConstRef<double> T, Field3DConstRef<double> J, Field3DRef<Grain> grains,
                       const RealType* m, double N_s, MoleculeRef mol, Field3DRef<double> rhos, Field3DRef<double> rhos_0, FieldRef<double> F_UV, int Jbin_idx, double* lam_bins,
                       FieldRef<double> area_tot, double dt, 
                       Field3DRef<double> R_ads, Field3DRef<double> R_des, Field3DRef<double> A_terms, Field3DRef<double> B_terms) {

    int iidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int istride = gridDim.z * blockDim.z ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.x * blockDim.x ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost; j+=jstride) {
            for (int k=kidx; k<mol.ice.Nd; k+=kstride) {

                ChemRate R_a = R_a_jac(mol, T, W, grains, m, i, j, k);
                ChemRate R_d = R_d_ph_jac(mol, rhos, N_s, T, grains, W, m, J, Wg, F_UV, Jbin_idx, lam_bins, area_tot(i,j), i,j,k);

                R_ads(i,j,k) = R_a.rate ;
                R_des(i,j,k) = R_d.rate ;
                A_terms(i,j,k) = (R_d.rate) * dt * rhos_0(i,j,k) / (1. + (R_d.rate) * dt);
                B_terms(i,j,k) = R_a.rate * dt / (1. + (R_d.rate) * dt);
            }
        }
    }
}

__global__ void _update_vapour(GridRef g, Field3DRef<double> rhos, Field3DRef<double> rhos_0, FieldRef<double> A, FieldRef<double> B, int ndust) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride)
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost; j+=jstride)
            rhos(i,j,ndust) = (rhos_0(i,j,ndust) + A(i,j)) / (1. + B(i,j));
}

__global__ void _update_ice(GridRef g, Field3DRef<double> rhos, Field3DRef<double> rhos_0, 
                            Field3DRef<double> R_ads, Field3DRef<double> R_des, int ndust, double dt) {

    int iidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int istride = gridDim.z * blockDim.z ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.x * blockDim.x ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride)
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost; j+=jstride)
            for (int k=kidx; k<ndust; k+=kstride)
                rhos(i,j,k) = (rhos_0(i,j,k) + R_ads(i,j,k) * dt * rhos(i,j,ndust))  / (1. + R_des(i,j,k) * dt);
}

__global__ void _implicit_update(GridRef g, Field3DRef<Prims1D> W, FieldRef<Prims1D> Wg, FieldConstRef<double> T, Field3DRef<Grain> grains,
                                    const RealType* m, double N_s, MoleculeRef mol, Field3DRef<double> rhos, Field3DRef<double> rhos_0, double mu, double alpha, double GMstar, double dt) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost; j+=jstride) {

            double A = 0., B = 0.;
            int ndust = mol.ice.Nd;

            for (int k=0; k < ndust; k++) {
                
                ChemRate R_a = R_a_jac(g, mol, T, W, Wg, grains, m, mu, alpha, GMstar, i,j,k);
                ChemRate R_d = R_d_jac(mol, rhos, N_s, T, grains, W, m, i,j,k);
                // ChemRate R_phd = R_ph_jac(mol, rhos, N_s, J, Wg, W, m, grains, Jbin_idx, lam_bins, i,j,k);
                A += (R_d.rate) * dt * rhos_0(i,j,k) / (1. + (R_d.rate) * dt);
                B += R_a.rate * dt / (1. + (R_d.rate) * dt);
            }

            rhos(i,j,ndust) = (rhos_0(i,j,ndust) + A) / (1. + B);

            for (int k=0; k < ndust; k++) {

                ChemRate R_a = R_a_jac(g, mol, T, W, Wg, grains, m, mu, alpha, GMstar, i,j,k);
                ChemRate R_d = R_d_jac(mol, rhos, N_s, T, grains, W, m, i,j,k);
                // ChemRate R_phd = R_ph_jac(mol, rhos, N_s, J, Wg, W, m, grains, Jbin_idx, lam_bins, i,j,k);

                rhos(i,j,k) = (rhos_0(i,j,k) + R_a.rate * dt * rhos(i,j,ndust))  / (1. + (R_d.rate) * dt);

            }

        }
    }

}

template<typename T>
__global__ void copy_initial_values(GridRef g, Field3DRef<double> rhos, MoleculeRef mol, Field3DRef<T> w_nof, Field3DRef<T> w, FieldRef<T> wg, double floor) {

    int iidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int istride = gridDim.z * blockDim.z ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.x * blockDim.x ;

    int n_grains = mol.ice.Nd;

    // k = n_grains is the vapour
    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost; j+=jstride) {
            for (int k=kidx; k<=n_grains; k+=kstride) {
                if (k == n_grains) {
                    rhos(i,j,n_grains) = max(mol.vap(i,j) - 1.1e-100*floor*wg(i,j)[0], 0.);
                    continue ;
                }
                w_nof(i,j,k)[0] = max(w(i,j,k)[0] - 1.1*floor*wg(i,j)[0], 0.);
                rhos(i,j,k) = max(mol.ice(i,j,k) - 1.1e-100*floor*wg(i,j)[0], 0.);
                if (w_nof(i,j,k)[0] == 0.) {
                    rhos(i,j,k) = 0.;
                }
            }
        }
    }

}
__global__ void _copy_rhos(GridRef g, Field3DRef<double> rhos, Field3DRef<double> rhos_0) {

    int iidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int istride = gridDim.z * blockDim.z ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.x * blockDim.x ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost; j+=jstride) {
            for (int k=kidx; k<rhos.Nd; k+=kstride) {
            
                rhos_0(i,j,k) = rhos(i,j,k);
            }
        }
    }

}

template<typename T>
__global__ void copy_final_values(GridRef g, Field3DRef<double> rhos, MoleculeRef mol, double floor, FieldRef<T> wg) {

    int iidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int istride = gridDim.z * blockDim.z ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.x * blockDim.x ;

    int n_grains = mol.ice.Nd;

    // k = n_grains is the vapour
    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost; j+=jstride) {
            for (int k=kidx; k<=n_grains; k+=kstride) {
                if (k == n_grains)
                    mol.vap(i,j) = rhos(i,j,n_grains)+floor*1e-100*wg(i,j)[0];
                else
                    mol.ice(i,j,k) = rhos(i,j,k)+floor*1e-100*wg(i,j)[0];
            }
        }
    }

}

namespace {

// Relative change of each species (k = ngrains is the vapour) in the interior cells
template<typename T>
__global__ void _relative_change_terms(GridRef g, Field3DConstRef<double> rhos, Field3DConstRef<double> rhos_0, 
                                       int ngrains, double floor, FieldRef<T> wg, Field3DRef<double> terms) {

    int iidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int istride = gridDim.z * blockDim.z ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.x * blockDim.x ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost; j+=jstride) {
            for (int k=kidx; k<=ngrains; k+=kstride) {
                double rho_floor = floor*wg(i,j)[0] ;
                double r0 = max(rhos_0(i,j,k), rho_floor) ;
                terms(i,j,k) = (max(rhos(i,j,k), rho_floor) - r0) / r0 ;
            }
        }
    }
}

/* _relative_change
 *
 * Mean relative change between rhos and rhos_0 over the interior cells and
 * all species, i.e. sum |max(rhos, floor) - max(rhos_0, floor)| / max(rhos_0, floor) / N.
 * The ghost cells and padding of the terms are zeroed, so the whole buffer is summed.
 */
template<typename T>
double _relative_change(GridRef g, Field3DRef<double> rhos, Field3DRef<double> rhos_0, int ngrains, double floor, FieldRef<T> wg) {

    Field3D<double> terms(g.NR + 2*g.Nghost, g.Nphi + 2*g.Nghost, ngrains+1) ;
    int size = (g.NR + 2*g.Nghost) * terms.stride_Zd ;
    (void) cudaMemset(terms.get().get(), 0, size*sizeof(double)) ;

    dim3 threads(32,4,2) ;
    dim3 blocks((ngrains+1 + 31)/32, (g.Nphi + 2*g.Nghost + 3)/4, (g.NR + 2*g.Nghost + 1)/2) ;
    _relative_change_terms<T><<<blocks, threads>>>(g, rhos, rhos_0, ngrains, floor, wg, terms) ;
    check_CUDA_errors("_relative_change_terms") ;

    double sum = Reduction::sum_abs(terms.get(), size) ;
    return sum / ((double)g.NR * g.Nphi * (ngrains+1)) ;
}

} // namespace

__global__ void _calc_drhovdt(GridRef g, FieldRef<double> drhovdt, Field3DRef<double> rhos, Field3DRef<double> rhos_0, double dt) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) {
            drhovdt(i,j) = (rhos(i,j,rhos.Nd-1)-rhos_0(i,j,rhos.Nd-1))/dt;
        }
    }
}

__global__ void _floor_above_phdiss(GridRef g, FieldRef<Prims> wg, MoleculeRef mol, double* h, double _floor) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ; 

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) {  
            if (g.Zc(i,j) > h[i] || h[i] == g.Zc(i,g.Nghost)) {
                mol.vap(i,j) = 1e-100*_floor*wg(i,j).rho ;
            }
        }
    }
}

__global__ void _compute_mu(GridRef g, FieldRef<Prims> wg, FieldRef<double> vap, FieldRef<double> mu, double mu_HHe, double mu_vap) {

    // Calculates mean molecular weight

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ; 

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) {  
            double f_v = vap(i,j) / (vap(i,j) + wg(i,j).rho);
            mu(i,j) = 1./(f_v/mu_vap + (1.-f_v)/mu_HHe);
        }
    }
}


void IceVapChem::imp_update(double dt, double& dt_chem) {

    dim3 threads(32,16,1) ;
    dim3 blocks((_g.NR + 2*_g.Nghost+31)/32,(_g.Nphi + 2*_g.Nghost+15)/16,1) ;
          
    // 3D kernels: grain size (contiguous in memory) on x
    dim3 threads2(32,4,2) ;
    dim3 blocks2((_W.Nd+1 + 31)/32,(_g.Nphi + 2*_g.Nghost+3)/4,(_g.NR + 2*_g.Nghost+1)/2) ;

    dim3 threads3(32,4,2) ;
    dim3 blocks3((_W.Nd + 31)/32,(_g.Nphi + 2*_g.Nghost+3)/4,(_g.NR + 2*_g.Nghost+1)/2) ;
          
    Field3D<double> rhos = Field3D<double>(_g.NR + 2*_g.Nghost, _g.Nphi + 2*_g.Nghost, _W.Nd+1);
    Field3D<double> rhos_1 = Field3D<double>(_g.NR + 2*_g.Nghost, _g.Nphi + 2*_g.Nghost, _W.Nd+1);
    Field3D<double> rhos_0 = Field3D<double>(_g.NR + 2*_g.Nghost, _g.Nphi + 2*_g.Nghost, _W.Nd+1);
    Field3D<Prims> W_nofloor = Field3D<Prims>(_g.NR + 2*_g.Nghost, _g.Nphi + 2*_g.Nghost, _W.Nd);

    _floor_above_phdiss<<<blocks,threads>>>(_g, _Wg, _mol, _h_phdiss.get(), _floor);

    // Temporaries for the implicit update
    Field3D<double> area = Field3D<double>(_g.NR + 2*_g.Nghost, _g.Nphi + 2*_g.Nghost, _W.Nd);
    Field3D<double> R_ads = Field3D<double>(_g.NR + 2*_g.Nghost, _g.Nphi + 2*_g.Nghost, _W.Nd);
    Field3D<double> R_des = Field3D<double>(_g.NR + 2*_g.Nghost, _g.Nphi + 2*_g.Nghost, _W.Nd);
    Field3D<double> A_terms = Field3D<double>(_g.NR + 2*_g.Nghost, _g.Nphi + 2*_g.Nghost, _W.Nd);
    Field3D<double> B_terms = Field3D<double>(_g.NR + 2*_g.Nghost, _g.Nphi + 2*_g.Nghost, _W.Nd);
    Field<double> area_tot = create_field<double>(_g);
    Field<double> A_sum = create_field<double>(_g);
    Field<double> B_sum = create_field<double>(_g);

    copy_initial_values<<<blocks2, threads2>>>(_g, rhos, _mol, Field3DRef<Prims>(W_nofloor), _W, _Wg, _floor) ; 
    _copy_rhos<<<blocks2,threads2>>>(_g, rhos, rhos_0);

    int it = 0;
    double err_tot= 1;

    while (err_tot > 1e-5) {

        _copy_rhos<<<blocks2,threads2>>>(_g, rhos, rhos_1);

        // Implicit update (see _area_terms): per-species rates and sums over species,
        // then the vapour and the ice
        _area_terms<<<blocks2,threads2>>>(_g, Field3DRef<Prims>(W_nofloor), _sizes.grain_props, _sizes.grain_masses(), area);
        Reduction::sum_species(_g, area, area_tot);

        _rates<<<blocks2,threads2>>>(_g, Field3DRef<Prims>(W_nofloor), _Wg, _T, _J, _sizes.grain_props, _sizes.grain_masses(), N_s, _mol, rhos, rhos_0, _F_UV, _Jbin_idx, _bins.bands.get(),
                                     area_tot, dt, R_ads, R_des, A_terms, B_terms);
        Reduction::sum_species(_g, A_terms, A_sum);
        Reduction::sum_species(_g, B_terms, B_sum);

        _update_vapour<<<blocks,threads>>>(_g, rhos, rhos_0, A_sum, B_sum, _W.Nd);
        _update_ice<<<blocks2,threads2>>>(_g, rhos, rhos_0, R_ads, R_des, _W.Nd, dt);

        err_tot = _relative_change(_g, rhos, rhos_1, _W.Nd, _floor, _Wg);
        
        it++;
        _update_sizegrid<<<blocks3,threads3>>>(_g, _sizes.grain_props, density_view(_W), density_view(rhos), _sizes.grain_masses(), _sizes.solid_density(), _sizes.ice_density());
    }

    dt_chem = dt * 0.01/_relative_change(_g, rhos, rhos_0, _W.Nd, _floor, _Wg);

    _calc_drhovdt<<<blocks,threads>>>(_g, _drhovdt, rhos, rhos_0, dt);

    copy_final_values<<<blocks2,threads2>>>(_g, rhos, _mol, _floor, _Wg);

    _compute_mu<<<blocks,threads>>>(_g, _Wg, _mol.vap, _mu, _mu_HHe, _mol.m_mol/m_H);
}

void IceVapChem1D::imp_update(double dt, double& dt_chem) {

    size_t threads = 512;
    size_t blocks = (_g.NR + 2*_g.Nghost+511)/512 ;
          
    // 3D kernels: grain size (contiguous in memory) on x
    dim3 threads2(32,1,32) ;
    dim3 blocks2((_W.Nd+1 + 31)/32,(_g.Nphi + 2*_g.Nghost),(_g.NR + 2*_g.Nghost+31)/32) ;

    dim3 threads3(32,1,32) ;
    dim3 blocks3((_W.Nd + 31)/32,(_g.Nphi + 2*_g.Nghost),(_g.NR + 2*_g.Nghost+31)/32) ;
          
    Field3D<double> Sigs = Field3D<double>(_g.NR + 2*_g.Nghost, _g.Nphi + 2*_g.Nghost, _W.Nd+1);
    Field3D<double> Sigs_1 = Field3D<double>(_g.NR + 2*_g.Nghost, _g.Nphi + 2*_g.Nghost, _W.Nd+1);
    Field3D<double> Sigs_0 = Field3D<double>(_g.NR + 2*_g.Nghost, _g.Nphi + 2*_g.Nghost, _W.Nd+1);
    Field3D<Prims1D> W_nofloor = Field3D<Prims1D>(_g.NR + 2*_g.Nghost, _g.Nphi + 2*_g.Nghost, _W.Nd);

    copy_initial_values<<<blocks2, threads2>>>(_g, Sigs, _mol, Field3DRef<Prims1D>(W_nofloor), _W, _Wg, _floor) ; 
    _copy_rhos<<<blocks2,threads2>>>(_g, Sigs, Sigs_0);

    int it = 0;
    double err_tot= 1;

    while (err_tot > 1e-5) {

        _copy_rhos<<<blocks2,threads2>>>(_g, Sigs, Sigs_1);

        _implicit_update<<<blocks,threads>>>(_g, Field3DRef<Prims1D>(W_nofloor), _Wg, _T, _sizes.grain_props, _sizes.grain_masses(), N_s, _mol, Sigs, Sigs_0, _mu, _alpha, _GMstar, dt);

        err_tot = _relative_change(_g, Sigs, Sigs_1, _W.Nd, _floor, _Wg);

        it++;
        _update_sizegrid<<<blocks3,threads3>>>(_g, _sizes.grain_props, density_view(_W), density_view(Sigs), _sizes.grain_masses(), _sizes.solid_density(), _sizes.ice_density());
        cudaDeviceSynchronize();
    }
    dt_chem = dt * 0.01/_relative_change(_g, Sigs, Sigs_0, _W.Nd, _floor, _Wg);

    copy_final_values<<<blocks2,threads2>>>(_g, Sigs, _mol, _floor, _Wg);
}

void SizeGridIce::launch_update_sizegrid(DensityView rho_dust, DensityView rho_ice) {

    dim3 threads3(32,4,2) ;  // grain size (contiguous in memory) on x
    dim3 blocks3((rho_dust.Nd + 31)/32,(_g.Nphi + 2*_g.Nghost+3)/4,(_g.NR + 2*_g.Nghost+1)/2) ;

    _update_sizegrid<<<blocks3,threads3>>>(_g, grain_props, rho_dust, rho_ice,
                                            grain_masses(), solid_density(), ice_density());
    cudaDeviceSynchronize();
}

void SizeGridIce::_update_sizes(const Field<Prims>& /*Wg*/, DensityView rho_dust, DensityView rho_ice) {
    launch_update_sizegrid(rho_dust, rho_ice);
}

void SizeGridIce::_update_sizes(const Field<Prims1D>& /*Wg*/, DensityView rho_dust, DensityView rho_ice) {
    launch_update_sizegrid(rho_dust, rho_ice);
}

// DensityView construction (see density_view.h). get_rho on the first
// element gives the address of the first density; nothing is read.

template<typename T>
static DensityView make_density_view(Field3DRef<T> f) {
    static_assert(std::is_same<decltype(get_rho(*f.get())), double&>::value,
                  "DensityView requires the density to be stored as a double") ;
    return DensityView{reinterpret_cast<char*>(&get_rho(*f.get())), sizeof(T), f.Nd, f.stride_Zd, f.stride_d} ;
}

DensityView density_view(Field3DRef<double> f)  { return make_density_view(f) ; }
DensityView density_view(Field3DRef<Prims> f)   { return make_density_view(f) ; }
DensityView density_view(Field3DRef<Prims1D> f) { return make_density_view(f) ; }
DensityView density_view(Field3DRef<Quants> f)  { return make_density_view(f) ; }



// Latent heat calculation

__global__ void add_latent_heating_device(GridRef g, double L_latent, FieldRef<double> drhovdt, FieldRef<double> heating) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ; 

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) {  
            heating(i,j) -= drhovdt(i,j) * L_latent;
        }
    }
}

/**
 * Latent heating calculation following https://arxiv.org/pdf/2502.08936
 */
void IceVapChem::add_latent_heating(double L_latent, Field<double>& heating) {

    dim3 threads(32,16,1) ;
    dim3 blocks((_g.NR + 2*_g.Nghost+31)/32,(_g.Nphi + 2*_g.Nghost+15)/16,1) ;

    add_latent_heating_device<<<blocks, threads>>>(_g, L_latent, _drhovdt, heating) ;
}

template __host__ __device__ ChemRate R_d_jac<Prims>(MoleculeRef mol, Field3DRef<double> ice_grain, double N_s, FieldConstRef<double>& T, Field3DRef<Grain>& grains, 
                    Field3DRef<Prims>& W, const RealType* m, int i, int j, int k);
template __host__ __device__ ChemRate R_d_jac<Prims1D>(MoleculeRef mol, Field3DRef<double> ice_grain, double N_s, FieldConstRef<double>& T, Field3DRef<Grain>& grains, 
                    Field3DRef<Prims1D>& W, const RealType* m, int i, int j, int k);

template __global__ void copy_initial_values<Prims>(GridRef g, Field3DRef<double> rhos, MoleculeRef mol, Field3DRef<Prims> w_nof, Field3DRef<Prims> w, FieldRef<Prims> wg, double floor);
template __global__ void copy_initial_values<Prims1D>(GridRef g, Field3DRef<double> rhos, MoleculeRef mol, Field3DRef<Prims1D> w_nof, Field3DRef<Prims1D> w, FieldRef<Prims1D> wg, double floor);

template __global__ void copy_final_values<Prims>(GridRef g, Field3DRef<double> rhos, MoleculeRef mol, double floor, FieldRef<Prims> wg);
template __global__ void copy_final_values<Prims1D>(GridRef g, Field3DRef<double> rhos, MoleculeRef mol, double floor, FieldRef<Prims1D> wg);


