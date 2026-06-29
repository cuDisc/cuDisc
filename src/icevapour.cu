#include <iostream>
#include <algorithm>
#include <cassert>

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

    drho_vap/dt = -sum(R_a,n) rho_vap + sum(R_d,n rho_ice,n)

    drho_ice,n/dt = R_a,n rho_vap - R_d,n rho_ice,n

*/

__host__ __device__
double nu_i(MoleculeRef mol, double N_s) {

    double nu_0 = std::sqrt(2*N_s*k_B / (m_H * M_PI*M_PI));

    return nu_0 * std::sqrt(mol.T_bind/(mol.m_mol/m_H));
}

template<typename Te>
__host__ __device__
ChemRate R_d_jac(MoleculeRef mol, Field3DRef<double> ice_grain, double N_s, FieldConstRef<double>& T, Field3DRef<Ice>& ice, 
                    Field3DRef<Te>& W, const RealType* a, const RealType* m, int i, int j, int k) {

    ChemRate Rd;
    
    // Scaled 0th order rate 
    if (W(i,j,k)[0] == 0.) {
        Rd.rate = 0.;
        Rd.jac = 0.;
    }
    else {
        double R = nu_i(mol, N_s) * std::exp(-mol.T_bind/T(i,j));

        double mass_per_layer = 4.*M_PI * ice(i,j,k).a * ice(i,j,k).a * N_s * W(i,j,k)[0] / m[k] * mol.m_mol;
        // double mass_per_layer = 4.*M_PI * a[k] * a[k] * N_s * W(i,j,k).rho / m[k] * mol.m_mol;
        double num_layers = ice_grain(i,j,k) / max(mass_per_layer,1e-100); 

        Rd.rate = R * (-expm1(-num_layers))/ max(num_layers,1e-100);//mass_per_layer * R / max(ice_grain(i,j,k),1e-100);// (1+num_layers);
        // if (i==50 && j==170 && k<10) {printf("%d %g %g\n", k, num_layers, (1.-std::exp(-num_layers))/ max(num_layers,1e-100));}
        Rd.jac = -R * num_layers / ((1+num_layers)*(1+num_layers));
    }

    return Rd;
}

__host__ __device__
ChemRate R_d_ph_jac(MoleculeRef mol, Field3DRef<double> ice_grain, double N_s, FieldConstRef<double>& T, Field3DRef<Ice>& ice, 
                    Field3DRef<Prims>& W, const RealType* a, const RealType* m, Field3DConstRef<double> J, FieldRef<Prims> Wg, 
                    FieldRef<double>& F_UV, int Jbin_idx, double* lam_bins, double area_tot, int i, int j, int k) {

    ChemRate Rd;
    
    // Scaled 0th order rate 
    if (W(i,j,k)[0] == 0.) {
        Rd.rate = 0.;
        Rd.jac = 0.;
    }
    else {
        double R = nu_i(mol, N_s) * std::exp(-mol.T_bind/T(i,j));

        double temp = 4.*M_PI * ice(i,j,k).a * ice(i,j,k).a * N_s * W(i,j,k)[0] / m[k] * mol.m_mol;
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
                    Field3DRef<Prims>& W, const RealType* m, Field3DRef<Ice>& ice, int Jbin_idx, double* lam_bins, int i, int j, int k) {
    
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

        for (int l=0; l<ice.Nd; l++) {
            sum_mfp += M_PI * ice(i,j,l).a * ice(i,j,l).a * W(i,j,l).rho / m[l];   
        }

        double gamma_CR = 0.15*eta_CR*n_H / max(sum_mfp,1e-100);

        double mass_per_layer = 4.*M_PI * ice(i,j,k).a * ice(i,j,k).a * N_s * W(i,j,k).rho / m[k] * mol.m_mol;
        double num_layers = ice_grain(i,j,k) / max(mass_per_layer,1e-100); 

        
        Rd.rate  = (gamma_UV+gamma_CR) * Y * (-expm1(-num_layers))/ max(4.*N_s*num_layers,1e-100);
        Rd.jac = 0.;
    }

    return Rd;
}


__host__ __device__
ChemRate R_a_jac(MoleculeRef mol, FieldConstRef<double> T, Field3DRef<Prims>& W, Field3DRef<Ice>& ice, const RealType* m, const RealType* a, int i, int j, int k) {
    
    double v_th = std::sqrt(8.*k_B*T(i,j)/(M_PI*mol.m_mol));

    double R = M_PI * ice(i,j,k).a * ice(i,j,k).a * v_th * W(i,j,k)[0] / m[k];
    
    ChemRate Ra;
    Ra.rate = R;
    Ra.jac = 0.;

    return Ra;
}

__host__ __device__
ChemRate R_a_jac(GridRef g, MoleculeRef mol, FieldConstRef<double> T, Field3DRef<Prims1D>& W, FieldRef<Prims1D>& W_g, Field3DRef<Ice>& ice, const RealType* m, const RealType* a, double mu, double alpha, double GMstar, int i, int j, int k) {
    
    double v_th = std::sqrt(8.*k_B*T(i,j)/(M_PI*mol.m_mol));

    // double St = M_PI/2. * ice(i,j,k).a*ice(i,j,k).rho / W_g(i,j).Sig;
    double H = std::sqrt(k_B*T(i,j)*g.Rc(i)*g.Rc(i)*g.Rc(i)/(mu*m_H*GMstar));//*min(1.,sqrt(alpha/(min(St,0.5)*(1.+St*St))));
    double R = M_PI * ice(i,j,k).a * ice(i,j,k).a * v_th * W(i,j,k)[0] / m[k] / (std::sqrt(2.*M_PI)*H);

    ChemRate Ra;
    Ra.rate = R;
    Ra.jac = 0.;

    return Ra;
}

template<typename Te>
__global__ void _update_sizegrid(GridRef g, Field3DRef<Ice> ice, Field3DRef<Te> W, Field3DRef<double> rho_ice, const RealType* m, RealType rho_ms, RealType rho_mi) {

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
                double rho_1 = (max(rhoice(i,j,k).rho,0.)/(W(i,j,k)[0] * rho_mi) + 1./rho_ms);
                ice(i,j,k).a = pow((3.*m[k]/(4.*M_PI)) * rho_1, 1./3.);
                ice(i,j,k).rho = (max(rhoice(i,j,k).rho,0.) + W(i,j,k)[0]) / (W(i,j,k)[0] * rho_1);
                // if (i==g.NR+g.Nghost-1 && j==82 && k==0) {printf("%g %g\n",rhoice(i,j,k).rho,W(i,j,k)[0]);}// rho_1, ice(i,j,k).a, ice(i,j,k).rho);}
            } 
        }
    }

}
__global__ void _update_sizegrid(GridRef g, Field3DRef<Ice> ice, Field3DRef<Prims1D> W, Field3DRef<Prims1D> rhoice, const RealType* m, RealType rho_ms, RealType rho_mi) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.z * blockDim.z ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost; j+=jstride) {
            for (int k=kidx; k<W.Nd; k+=kstride) {
                double rho_1 = (max(rhoice(i,j,k)[0],0.)/(W(i,j,k)[0] * rho_mi) + 1./rho_ms);
                ice(i,j,k).a = pow((3.*m[k]/(4.*M_PI)) * rho_1, 1./3.);
                ice(i,j,k).rho = (max(rhoice(i,j,k)[0],0.) + W(i,j,k)[0]) / (W(i,j,k)[0] * rho_1);
            } 
        }
    }

}


__global__ void _implicit_update(GridRef g, Field3DRef<Prims> W, FieldRef<Prims> Wg, FieldConstRef<double> T, Field3DConstRef<double> J, Field3DRef<Ice> ice, const RealType* a, 
                                    const RealType* m, double N_s, MoleculeRef mol, Field3DRef<double> rhos, Field3DRef<double> rhos_0, FieldRef<double> F_UV, int Jbin_idx, double* lam_bins, double dt) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost; j+=jstride) {

            double A = 0., B = 0., area_tot=0.;
            int ndust = mol.ice.Nd;

            for (int k=0; k < ndust; k++) {
                area_tot += M_PI * ice(i,j,k).a * ice(i,j,k).a * W(i,j,k)[0]/ m[k];   
            }

            for (int k=0; k < ndust; k++) {
                
                ChemRate R_a = R_a_jac(mol, T, W, ice, m,a,i,j,k);
                ChemRate R_d = R_d_ph_jac(mol, rhos, N_s, T, ice, W, a, m, J, Wg, F_UV, Jbin_idx, lam_bins, area_tot, i,j,k);
                A += (R_d.rate) * dt * rhos_0(i,j,k) / (1. + (R_d.rate) * dt);
                B += R_a.rate * dt / (1. + (R_d.rate) * dt);
            }

            rhos(i,j,ndust) = (rhos_0(i,j,ndust) + A) / (1. + B);

            for (int k=0; k < ndust; k++) {

                ChemRate R_a = R_a_jac(mol, T, W, ice, m,a, i,j,k);
                ChemRate R_d = R_d_ph_jac(mol, rhos, N_s, T, ice, W, a, m, J, Wg, F_UV, Jbin_idx, lam_bins, area_tot, i,j,k);

                rhos(i,j,k) = (rhos_0(i,j,k) + R_a.rate * dt * rhos(i,j,ndust))  / (1. + (R_d.rate) * dt);

            }

        }
    }

}

__global__ void _implicit_update(GridRef g, Field3DRef<Prims1D> W, FieldRef<Prims1D> Wg, FieldConstRef<double> T, Field3DRef<Ice> ice, const RealType* a, 
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
                
                ChemRate R_a = R_a_jac(g, mol, T, W, Wg, ice, m, a, mu, alpha, GMstar, i,j,k);
                ChemRate R_d = R_d_jac(mol, rhos, N_s, T, ice, W, a, m, i,j,k);
                // ChemRate R_phd = R_ph_jac(mol, rhos, N_s, J, Wg, W, m, ice, Jbin_idx, lam_bins, i,j,k);
                A += (R_d.rate) * dt * rhos_0(i,j,k) / (1. + (R_d.rate) * dt);
                B += R_a.rate * dt / (1. + (R_d.rate) * dt);
            }

            rhos(i,j,ndust) = (rhos_0(i,j,ndust) + A) / (1. + B);

            for (int k=0; k < ndust; k++) {

                ChemRate R_a = R_a_jac(g, mol, T, W, Wg, ice, m, a, mu, alpha, GMstar, i,j,k);
                ChemRate R_d = R_d_jac(mol, rhos, N_s, T, ice, W, a, m, i,j,k);
                // ChemRate R_phd = R_ph_jac(mol, rhos, N_s, J, Wg, W, m, ice, Jbin_idx, lam_bins, i,j,k);

                rhos(i,j,k) = (rhos_0(i,j,k) + R_a.rate * dt * rhos(i,j,ndust))  / (1. + (R_d.rate) * dt);

            }

        }
    }

}

template<typename T>
__global__ void copy_initial_values(GridRef g, Field3DRef<double> rhos, MoleculeRef mol, Field3DRef<T> w_nof, Field3DRef<T> w, FieldRef<T> wg, double floor) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    int n_grains = mol.ice.Nd;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost; j+=jstride) {
            
            rhos(i,j,n_grains) = max(mol.vap(i,j) - 1.1e-100*floor*wg(i,j)[0], 0.);
            for (int k=0; k<n_grains; k++) {
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

template<typename T>
__global__ void copy_final_values(GridRef g, Field3DRef<double> rhos, MoleculeRef mol, double floor, FieldRef<T> wg) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    int n_grains = mol.ice.Nd;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost; j+=jstride) {
            
            mol.vap(i,j) = rhos(i,j,n_grains)+floor*1e-100*wg(i,j)[0];

            for (int k=0; k<n_grains; k++) {
                mol.ice(i,j,k) = rhos(i,j,k)+floor*1e-100*wg(i,j)[0];
            }
        }
    }

}

template<typename T>
__global__ void get_tol(Field3DRef<double> rhos, Field3DRef<double> rhos_0, GridRef g, int ngrains, FieldRef<double> err, double floor, FieldRef<T> wg) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost; j+=jstride) {
            for (int k=0; k<ngrains+1; k++) {
                err(i,j) += abs(max(rhos(i,j,k),floor*wg(i,j)[0]) - max(rhos_0(i,j,k),floor*wg(i,j)[0])) / max(rhos_0(i,j,k),floor*wg(i,j)[0]) / ((ngrains + 1) * g.NR * g.Nphi);
            }
        }
    }
}
__global__ void set_tol(GridRef g, FieldRef<double> err) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) {
            err(i,j) = 0.;
        }
    }
}

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
          
    dim3 threads2(16,16,4) ;
    dim3 blocks2((_g.NR + 2*_g.Nghost+15)/16,(_g.Nphi + 2*_g.Nghost+15)/16,(_W.Nd+1 + 3)/4) ;

    dim3 threads3(16,16,4) ;
    dim3 blocks3((_g.NR + 2*_g.Nghost+15)/16,(_g.Nphi + 2*_g.Nghost+15)/16, (_W.Nd + 3)/4);
          
    Field3D<double> rhos = Field3D<double>(_g.NR + 2*_g.Nghost, _g.Nphi + 2*_g.Nghost, _W.Nd+1);
    Field3D<double> rhos_1 = Field3D<double>(_g.NR + 2*_g.Nghost, _g.Nphi + 2*_g.Nghost, _W.Nd+1);
    Field3D<double> rhos_0 = Field3D<double>(_g.NR + 2*_g.Nghost, _g.Nphi + 2*_g.Nghost, _W.Nd+1);
    Field3D<Prims> W_nofloor = Field3D<Prims>(_g.NR + 2*_g.Nghost, _g.Nphi + 2*_g.Nghost, _W.Nd);

    _floor_above_phdiss<<<blocks,threads>>>(_g, _Wg, _mol, _h_phdiss.get(), _floor);

    copy_initial_values<<<blocks, threads>>>(_g, rhos, _mol, Field3DRef<Prims>(W_nofloor), _W, _Wg, _floor) ; 
    _copy_rhos<<<blocks2,threads2>>>(_g, rhos, rhos_0);

    int it = 0;
    Field<double> err = Field<double>(_g.NR + 2*_g.Nghost, _g.Nphi + 2*_g.Nghost);
    double err_tot= 1;

    while (err_tot > 1e-5) {

        set_tol<<<blocks,threads>>>(_g, err);

        _copy_rhos<<<blocks2,threads2>>>(_g, rhos, rhos_1);

        _implicit_update<<<blocks,threads>>>(_g, Field3DRef<Prims>(W_nofloor), _Wg, _T, _J, _sizes.ice, _sizes.grain_sizes(), _sizes.grain_masses(), N_s, _mol, rhos, rhos_0, _F_UV, _Jbin_idx, _bins.bands.get(), dt);

        get_tol<<<blocks,threads>>>(rhos, rhos_1, _g, _W.Nd, err, _floor, _Wg);
        Reduction::scan_R_sum(_g,err);
        Reduction::scan_Z_sum(_g,err);
        cudaDeviceSynchronize();
        err_tot = err(_g.NR + 2*_g.Nghost-1,_g.Nphi + 2*_g.Nghost-1);
        
        it++;
        _update_sizegrid<<<blocks3,threads3>>>(_g, _sizes.ice, _W, rhos, _sizes.grain_masses(), _sizes.solid_density(), _sizes.ice_density());
    }

    set_tol<<<blocks,threads>>>(_g, err);
    get_tol<<<blocks,threads>>>(rhos, rhos_0, _g, _W.Nd, err, _floor, _Wg);
    Reduction::scan_R_sum(_g,err);
    Reduction::scan_Z_sum(_g,err);
    cudaDeviceSynchronize();
    dt_chem = dt * 0.01/err(_g.NR + 2*_g.Nghost-1,_g.Nphi + 2*_g.Nghost-1);

    _calc_drhovdt<<<blocks,threads>>>(_g, _drhovdt, rhos, rhos_0, dt);

    copy_final_values<<<blocks,threads>>>(_g, rhos, _mol, _floor, _Wg);

    _compute_mu<<<blocks,threads>>>(_g, _Wg, _mol.vap, _mu, _mu_HHe, _mol.m_mol/m_H);
}

void IceVapChem1D::imp_update(double dt, double& dt_chem) {

    size_t threads = 512;
    size_t blocks = (_g.NR + 2*_g.Nghost+511)/512 ;
          
    dim3 threads2(32,1,32) ;
    dim3 blocks2((_g.NR + 2*_g.Nghost+15)/16,(_g.Nphi + 2*_g.Nghost),(_W.Nd+1 + 31)/32) ;

    dim3 threads3(32,1,32) ;
    dim3 blocks3((_g.NR + 2*_g.Nghost+15)/16,(_g.Nphi + 2*_g.Nghost), (_W.Nd + 31)/32);
          
    Field3D<double> Sigs = Field3D<double>(_g.NR + 2*_g.Nghost, _g.Nphi + 2*_g.Nghost, _W.Nd+1);
    Field3D<double> Sigs_1 = Field3D<double>(_g.NR + 2*_g.Nghost, _g.Nphi + 2*_g.Nghost, _W.Nd+1);
    Field3D<double> Sigs_0 = Field3D<double>(_g.NR + 2*_g.Nghost, _g.Nphi + 2*_g.Nghost, _W.Nd+1);
    Field3D<Prims1D> W_nofloor = Field3D<Prims1D>(_g.NR + 2*_g.Nghost, _g.Nphi + 2*_g.Nghost, _W.Nd);

    copy_initial_values<<<blocks, threads>>>(_g, Sigs, _mol, Field3DRef<Prims1D>(W_nofloor), _W, _Wg, _floor) ; 
    _copy_rhos<<<blocks2,threads2>>>(_g, Sigs, Sigs_0);

    int it = 0;
    Field<double> err = Field<double>(_g.NR + 2*_g.Nghost, _g.Nphi + 2*_g.Nghost);
    double err_tot= 1;

    while (err_tot > 1e-5) {

        set_tol<<<blocks,threads>>>(_g, err);

        _copy_rhos<<<blocks2,threads2>>>(_g, Sigs, Sigs_1);

        _implicit_update<<<blocks,threads>>>(_g, Field3DRef<Prims1D>(W_nofloor), _Wg, _T, _sizes.ice, _sizes.grain_sizes(), _sizes.grain_masses(), N_s, _mol, Sigs, Sigs_0, _mu, _alpha, _GMstar, dt);

        get_tol<<<blocks2,threads2>>>(Sigs, Sigs_1, _g, _W.Nd, err, _floor, _Wg);
        Reduction::scan_R_sum(_g,err);
        Reduction::scan_Z_sum(_g,err);
        cudaDeviceSynchronize();
        err_tot = err(_g.NR + 2*_g.Nghost-1,_g.Nphi + 2*_g.Nghost-1);

        it++;
        _update_sizegrid<<<blocks3,threads3>>>(_g, _sizes.ice, _W, Sigs, _sizes.grain_masses(), _sizes.solid_density(), _sizes.ice_density());
        cudaDeviceSynchronize();
    }
    set_tol<<<blocks,threads>>>(_g, err);
    get_tol<<<blocks2,threads2>>>(Sigs, Sigs_0, _g, _W.Nd, err, _floor, _Wg);
    Reduction::scan_R_sum(_g,err);
    Reduction::scan_Z_sum(_g,err);
    cudaDeviceSynchronize();
    dt_chem = dt * 0.01/err(_g.NR + 2*_g.Nghost-1,_g.Nphi + 2*_g.Nghost-1);

    copy_final_values<<<blocks,threads>>>(_g, Sigs, _mol, _floor, _Wg);
}


void update_sizegrid(Grid& g, SizeGridIce& sizes, Field3D<Quants>& Qd, Field3D<Quants>& ice) {

    dim3 threads3(16,16,4) ;
    dim3 blocks3((g.NR + 2*g.Nghost+15)/16,(g.Nphi + 2*g.Nghost+15)/16, (Qd.Nd + 3)/4);

    _update_sizegrid<<<blocks3,threads3>>>(g, sizes.ice, Qd, ice, sizes.grain_masses(), sizes.solid_density(), sizes.ice_density());
}

void update_sizegrid(Grid& g, SizeGridIce& sizes, Field3D<Prims>& Qd, Field3D<double>& ice) {

    dim3 threads3(16,16,4) ;
    dim3 blocks3((g.NR + 2*g.Nghost+15)/16,(g.Nphi + 2*g.Nghost+15)/16, (Qd.Nd + 3)/4);

    _update_sizegrid<<<blocks3,threads3>>>(g, sizes.ice, Field3DRef<Prims>(Qd), ice, sizes.grain_masses(), sizes.solid_density(), sizes.ice_density());
    cudaDeviceSynchronize();
}

void update_sizegrid(Grid& g, SizeGridIce& sizes, Field3D<Prims1D>& Qd, Field3D<Prims1D>& ice) {

    dim3 threads3(32,1,32) ;
    dim3 blocks3((g.NR + 2*g.Nghost+31)/32,(g.Nphi + 2*g.Nghost), (Qd.Nd + 31)/32);

    _update_sizegrid<<<blocks3,threads3>>>(g, sizes.ice, Field3DRef<Prims1D>(Qd), Field3DRef<Prims1D>(ice), sizes.grain_masses(), sizes.solid_density(), sizes.ice_density());
    cudaDeviceSynchronize();
}


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


template __global__ void _update_sizegrid<Prims>(GridRef g, Field3DRef<Ice> ice, Field3DRef<Prims> W, Field3DRef<double> rho_ice, const RealType* m, RealType rho_ms, RealType rho_mi);
template __global__ void _update_sizegrid<Prims1D>(GridRef g, Field3DRef<Ice> ice, Field3DRef<Prims1D> W, Field3DRef<double> rho_ice, const RealType* m, RealType rho_ms, RealType rho_mi);

template __host__ __device__ ChemRate R_d_jac<Prims>(MoleculeRef mol, Field3DRef<double> ice_grain, double N_s, FieldConstRef<double>& T, Field3DRef<Ice>& ice, 
                    Field3DRef<Prims>& W, const RealType* a, const RealType* m, int i, int j, int k);
template __host__ __device__ ChemRate R_d_jac<Prims1D>(MoleculeRef mol, Field3DRef<double> ice_grain, double N_s, FieldConstRef<double>& T, Field3DRef<Ice>& ice, 
                    Field3DRef<Prims1D>& W, const RealType* a, const RealType* m, int i, int j, int k);

template __global__ void copy_initial_values<Prims>(GridRef g, Field3DRef<double> rhos, MoleculeRef mol, Field3DRef<Prims> w_nof, Field3DRef<Prims> w, FieldRef<Prims> wg, double floor);
template __global__ void copy_initial_values<Prims1D>(GridRef g, Field3DRef<double> rhos, MoleculeRef mol, Field3DRef<Prims1D> w_nof, Field3DRef<Prims1D> w, FieldRef<Prims1D> wg, double floor);

template __global__ void copy_final_values<Prims>(GridRef g, Field3DRef<double> rhos, MoleculeRef mol, double floor, FieldRef<Prims> wg);
template __global__ void copy_final_values<Prims1D>(GridRef g, Field3DRef<double> rhos, MoleculeRef mol, double floor, FieldRef<Prims1D> wg);

template __global__ void get_tol(Field3DRef<double> rhos, Field3DRef<double> rhos_0, GridRef g, int ngrains, FieldRef<double> err, double floor, FieldRef<Prims> wg);
template __global__ void get_tol(Field3DRef<double> rhos, Field3DRef<double> rhos_0, GridRef g, int ngrains, FieldRef<double> err, double floor, FieldRef<Prims1D> wg);