#include <iostream>
#include <cuda_runtime.h>

#include "grid.h"
#include "field.h"
#include "cuda_array.h"
#include "dustdynamics.h"
#include "constants.h"
#include "sources.h"

#include "coagulation/size_grid.h"


// FARGO3D source term update https://iopscience.iop.org/article/10.3847/1538-4365/ab0a0e/pdf

__device__
double alpha_ij(Field3DConstRef<double> &t_stop, Field3DConstRef<Prims> &q, int i, int j, int l, int m) {

    // gas is index 0

    if (j == 0) {  
        return 1./t_stop(l,m,i);
    }
    else if (i == 0) {
        return (q(l,m,j).rho/q(l,m,0).rho) * (1./t_stop(l,m,j));
    }
    else {
        return 0;
    }
}

__device__ 
double M_ij(Field3DConstRef<double> &t_stop, Field3DConstRef<Prims> &q, int i, int j, int l, int m) {

    double M_ij = 0;

    if (i == j) {
        for (int k=0; k<q.Nd; k++) {
            if (k != i) {
                M_ij += alpha_ij(t_stop, q, i, k, l, m);
            }
        }
        return M_ij;
    }
    else { 
        M_ij -= alpha_ij(t_stop, q, i, j, l, m);
        return M_ij;
    }
}

__device__
void generateTmatrix(Field3DConstRef<double> &t_stop, Field3DConstRef<Prims> &q, 
                        double dt, FieldRef<double> &Tmat, int l, int m) {

    for (int i=0; i<t_stop.Nd; i++) {
        for (int j=0; j<t_stop.Nd; j++) {

            if (i == j) {
                Tmat(i,j) = 1. + dt * M_ij(t_stop, q, i, j, l, m);
            }
            else {
                Tmat(i,j) = dt * M_ij(t_stop, q, i, j, l, m);
            }
        }
    }
}

__global__
void testT(Field3DConstRef<double> t_stop, Field3DConstRef<Prims> q, FieldRef<double> Tmat,
                        double dt, int l, int m) {

    generateTmatrix(t_stop, q, dt, Tmat, l, m);
    printf("%g %g %g %g \n", Tmat(0,0), Tmat(0,1), Tmat(0,2), Tmat(0,3));
    printf("%g %g %g %g \n", Tmat(1,0), Tmat(1,1), Tmat(1,2), Tmat(1,3));
    printf("%g %g %g %g \n", Tmat(2,0), Tmat(2,1), Tmat(2,2), Tmat(2,3));
    printf("%g %g %g %g \n", Tmat(3,0), Tmat(3,1), Tmat(3,2), Tmat(3,3));

}

void testTcpu(const Field3D<double> &t_stop, const Field3D<Prims> &q, 
                        double dt, int l, int m) {

    Field<double> Tmat = Field<double>(q.Nd, q.Nd);
    
    testT<<<1,1>>>(t_stop, q, Tmat, dt, l, m);
}

// Simple Scheme

__device__
double OmK2(GridRef& g, double Mstar, int i, int j) {

    return GMsun * Mstar / std::pow(g.Rc(i)*g.Rc(i)+g.Zc(i,j)*g.Zc(i,j), 1.5);

}
__device__
double t_s(double rho_g, double rho_m, double s, double T, double mu) {

    double v_th = sqrt(8*k_B*T / (mu*m_H*M_PI));

    return rho_m * s / (rho_g * v_th);
}

__global__
void _source_curv_grav(GridRef g, Field3DRef<Prims> w, Field3DRef<Quants> u, FieldConstRef<Prims> wg, double dt, double Mstar, double floor) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.z * blockDim.z ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) {
            for (int k=kidx; k<w.Nd; k+=kstride) {

                if (w(i,j,k).rho > 1.1*wg(i,j).rho*floor) {

                    double f1 = w(i,j,k).v_phi*w(i,j,k).v_phi/g.Rc(i) - OmK2(g, Mstar, i, j)*g.Rc(i);
                    double f2 = -OmK2(g, Mstar, i, j)*g.Zc(i,j);

                    u(i,j,k).mom_R += dt * f1 * w(i,j,k).rho;
                    u(i,j,k).mom_Z += dt * f2 * w(i,j,k).rho;
                }
            }
        }
    }
}

__global__
void _source_curv_grav(GridRef g, Field3DRef<Prims> w, double dt, double Mstar) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.z * blockDim.z ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) {
            for (int k=kidx; k<w.Nd; k+=kstride) {

                double f1 = w(i,j,k).v_phi*w(i,j,k).v_phi/g.Rc(i) - OmK2(g, Mstar, i, j)*g.Rc(i);
                double f2 = -OmK2(g, Mstar, i, j)*g.Zc(i,j);

                w(i,j,k).v_R += dt * f1;
                w(i,j,k).v_Z += dt * f2;
            }
        }
    }

}

__global__
void _source_curv_grav_pressure(GridRef g, Field3DRef<Prims> w, Field3DConstRef<double> f_rad, double dt, double Mstar) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.z * blockDim.z ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) {
            for (int k=kidx; k<w.Nd; k+=kstride) {

                double f1 = w(i,j,k).v_phi*w(i,j,k).v_phi/g.Rc(i) - OmK2(g, Mstar, i, j)*g.Rc(i) + f_rad(i,j,k)*g.Rc(i)/(w(i,j,k).rho*g.rc(i,j));
                double f2 = -OmK2(g, Mstar, i, j)*g.Zc(i,j) + f_rad(i,j,k)*g.Zc(i,j)/(w(i,j,k).rho*g.rc(i,j));
                
                // if (i==50 && k==10) {
                //     printf("%g %g %g \n", w(i,j,k).v_phi*w(i,j,k).v_phi/(w(i,j,k).rho*g.Rc(i)*g.Rc(i)*g.Rc(i)), w(i,j,k).rho*OmK2(g, Mstar, i, j)*g.Rc(i), f_rad(i,j,k)*g.Rc(i)/g.rc(i,j));
                // }
                
                w(i,j,k).v_R += dt * f1;
                w(i,j,k).v_Z += dt * f2;
            }
        }
    }

}

__global__
void _source_curv_grav_pressure(GridRef g, Field3DRef<Prims> w, Field3DRef<Quants> u, FieldConstRef<Prims> wg, Field3DConstRef<double> f_rad, double dt, double Mstar, double floor) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.z * blockDim.z ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) {
            for (int k=kidx; k<w.Nd; k+=kstride) {

                if (w(i,j,k).rho > 1.1*wg(i,j).rho*floor) {

                    double f1 = w(i,j,k).v_phi*w(i,j,k).v_phi/g.Rc(i) - OmK2(g, Mstar, i, j)*g.Rc(i) + f_rad(i,j,k)*g.Rc(i)/(w(i,j,k).rho*g.rc(i,j));
                    double f2 = -OmK2(g, Mstar, i, j)*g.Zc(i,j) + f_rad(i,j,k)*g.Zc(i,j)/(w(i,j,k).rho*g.rc(i,j));

                    u(i,j,k).mom_R += dt * f1 * w(i,j,k).rho;
                    u(i,j,k).mom_Z += dt * f2 * w(i,j,k).rho;

                }
            }
        }
    }
}


__global__
void _source_drag(GridRef g, Field3DRef<Prims> w, FieldConstRef<Prims> w_gas, Field3DConstRef<double> t_stop, double dt, double Mstar, double floor) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.z * blockDim.z ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) {
            for (int k=kidx; k<w.Nd; k+=kstride) {

                // semi-implicit eulerian [1-dt*df/dy]*dy = dt*f where y=(momR, vphi, momZ) and f is the vector of drag terms

                if (w(i,j,k).rho < floor) {
                    w(i,j,k).v_R = 0.; 
                    w(i,j,k).v_phi = 0.; 
                    w(i,j,k).v_Z = 0.; 
                    continue;
                }

                double f1 = -(1/t_stop(i,j,k))*(w(i,j,k).v_R - w_gas(i,j).v_R);
                double f2 = -(1/t_stop(i,j,k))*(w(i,j,k).v_phi - w_gas(i,j).v_phi);
                double f3 = -(1/t_stop(i,j,k))*(w(i,j,k).v_Z - w_gas(i,j).v_Z);

                w(i,j,k).v_R += dt/(1+dt/t_stop(i,j,k)) * f1;
                w(i,j,k).v_phi += dt/(1+dt/t_stop(i,j,k)) * f2;
                w(i,j,k).v_Z += dt/(1+dt/t_stop(i,j,k)) * f3;

                // double max_v = 1.e5; 

                // if (w(i,j,k).v_Z > max_v) { w(i,j,k).v_Z = max_v; }
                // if (w(i,j,k).v_Z < -max_v) { w(i,j,k).v_Z = -max_v; }
                // if (w(i,j,k).v_R > max_v) { w(i,j,k).v_R = max_v; }
                // if (w(i,j,k).v_R < -max_v) { w(i,j,k).v_R = -max_v; }

                //w(i,j,k).v_R = 0.;
                //w(i,j,k).v_R = w_gas(i,j).v_R;
            }
        }
    }

}
__global__
void _source_drag(GridRef g, Field3DRef<Prims> w, FieldConstRef<Prims> w_gas, Field3DConstRef<double> t_stop, double dt, double Mstar) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.z * blockDim.z ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) {
            for (int k=kidx; k<w.Nd; k+=kstride) {

                // semi-implicit eulerian [1-dt*df/dy]*dy = dt*f where y=(momR, vphi, momZ) and f is the vector of drag terms

                double dv_R   = - (w(i,j,k).v_R - w_gas(i,j).v_R);
                double dv_phi = - (w(i,j,k).v_phi - w_gas(i,j).v_phi);
                double dv_Z   = - (w(i,j,k).v_Z - w_gas(i,j).v_Z);

                double ft = dt/(dt + t_stop(i,j,k)) ;

                w(i,j,k).v_R += ft * dv_R ;
                w(i,j,k).v_phi += ft * dv_phi ;
                w(i,j,k).v_Z += ft * dv_Z ;


                double max_v = 5.e5; 

                if (w(i,j,k).v_Z > max_v) { w(i,j,k).v_Z = max_v; }
                if (w(i,j,k).v_Z < -max_v) { w(i,j,k).v_Z = -max_v; }
                if (w(i,j,k).v_R > max_v) { w(i,j,k).v_R = max_v; }
                if (w(i,j,k).v_R < -max_v) { w(i,j,k).v_R = -max_v; }
            }
        }
    }

}

__global__
void _calc_t_sTL(GridRef g, Field3DConstRef<Prims> w, FieldConstRef<Prims> w_gas, FieldConstRef<double> T, 
                    Field3DRef<double> t_stop, const double* s, double rho_m, double mu) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.z * blockDim.z ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) {
            for (int k=kidx; k<w.Nd; k+=kstride) {

                t_stop(i,j,k) = t_s(w_gas(i,j).rho, rho_m, s[k], T(i,j), mu);
                // if (t_stop(i,j,k)*sqrt(OmK2(g, 1., i, j)) > 0.1) { t_stop(i,j,k) = 0.1/sqrt(OmK2(g, 1., i, j)); }

            }
        }
    }
}

__global__
void _calc_t_s(GridRef g, Field3DConstRef<Prims> q, FieldConstRef<Prims> w_gas, FieldConstRef<double> T, 
                    Field3DRef<double> t_stop, const RealType* s, double rho_m, double mu) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.z * blockDim.z ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) {
            for (int k=kidx; k<q.Nd; k+=kstride) {

                t_stop(i,j,k) = t_s(w_gas(i,j).rho, rho_m, s[k], T(i,j), mu);
                //if (t_stop(i,j,k)*sqrt(OmK2(g, 1., i, j)) > 1.) { t_stop(i,j,k) = 1./sqrt(OmK2(g, 1., i, j)); }
            }
        }
    }
}

__global__
void _calc_t_s(GridRef g, FieldConstRef<Prims> w_gas, FieldConstRef<double> T, 
                    Field3DRef<double> t_stop, const RealType* s, RealType rho_m, double mu) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.z * blockDim.z ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) {
            for (int k=kidx; k<t_stop.Nd; k+=kstride) {

                t_stop(i,j,k) = t_s(w_gas(i,j).rho, rho_m, s[k], T(i,j), mu);
                // if (t_stop(i,j,k)*sqrt(OmK2(g, 1., i, j)) > 0.1) { t_stop(i,j,k) = 0.1/sqrt(OmK2(g, 1., i, j)); }

            }
        }
    }
}

void source_term_updateTL(Grid& g, Field3D<Prims>& q, const Field<Prims>& w_gas, const Field<double>& T, 
                        const CudaArray<double>& s, double dt, double Mstar, double rho_m, double mu, int bound, double floor) {

    Field3D<double> t_stop = Field3D<double>(g.NR+2*g.Nghost,g.Nphi+2*g.Nghost,q.Nd);

    dim3 threads(16,16,4);
    dim3 blocks((g.NR + 2*g.Nghost+15)/16,(g.Nphi + 2*g.Nghost+15)/16, (q.Nd+3)/4) ;
    //dim3 blocks(48,48,48);

    _set_boundaries<<<blocks,threads>>>(g, q, bound, floor);

    _calc_t_sTL<<<blocks,threads>>>(g, q, w_gas, T, t_stop, s.get(), rho_m, mu);

    _source_curv_grav<<<blocks,threads>>>(g, q, dt, Mstar);
    _source_drag<<<blocks,threads>>>(g, q, w_gas, t_stop, dt, Mstar, floor);
}

void source_term_update(Grid& g, Field3D<Prims>& q, const Field<Prims>& w_gas, const Field<double>& T, 
                        const SizeGrid& s, double dt, double Mstar, double rho_m, double mu, int bound, double floor) {

    Field3D<double> t_stop = Field3D<double>(g.NR+2*g.Nghost,g.Nphi+2*g.Nghost,q.Nd);

    dim3 threads(16,8,8);
    dim3 blocks((g.NR + 2*g.Nghost+15)/16,(g.Nphi + 2*g.Nghost+7)/8, (q.Nd+7)/8) ;
    //dim3 blocks(48,48,48);

    _set_boundaries<<<blocks,threads>>>(g, q, bound, floor);

    _calc_t_s<<<blocks,threads>>>(g, q, w_gas, T, t_stop, s.grain_sizes(), rho_m, mu);

    _source_curv_grav<<<blocks,threads>>>(g, q, dt, Mstar);
    _source_drag<<<blocks,threads>>>(g, q, w_gas, t_stop, dt, Mstar, floor);
}

void source_term_update(Grid& g, Field3D<Prims>& q, const Field<Prims>& w_gas, const Field<double>& T, const Field3D<double>& f_rad, 
                        const SizeGrid& s, double dt, double Mstar, double rho_m, double mu, int bound, double floor) {

    Field3D<double> t_stop = Field3D<double>(g.NR+2*g.Nghost,g.Nphi+2*g.Nghost,q.Nd);

    dim3 threads(16,8,8);
    dim3 blocks((g.NR + 2*g.Nghost+15)/16,(g.Nphi + 2*g.Nghost+7)/8, (q.Nd+7)/8) ;
    //dim3 blocks(48,48,48);

    _set_boundaries<<<blocks,threads>>>(g, q, bound, floor);

    _calc_t_s<<<blocks,threads>>>(g, q, w_gas, T, t_stop, s.grain_sizes(), rho_m, mu);

    _source_curv_grav_pressure<<<blocks,threads>>>(g, q, f_rad, dt, Mstar);
    _source_drag<<<blocks,threads>>>(g, q, w_gas, t_stop, dt, Mstar, floor);
}

void Sources::source_exp(Grid& g, Field3D<Prims>& w, Field3D<Quants>& u, double dt) {

    dim3 threads(16,8,8);
    dim3 blocks((g.NR + 2*g.Nghost+15)/16,(g.Nphi + 2*g.Nghost+7)/8, (u.Nd+7)/8) ;

    _source_curv_grav<<<blocks,threads>>>(g, w, u, _w_gas, dt, _Mstar, _floor);
}

void Sources::source_imp(Grid& g, Field3D<Prims>& w, double dt) {

    Field3D<double> t_stop = Field3D<double>(g.NR+2*g.Nghost,g.Nphi+2*g.Nghost,w.Nd);

    dim3 threads(16,8,8);
    dim3 blocks((g.NR + 2*g.Nghost+15)/16,(g.Nphi + 2*g.Nghost+7)/8, (w.Nd+7)/8) ;

    _calc_t_s<<<blocks,threads>>>(g, w, _w_gas, _T, t_stop, _sizes.grain_sizes(), _sizes.solid_density(), _mu);
    _source_drag<<<blocks,threads>>>(g, w, _w_gas, t_stop, dt, _Mstar);
}

void SourcesRad::source_exp(Grid& g, Field3D<Prims>& w, Field3D<Quants>& u, double dt) {

    dim3 threads(16,8,8);
    dim3 blocks((g.NR + 2*g.Nghost+15)/16,(g.Nphi + 2*g.Nghost+7)/8, (u.Nd+7)/8) ;

    _source_curv_grav_pressure<<<blocks,threads>>>(g, w, u, _w_gas, _f_rad, dt, _Mstar, _floor);
}

void SourcesRad::source_imp(Grid& g, Field3D<Prims>& w, double dt) {

    Field3D<double> t_stop = Field3D<double>(g.NR+2*g.Nghost,g.Nphi+2*g.Nghost,w.Nd);

    dim3 threads(16,8,8);
    dim3 blocks((g.NR + 2*g.Nghost+15)/16,(g.Nphi + 2*g.Nghost+7)/8, (w.Nd+7)/8) ;

    _calc_t_s<<<blocks,threads>>>(g, w, _w_gas, _T, t_stop, _sizes.grain_sizes(), _sizes.solid_density(), _mu);
    _source_drag<<<blocks,threads>>>(g, w, _w_gas, t_stop, dt, _Mstar);
}