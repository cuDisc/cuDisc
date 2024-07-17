#include <iostream>
#include <cuda_runtime.h>

#include "grid.h"
#include "field.h"
#include "cuda_array.h"
#include "dustdynamics.h"
#include "constants.h"
#include "sources.h"
#include "drag_const.h"

#include "coagulation/size_grid.h"

// Simple Scheme

__device__
double OmK2(GridRef& g, double Mstar, int i, int j) {

    return GMsun * Mstar / std::pow(g.Rc(i)*g.Rc(i)+g.Zc(i,j)*g.Zc(i,j), 1.5);

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
void _source_drag(GridRef g, Field3DRef<Prims> w, FieldConstRef<Prims> w_gas, Field3DConstRef<double> t_stop, double dt, double Mstar) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.z * blockDim.z ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost; j+=jstride) {
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

template<bool full_stokes>
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
                double cs = sqrt(k_B*T(i,j)/(mu*m_H));
                t_stop(i,j,k) = calc_t_s<full_stokes>(q(i,j,k), w_gas(i,j), s[k], rho_m, cs, mu);
            }
        }
    }
}

template<bool use_full_stokes>
void Sources<use_full_stokes>::source_exp(Grid& g, Field3D<Prims>& w, Field3D<Quants>& u, double dt) {

    dim3 threads(16,8,8);
    dim3 blocks((g.NR + 2*g.Nghost+15)/16,(g.Nphi + 2*g.Nghost+7)/8, (u.Nd+7)/8) ;

    _source_curv_grav<<<blocks,threads>>>(g, w, u, _w_gas, dt, _Mstar, _floor);
}

template<bool use_full_stokes>
void Sources<use_full_stokes>::source_imp(Grid& g, Field3D<Prims>& w, double dt) {

    Field3D<double> t_stop = Field3D<double>(g.NR+2*g.Nghost,g.Nphi+2*g.Nghost,w.Nd);

    dim3 threads(16,8,8);
    dim3 blocks((g.NR + 2*g.Nghost+15)/16,(g.Nphi + 2*g.Nghost+7)/8, (w.Nd+7)/8) ;

    if (use_full_stokes) 
        _calc_t_s<true><<<blocks,threads>>>(g, w, _w_gas, _T, t_stop, _sizes.grain_sizes(), _sizes.solid_density(), _mu);
    else
        _calc_t_s<false><<<blocks,threads>>>(g, w, _w_gas, _T, t_stop, _sizes.grain_sizes(), _sizes.solid_density(), _mu);
    _source_drag<<<blocks,threads>>>(g, w, _w_gas, t_stop, dt, _Mstar);
}

template<bool use_full_stokes>
void SourcesRad<use_full_stokes>::source_exp(Grid& g, Field3D<Prims>& w, Field3D<Quants>& u, double dt) {

    dim3 threads(16,8,8);
    dim3 blocks((g.NR + 2*g.Nghost+15)/16,(g.Nphi + 2*g.Nghost+7)/8, (u.Nd+7)/8) ;

    _source_curv_grav_pressure<<<blocks,threads>>>(g, w, u, _w_gas, _f_rad, dt, _Mstar, _floor);
}

template<bool use_full_stokes>
void SourcesRad<use_full_stokes>::source_imp(Grid& g, Field3D<Prims>& w, double dt) {

    Field3D<double> t_stop = Field3D<double>(g.NR+2*g.Nghost,g.Nphi+2*g.Nghost,w.Nd);

    dim3 threads(16,8,8);
    dim3 blocks((g.NR + 2*g.Nghost+15)/16,(g.Nphi + 2*g.Nghost+7)/8, (w.Nd+7)/8) ;

    if (use_full_stokes) 
        _calc_t_s<true><<<blocks,threads>>>(g, w, _w_gas, _T, t_stop, _sizes.grain_sizes(), _sizes.solid_density(), _mu);
    else
        _calc_t_s<false><<<blocks,threads>>>(g, w, _w_gas, _T, t_stop, _sizes.grain_sizes(), _sizes.solid_density(), _mu);
    _source_drag<<<blocks,threads>>>(g, w, _w_gas, t_stop, dt, _Mstar);
}

template class Sources<true>;
template class Sources<false>;
template class SourcesRad<true>;
template class SourcesRad<false>;