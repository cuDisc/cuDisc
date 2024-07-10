#include <iostream>
#include "cuda_runtime.h"

#include "grid.h"
#include "field.h"
#include "dustdynamics.h"
#include "dustdynamics1D.h"
#include "cuda_array.h"
#include "reductions.h"
#include "constants.h"
#include "scan.h"
#include "star.h"

/*
1D gas surface density viscous evolution solver
*/

__device__
double slope4Or(GridRef& g, FieldConstRef<double>& Qty, int i, int j) {

    double dim2 = log(g.rc(i,j))-log(g.rc(i-2,j));
    double dim1 = log(g.rc(i,j))-log(g.rc(i-1,j));
    double dip1 = log(g.rc(i+1,j))-log(g.rc(i,j));
    double dip2 = log(g.rc(i+2,j))-log(g.rc(i,j));

    double a = dim1*dip1*dip2 / (dim2*(dim2+dip1)*(dim2-dim1)*(dim2+dip2));
    double b = -dim2*dip1*dip2 / (dim1*(dim2-dim1)*(dim1+dip1)*(dim1+dip2));
    double c = dim2*dim1*dip2 / (dip1*(dim1+dip1)*(dim2+dip1)*(dip2-dip1));
    double d = -dim2*dim1*dip1 / (dip2*(dip2-dip1)*(dim1+dip2)*(dim2+dip2));

    // if ((Qty(i,j)-Qty(i-1,j)) * (Qty(i+1,j)-Qty(i,j)) > 0.) {
    return (a*Qty(i-2,j) + b*Qty(i-1,j) - (a+b+c+d)*Qty(i,j) + c*Qty(i+1,j) + d*Qty(i+2,j))/g.rc(i,j);
    // }
    // else {
    //     return 0.;
    // }
}
__device__
double slope3Or(GridRef& g, FieldConstRef<double>& Qty, int i, int j) {

    double dim2 = log(g.rc(i,j))-log(g.rc(i-2,j));
    double dim1 = log(g.rc(i,j))-log(g.rc(i-1,j));
    double dip1 = log(g.rc(i+1,j))-log(g.rc(i,j));

    double a = dim1*dip1 / (dim2*(dim2+dip1)*(dim2-dim1));
    double b = -dim2*dip1 / (dim1*(dim2-dim1)*(dim1+dip1));
    double c = dim2*dim1 / (dip1*(dim1+dip1)*(dim2+dip1));

    // if ((Qty(i,j)-Qty(i-1,j)) * (Qty(i+1,j)-Qty(i,j)) > 0.) {
    return (a*Qty(i-2,j) + b*Qty(i-1,j) - (a+b+c)*Qty(i,j) + c*Qty(i+1,j))/g.rc(i,j);
    // }
    // else {
    //     return 0.;
    // }
}

__device__
double slope3OZ(GridRef& g, FieldConstRef<double>& Qty, int i, int j) {

    double dim2 = g.Zc(i,j)-g.Zc(i,j-2);
    double dim1 = g.Zc(i,j)-g.Zc(i,j-1);
    double dip1 = g.Zc(i,j+1)-g.Zc(i,j);

    double a = dim1*dip1 / (dim2*(dim2+dip1)*(dim2-dim1));
    double b = -dim2*dip1 / (dim1*(dim2-dim1)*(dim1+dip1));
    double c = dim2*dim1 / (dip1*(dim1+dip1)*(dim2+dip1));

    return a*Qty(i,j-2) + b*Qty(i,j-1) - (a+b+c)*Qty(i,j) + c*Qty(i,j+1);
}

__device__
double _vl_slope(double dQF, double dQB, double cF, double cB) {

    if (dQF*dQB > 0.) {
        double v = dQB/dQF ;
        return dQB * (cF*v + cB) / (v*v + (cF + cB - 2)*v + 1.) ;
    } 
    else {
        return 0. ;
    }
    
}

__device__
double vl_r2Dlog(GridRef& g, FieldConstRef<double>& Qty, int i, int j) {

    double rc = log(g.rc(i,j));

    double cF = (log(g.rc(i+1,j)) - rc) / (log(g.re(i+1,j))-rc) ;
    double cB = (log(g.rc(i-1,j)) - rc) / (log(g.re(i,j))-rc) ;

    double dQF = (Qty(i+1, j) - Qty(i, j)) / (log(g.rc(i+1,j)) - rc) ;
    double dQB = (Qty(i-1, j) - Qty(i, j)) / (log(g.rc(i-1,j)) - rc) ;

    return _vl_slope(dQF, dQB, cF, cB) / g.rc(i,j) ;
}

__device__
double vl_r2D(GridRef& g, FieldConstRef<double>& Qty, int i, int j) {

    double rc = (g.rc(i,j));

    double cF = ((g.rc(i+1,j)) - rc) / ((g.re(i+1,j))-rc) ;
    double cB = ((g.rc(i-1,j)) - rc) / ((g.re(i,j))-rc) ;

    double dQF = (Qty(i+1, j) - Qty(i, j)) / ((g.rc(i+1,j)) - rc) ;
    double dQB = (Qty(i-1, j) - Qty(i, j)) / ((g.rc(i-1,j)) - rc) ;

    return _vl_slope(dQF, dQB, cF, cB) ;
}

__device__
double vl_Z2D(GridRef& g, FieldConstRef<double>& Qty, int i, int j) {

    double Zc = g.Zc(i,j);

    double cF = (g.Zc(i,j+1) - Zc) / (g.Ze(i,j+1)-Zc) ;
    double cB = (g.Zc(i,j-1) - Zc) / (g.Ze(i,j)-Zc) ;

    double dQF = (Qty(i, j+1) - Qty(i, j)) / (g.Zc(i,j+1) - Zc) ;
    double dQB = (Qty(i, j-1) - Qty(i, j)) / (g.Zc(i,j-1) - Zc) ;

    return _vl_slope(dQF, dQB, cF, cB) ;
}

void _set_bounds(Grid& g, double* Sig_g, int bound, double floor) {

    if (bound & BoundaryFlags::open_R_inner) {
        // Sig_g[g.Nghost-1] = Sig_g[g.Nghost];
        Sig_g[g.Nghost-2] = Sig_g[g.Nghost-1];
    }
    else if (bound & BoundaryFlags::set_ext_R_inner) {} //set externally (e.g. inflow)
    else {
        Sig_g[g.Nghost-2] = floor;
    }


    if (bound & BoundaryFlags::open_R_outer) {
        // Sig_g[g.NR+g.Nghost] = Sig_g[g.NR+g.Nghost-1];
        Sig_g[g.NR+g.Nghost+1] = Sig_g[g.NR+g.Nghost];
    }
    else if (bound & BoundaryFlags::set_ext_R_inner) {} //set externally (e.g. inflow)
    else {
        Sig_g[g.NR+g.Nghost+1] = floor;
    }

}

__global__
void _set_v_bounds(GridRef g, FieldRef<Prims> wg, int bound, int coord, int buff) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) {
            if (i < g.Nghost+buff) {
                if (bound & BoundaryFlags::open_R_inner) {  //outflow
                    wg(i,j)[coord] = wg(g.Nghost+buff,j)[coord];
                }
                else if (bound & BoundaryFlags::set_ext_R_inner) {} //set externally (e.g. inflow)
                else {  // zero
                    wg(i,j)[coord] = 0.;
                }
            }

            if (j>=g.Nphi+g.Nghost-buff) {
                if (bound & BoundaryFlags::open_Z_outer) {
                    wg(i,j)[coord] = wg(i,g.Nphi+g.Nghost-buff-1)[coord];
                }
                else if (bound & BoundaryFlags::set_ext_Z_outer) {} //set externally (e.g. inflow)
                else { // Zero
                    wg(i,j)[coord] = 0.;
                }
            }        

            if (i>=g.NR+g.Nghost-buff) {
                if (bound & BoundaryFlags::open_R_outer) {
                    wg(i,j)[coord] = wg(g.NR+g.Nghost-buff-1,j)[coord];
                }
                else if (bound & BoundaryFlags::set_ext_R_outer) {} //set externally (e.g. inflow)
                else { // zero
                    wg(i,j)[coord] = 0.;
                }
            }    
            
            if (j < g.Nghost) {
                if (bound & BoundaryFlags::open_Z_inner) {  
                    wg(i,j)[coord] = wg(i,g.Nghost)[coord];
                }
                else if (bound & BoundaryFlags::set_ext_Z_inner) {} //set externally (e.g. inflow)
                else {  // reflecting
                    if (coord == 1) {
                        wg(i,j)[1] = wg(i,2*g.Nghost-1-j)[1] * (g.cos_th(j+1)*g.cos_th(j+1) - g.sin_th(j+1)*g.sin_th(j+1)) 
                                    + 2.*wg(i,2*g.Nghost-1-j)[3]*g.sin_th(j+1)*g.cos_th(j+1);
                    }
                    else if (coord == 2) {
                        wg(i,j)[2] = wg(i,2*g.Nghost-1-j)[2];
                    }
                    else if (coord == 3) {
                        wg(i,j)[3] = wg(i,2*g.Nghost-1-j)[3] * (-g.cos_th(j+1)*g.cos_th(j+1) + g.sin_th(j+1)*g.sin_th(j+1))
                                    + 2.*wg(i,2*g.Nghost-1-j)[1]*g.sin_th(j+1)*g.cos_th(j+1);
                    }
                }
            }    
        }
    }
}
__global__
void _set_vphi_bounds(GridRef g, FieldRef<double> v_phi, int bound) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) {
            if (i < g.Nghost) {
                if (bound & BoundaryFlags::open_R_inner) {  //outflow
                    v_phi(i,j) = v_phi(g.Nghost,j);
                }
                else if (bound & BoundaryFlags::set_ext_R_inner) {} //set externally (e.g. inflow)
                else {  // zero
                    v_phi(i,j) = 0.;
                }
            }

            if (j>=g.Nphi+g.Nghost) {
                if (bound & BoundaryFlags::open_Z_outer) {
                    v_phi(i,j) = v_phi(i,g.Nphi+g.Nghost-1);
                }
                else if (bound & BoundaryFlags::set_ext_Z_outer) {} //set externally (e.g. inflow)
                else { // Zero
                    v_phi(i,j) = 0.;
                }
            }        

            if (i>=g.NR+g.Nghost) {
                if (bound & BoundaryFlags::open_R_outer) {
                    v_phi(i,j) = v_phi(g.NR+g.Nghost-1,j);
                }
                else if (bound & BoundaryFlags::set_ext_R_outer) {} //set externally (e.g. inflow)
                else { // zero
                    v_phi(i,j) = 0.;
                }
            }    
            
            if (j < g.Nghost) {
                if (bound & BoundaryFlags::open_Z_inner) {  
                    v_phi(i,j) = v_phi(i,g.Nghost);
                }
                else if (bound & BoundaryFlags::set_ext_Z_inner) {} //set externally (e.g. inflow)
                else {  // reflecting
                    v_phi(i,j) = v_phi(i,2*g.Nghost-1-j);
                }
            }    
        }
    }
}

__global__
void _calc_Rflux(GridRef g, double* Sig_g, double* RF, const double* nu) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int istride = gridDim.x * blockDim.x ;

    for (int i=iidx+g.Nghost-1; i<g.NR+g.Nghost+2; i+=istride) {

        RF[i] = 3. * pow(g.Re(i), 0.5) * (pow(g.Rc(i),0.5) * nu[i] * Sig_g[i] - pow(g.Rc(i-1),0.5) * nu[i-1] * Sig_g[i-1]) / g.dRc(i-1);  

    }
}

__global__
void _update_Sig(GridRef g, double* Sig_g, double* RF, double dt, double floor) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int istride = gridDim.x * blockDim.x ;

    for (int i=iidx+g.Nghost-1; i<g.NR+g.Nghost+1; i+=istride) {

        double dV = 0.5 * (g.Re(i+1)*g.Re(i+1) - g.Re(i)*g.Re(i));

        Sig_g[i] += dt * (RF[i+1] - RF[i])/dV; 

        if (Sig_g[i] < 10.*floor) {Sig_g[i] = floor;}

    }
}

__global__
void _calc_f(GridRef g, double* Sig_g, double* f, double alpha, double GMstar, double Lstar) {

    double mu = 2.4;
    double T0 = std::pow(6.25e-3 * Lstar / (M_PI *au*au * sigma_SB), 0.25);
    double q = 0.5;

    double T, Om_k, c_s2, Sig_g_e;

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int istride = gridDim.x * blockDim.x ;

    for (int i=iidx; i<g.NR+2*g.Nghost+1; i+=istride) {
        
        T = T0 * pow(g.Re(i)/au, -q);
        Om_k = sqrt(GMstar/(g.Re(i)*g.Re(i)*g.Re(i)));
        c_s2 = k_B*T/(mu*m_H);

        if (i==0) {
            f[i] = sqrt(g.Re(i)) * Sig_g[i] * alpha * c_s2 / Om_k ;
        }

        else if (i==g.NR+2*g.Nghost) {
            f[i] = sqrt(g.Re(i)) * Sig_g[i-1] * alpha * c_s2 / Om_k ;
        }

        else {
            Sig_g_e = Sig_g[i] - (g.Rc(i)-g.Re(i))*(Sig_g[i]-Sig_g[i-1]) / g.dRc(i-1);

            f[i] = sqrt(g.Re(i)) * Sig_g_e * alpha * c_s2 / Om_k ;
        }
    
    }
}

__global__
void _calc_u_gas(GridRef g, double* Sig_g, double* f, double* u_gas) {


    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int istride = gridDim.x * blockDim.x ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {

        if (Sig_g[i] < 10.e-3) { 
            u_gas[i] = -100; 
            continue;
        }

        u_gas[i] = -3./(Sig_g[i] * sqrt(g.Rc(i))) * (f[i+1]-f[i])/g.dRe(i);

        if (u_gas[i] < -100) {
            u_gas[i] = -100;
        }
    
    }
}

__global__
void _calc_dt_diff(GridRef g, double* dt, double* nu) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int istride = gridDim.x * blockDim.x ;

    for (int i=iidx; i<g.NR+2*g.Nghost-1; i+=istride) {
        double dV = 0.5 * (g.Re(i+1)*g.Re(i+1) - g.Re(i)*g.Re(i));

        dt[i] = g.dRc(i)*dV / (6. * pow(g.Re(i+1),0.5) * pow(g.Rc(i),0.5) * nu[i]);
    }
}

__global__
void _calc_dt_diff(GridRef g, double* dt, FieldConstRef<double> nu) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int istride = gridDim.x * blockDim.x ;

    for (int i=iidx; i<g.NR+2*g.Nghost-1; i+=istride) {
        double dV = 0.5 * (g.Re(i+1)*g.Re(i+1) - g.Re(i)*g.Re(i));

        dt[i] = g.dRc(i)*dV / (6. * pow(g.Re(i+1),0.5) * pow(g.Rc(i),0.5) * nu(i,2));
    }
}

__global__ void _Zaverage_nu(GridRef g, double* nu1D, FieldConstRef<double> nu) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int istride = gridDim.x * blockDim.x ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        // for (int j=g.Nghost; j<g.Nphi+g.Nghost-1; j++) {
        //     nu1D[i] += 0.5*(nu(i,j) + nu(i,j+1)) * g.dZc(i,j);
        // }
        // nu1D[i] /= g.Zc(i,g.Nphi+g.Nghost-1) - g.Zc(i,g.Nghost);
        nu1D[i] = nu(i,2);
    }
}

void update_gas_sigma(Grid& g, CudaArray<double>& Sig_g, double dt, const CudaArray<double>& nu, int bound, double floor) {

    CudaArray<double> RF = make_CudaArray<double>(g.NR+2*g.Nghost);

    size_t threads = 256 ;
    size_t blocks = (g.NR + 2*g.Nghost+255)/256 ;

    _set_bounds(g, Sig_g.get(), bound, floor);
    _calc_Rflux<<<blocks, threads>>>(g, Sig_g.get(), RF.get(), nu.get());
    _update_Sig<<<blocks, threads>>>(g, Sig_g.get(), RF.get(), dt, floor);
}

void update_gas_sigma(Grid& g, CudaArray<double>& Sig_g, double dt, const Field<double>& nu, int bound, double floor) {

    CudaArray<double> RF = make_CudaArray<double>(g.NR+2*g.Nghost);
    CudaArray<double> nu1D = make_CudaArray<double>(g.NR+2*g.Nghost);

    size_t threads = 256 ;
    size_t blocks = (g.NR + 2*g.Nghost+255)/256 ;

    _set_bounds(g, Sig_g.get(), bound, floor);
    _Zaverage_nu<<<blocks, threads>>>(g, nu1D.get(), nu);
    _calc_Rflux<<<blocks, threads>>>(g, Sig_g.get(), RF.get(), nu1D.get());
    _update_Sig<<<blocks, threads>>>(g, Sig_g.get(), RF.get(), dt, floor);
}

void update_gas_vel(Grid& g, CudaArray<double>& Sig_g, CudaArray<double>& u_gas, double alpha, Star& star) {

    // https://www.aanda.org/articles/aa/pdf/2013/03/aa20812-12.pdf

    CudaArray<double> f = make_CudaArray<double>(g.NR+2*g.Nghost+1);

    size_t threads = 256 ;
    size_t blocks = (g.NR + 2*g.Nghost+255)/256 ;

    _calc_f<<<blocks, threads>>>(g, Sig_g.get(), f.get(), alpha, star.GM, star.L);
    _calc_u_gas<<<blocks, threads>>>(g, Sig_g.get(), f.get(), u_gas.get());

}

__global__ void _update_source(GridRef g, double* Sig_g, double* Sigdot, double dt, double floor) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int istride = gridDim.x * blockDim.x ;

    for (int i=iidx; i<g.NR+2*g.Nghost-1; i+=istride) {

        Sig_g[i] -= dt*Sigdot[i];
        if (Sig_g[i] < 1.1*floor) { Sig_g[i] = floor; }
    }
}

void update_gas_sources(Grid& g, CudaArray<double>& Sig_g, CudaArray<double>& Sigdot, double dt, int bound, double floor) {

    size_t threads = 256 ;
    size_t blocks = (g.NR + 2*g.Nghost+255)/256 ;

    _set_bounds(g, Sig_g.get(), bound, floor);

    _update_source<<<blocks, threads>>>(g, Sig_g.get(), Sigdot.get(), dt, floor);

}

double calc_dt(Grid& g, const CudaArray<double>& nu) {

    CudaArray<double> dt_R = make_CudaArray<double>(g.NR+2*g.Nghost-1);
    double dt = 1.e308;

    size_t threads = 256 ;
    size_t blocks = (g.NR + 2*g.Nghost+255)/256 ;

    _calc_dt_diff<<<blocks, threads>>>(g, dt_R.get(), nu.get());
    check_CUDA_errors("_calc_dt_diff");

    for (int i=0; i<g.NR + 2*g.Nghost-1; i++) {
        dt = std::min(dt, dt_R[i]);
    }

    return dt;
}

double calc_dt(Grid& g, const Field<double>& nu) {

    CudaArray<double> dt_R = make_CudaArray<double>(g.NR+2*g.Nghost-1);
    double dt = 1.e308;

    size_t threads = 256 ;
    size_t blocks = (g.NR + 2*g.Nghost+255)/256 ;

    _calc_dt_diff<<<blocks, threads>>>(g, dt_R.get(), nu);
    check_CUDA_errors("_calc_dt_diff");

    for (int i=0; i<g.NR + 2*g.Nghost-1; i++) {
        dt = std::min(dt, dt_R[i]);
    }

    return dt;
}

// __device__ void dydx(GridRef& g, )

__global__ void _set_vZ(GridRef g, FieldRef<Prims> wg, double* Sig_dot_w, double floor, double cav) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost; j+=jstride) {

            if (wg(i,j).rho < 10.*floor) {
                wg(i,j).v_Z = 0.;
                continue;
            }
            if (g.Rc(i) < cav) {
                wg(i,j).v_Z = 0.;
                continue;
            }

            wg(i,j).v_Z = Sig_dot_w[i] / wg(i,j).rho ;

            if (wg(i,j).v_Z > 2.e5) { wg(i,j).v_Z = 2.e5; }
        }
    }
}

__global__ void _calc_vphi(GridRef g, FieldConstRef<double> p, FieldRef<Prims> wg, FieldRef<double> vphig,double GMstar, double floor, int nbuffer, double cav) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx+g.Nghost+nbuffer; i<g.NR+g.Nghost-nbuffer; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost-nbuffer; j+=jstride) {

            double dpdZ = exp(p(i,j))*vl_Z2D(g, p, i ,j);

            double dpdr = exp(p(i,j))*vl_r2D(g, p, i, j);

            if ((GMstar*g.cos_th_c(j)*g.cos_th_c(j)/g.rc(i,j) 
                                        + (g.rc(i,j)/wg(i,j).rho) * (dpdr - g.sin_th_c(j) * dpdZ))< 0.) {
                wg(i,j).v_phi = vphig(i,j) = 0.;
            }
            else {
                wg(i,j).v_phi = sqrt(GMstar*g.cos_th_c(j)*g.cos_th_c(j)/g.rc(i,j) 
                                        + (g.rc(i,j)/wg(i,j).rho) * (dpdr - g.sin_th_c(j) * dpdZ));
                vphig(i,j) = wg(i,j).v_phi;
            }
        }
    }
}

__global__ void _calc_vphi(GridRef g, FieldConstRef<double> p, FieldRef<double> rho, FieldRef<double> vphig,double GMstar, double floor, int nbuffer, double cav) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx+g.Nghost+nbuffer; i<g.NR+g.Nghost-nbuffer; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost-nbuffer; j+=jstride) {

            if (rho(i,j) < 10.*floor) {
                vphig(i,j) = sqrt(GMstar / g.Rc(i));
                continue;
            }
            if (g.Rc(i) < cav) {
                vphig(i,j) = sqrt(GMstar / g.Rc(i));
                continue;
            }

            double dpdZ = exp(p(i,j))*vl_Z2D(g, p, i ,j);

            double dpdr = exp(p(i,j))*slope4Or(g, p, i, j);

            if (sqrt(GMstar*g.cos_th_c(j)*g.cos_th_c(j)/g.rc(i,j) 
                                    + (g.rc(i,j)/rho(i,j)) * (dpdr - g.sin_th_c(j) * dpdZ))< 0.) {
                vphig(i,j) = 0.;
            }
            else {
                vphig(i,j) = sqrt(GMstar*g.cos_th_c(j)*g.cos_th_c(j)/g.rc(i,j) 
                                    + (g.rc(i,j)/rho(i,j)) * (dpdr - g.sin_th_c(j) * dpdZ));
            }

        }
    }
}

__global__ void _calc_T(GridRef g, FieldRef<double> Trphi, FieldRef<double> TZphi, FieldConstRef<double> vphig,
                            FieldRef<double> dvphidr, FieldRef<double> dvphidZ, FieldRef<Prims> wg, double* nu, int nbuffer) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx+g.Nghost+nbuffer; i<g.NR+g.Nghost-nbuffer; i+=istride) {
        for (int j=jidx+g.Nghost-1; j<g.Nphi+g.Nghost-nbuffer; j+=jstride) {

            dvphidZ(i,j) = vl_Z2D(g, vphig, i, j);

            dvphidr(i,j) = slope4Or(g, vphig, i, j); 

            double C = wg(i,j).rho*nu[i]/(g.cos_th_c(j)*g.cos_th_c(j));

            Trphi(i,j) = C * (dvphidr(i,j) - vphig(i,j)/g.rc(i,j) - g.sin_th_c(j)*dvphidZ(i,j));
            TZphi(i,j) = C * (dvphidZ(i,j) - g.sin_th_c(j)*(dvphidr(i,j) - vphig(i,j)/g.rc(i,j)));
        }
    }
}

__global__ void _calc_T(GridRef g, FieldRef<double> Trphi, FieldRef<double> TZphi, FieldConstRef<double> vphig,
                            FieldRef<double> dvphidr, FieldRef<double> dvphidZ, FieldRef<Prims> wg, FieldRef<double> rho, double* nu, int nbuffer) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx+g.Nghost+nbuffer; i<g.NR+g.Nghost-nbuffer; i+=istride) {
        for (int j=jidx+g.Nghost-1; j<g.Nphi+g.Nghost-nbuffer; j+=jstride) {

            dvphidZ(i,j) = vl_Z2D(g, vphig, i, j);

            dvphidr(i,j) = slope4Or(g, vphig, i, j); 

            double C = rho(i,j)*nu[i]/(g.cos_th_c(j)*g.cos_th_c(j));

            Trphi(i,j) = C * (dvphidr(i,j) - vphig(i,j)/g.rc(i,j) - g.sin_th_c(j)*dvphidZ(i,j));
            TZphi(i,j) = C * (dvphidZ(i,j) - g.sin_th_c(j)*(dvphidr(i,j) - vphig(i,j)/g.rc(i,j)));
        }
    }
}

__global__ void _calc_vr(GridRef g, FieldConstRef<double> Trphi, FieldConstRef<double> TZphi, FieldRef<double> vphig,
                            FieldRef<double> dvphidr, FieldRef<double> dvphidZ, FieldRef<Prims> wg, FieldRef<double> rho, double floor, int nbuffer, double cav) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx+g.Nghost+nbuffer; i<g.NR+g.Nghost-nbuffer; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost-nbuffer; j+=jstride) {

            if (wg(i,j).rho < 50.*floor) {
                wg(i,j).v_R = 0.;
                continue;
            }
            if (g.Rc(i) < cav) {
                wg(i,j).v_R = 0.;
                continue;
            }
            
            double dTZphidZ = vl_Z2D(g, TZphi, i, j);

            double dTrphidr = slope4Or(g, Trphi, i, j);

            double divT_phi = dTrphidr + 3.*Trphi(i,j)/g.rc(i,j) + dTZphidZ;

            if ((g.rc(i,j)*dvphidr(i,j) + vphig(i,j) - g.Zc(i,j)*dvphidZ(i,j)) != 0) {
                double vr = g.rc(i,j)/(g.rc(i,j)*dvphidr(i,j) + vphig(i,j) - g.Zc(i,j)*dvphidZ(i,j)) * (1./rho(i,j) * divT_phi - wg(i,j).v_Z*dvphidZ(i,j));

                wg(i,j).v_R = vr*g.cos_th_c(j);  

                if (wg(i,j).v_R > 1000.) {wg(i,j).v_R = 1000.;}
                if (wg(i,j).v_R < -1000.) {wg(i,j).v_R = -1000.;}
            }
            else {
                wg(i,j).v_R = 0.;     
            }
        }
    }
}

__global__ void _calc_vr(GridRef g, FieldConstRef<double> Trphi, FieldConstRef<double> TZphi, FieldRef<double> vphig,
                            FieldRef<double> dvphidr, FieldRef<double> dvphidZ, FieldRef<Prims> wg, double floor, int nbuffer, double cav) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx+g.Nghost+nbuffer; i<g.NR+g.Nghost-nbuffer; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost-nbuffer; j+=jstride) {

            if (wg(i,j).rho < 50.*floor) {
                wg(i,j).v_R = 0.;
                continue;
            }
            if (g.Rc(i) < cav) {
                wg(i,j).v_R = 0.;
                continue;
            }
            
            double dTZphidZ = vl_Z2D(g, TZphi, i, j);

            double dTrphidr = slope4Or(g, Trphi, i, j);

            double divT_phi = dTrphidr + 3.*Trphi(i,j)/g.rc(i,j) + dTZphidZ;

            if ((g.rc(i,j)*dvphidr(i,j) + vphig(i,j) - g.Zc(i,j)*dvphidZ(i,j)) != 0) {
                double vr = g.rc(i,j)/(g.rc(i,j)*dvphidr(i,j) + vphig(i,j) - g.Zc(i,j)*dvphidZ(i,j)) * (1./wg(i,j).rho * divT_phi - wg(i,j).v_Z*dvphidZ(i,j));

                wg(i,j).v_R = vr*g.cos_th_c(j);  

                if (wg(i,j).v_R > 1000.) {wg(i,j).v_R = 1000.;}
                if (wg(i,j).v_R < -1000.) {wg(i,j).v_R = -1000.;}
            }
            else {
                wg(i,j).v_R = 0.;     
            }
        }
    }
}

__global__
void _calc_p(GridRef g, FieldRef<Prims> wg, FieldRef<double> cs2, FieldRef<double> p) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) {

            p(i,j) = log(wg(i,j).rho*cs2(i,j));

        }
    }
}

__global__
void _calc_p(GridRef g, FieldRef<double> rho, double* nu, double alpha, double GMstar, FieldRef<double> p) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) {

            double Om = sqrt(GMstar / (g.Rc(i)*g.Rc(i)*g.Rc(i)));
            p(i,j) = log(rho(i,j)*Om*nu[i] / alpha);
        }
    }
}

__global__
void _calc_rho(GridRef g, double* Sig_g, FieldRef<double> cs2, double GMstar, double Lstar, FieldRef<double> rho) {
    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) {

            double Om = sqrt(GMstar / (g.Rc(i)*g.Rc(i)*g.Rc(i)));
            double H = sqrt(k_B*std::pow(6.25e-3 * Lstar / (M_PI *au*au * sigma_SB), 0.25)*pow(g.Rc(i)/au, -0.5)/(2.4*m_H))/Om;
            rho(i,j) = Sig_g[i]/(sqrt(2.*M_PI)*H) * exp(-g.Zc(i,j)*g.Zc(i,j)/(2.*H*H));

        }
    }
}

__global__ 
void _correct_vr_cav(GridRef g, FieldRef<Prims> Ws_g, double cav) {
    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;
    
    int cavind;
    for (int i=0; i<g.NR+2*g.Nghost; i++) { 
        if (g.Rc(i) > 1.1*cav) {
            cavind = i;
            break;
        }
    }

    for (int i=iidx; i<cavind; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) {

            Ws_g(i,j).v_R = Ws_g(cavind,j).v_R;

        }
    }
}


void calc_gas_velocities(Grid& g, CudaArray<double>& Sig_g, Field<Prims>& wg, Field<double>& cs2, CudaArray<double>& nu, double alpha, Star& star, int bound, double floor, double cav) {

    Field<double> Trphi = create_field<double>(g);
    Field<double> TZphi = create_field<double>(g);
    Field<double> drvphidr = create_field<double>(g);
    Field<double> dvphidZ = create_field<double>(g);
    Field<double> p = create_field<double>(g);
    Field<double> rho = create_field<double>(g);
    Field<double> vphig = create_field<double>(g);

    dim3 threads(16,16) ;
    dim3 blocks((g.NR + 2*g.Nghost+15)/16,(g.Nphi + 2*g.Nghost+15)/16) ;
    int buff = 0;
    int vrbuff = 10;

    // Calc v_phi from true profile

    _calc_p<<<blocks,threads>>>(g, wg, cs2, p);
    _calc_vphi<<<blocks,threads>>>(g, p, wg, vphig, star.GM, floor, buff, cav);
    _set_v_bounds<<<blocks,threads>>>(g, wg, bound, 2, buff);    

    // Calc v_R from parametrised profiles

    _calc_rho<<<blocks,threads>>>(g, Sig_g.get(), cs2, star.GM, star.L, rho);
    _calc_p<<<blocks,threads>>>(g, rho, nu.get(), alpha, star.GM, p);
    _calc_vphi<<<blocks,threads>>>(g, p, rho, vphig, star.GM, floor, buff, cav);
    _set_vphi_bounds<<<blocks,threads>>>(g, vphig, bound);        
    _calc_T<<<blocks,threads>>>(g, Trphi, TZphi, vphig, drvphidr, dvphidZ, wg, rho, nu.get(), 2);
    _calc_vr<<<blocks,threads>>>(g, Trphi, TZphi, vphig, drvphidr, dvphidZ, wg, rho, floor, vrbuff, cav);

    _correct_vr_cav<<<blocks,threads>>>(g, wg, cav);
    _set_v_bounds<<<blocks,threads>>>(g, wg, bound, 1, vrbuff);
}

void calc_gas_velocities_full(Grid& g, CudaArray<double>& Sig_g, Field<Prims>& wg, Field<double>& cs2, CudaArray<double>& nu, double alpha, Star& star, int bound, double floor, double cav) {

    Field<double> Trphi = create_field<double>(g);
    Field<double> TZphi = create_field<double>(g);
    Field<double> drvphidr = create_field<double>(g);
    Field<double> dvphidZ = create_field<double>(g);
    Field<double> p = create_field<double>(g);
    Field<double> rho = create_field<double>(g);
    Field<double> vphig = create_field<double>(g);

    dim3 threads(16,16) ;
    dim3 blocks((g.NR + 2*g.Nghost+15)/16,(g.Nphi + 2*g.Nghost+15)/16) ;
    int buff = 0;
    int vrbuff = 10;

    // Calc v_phi from true profile

    _calc_p<<<blocks,threads>>>(g, wg, cs2, p);
    _calc_vphi<<<blocks,threads>>>(g, p, wg, vphig, star.GM, floor, buff, cav);
    _set_v_bounds<<<blocks,threads>>>(g, wg, bound, 2, buff);    

    _set_vphi_bounds<<<blocks,threads>>>(g, vphig, bound);     
    _calc_T<<<blocks,threads>>>(g, Trphi, TZphi, vphig, drvphidr, dvphidZ, wg, nu.get(), 0);
    _calc_vr<<<blocks,threads>>>(g, Trphi, TZphi, vphig, drvphidr, dvphidZ, wg, floor, 2, cav);
    _set_v_bounds<<<blocks,threads>>>(g, wg, bound, 1, vrbuff);
}

void calc_gas_velocities_wind(Grid& g, Field<Prims>& wg, CudaArray<double>& Sig_g, Field<double>& cs2, CudaArray<double>& nu, CudaArray<double>& Sig_dot_w, 
                                double alpha, Star& star, int bound, double floor, double cav) {

    Field<double> Trphi = create_field<double>(g);
    Field<double> TZphi = create_field<double>(g);
    Field<double> drvphidr = create_field<double>(g);
    Field<double> dvphidZ = create_field<double>(g);
    Field<double> p = create_field<double>(g);
    Field<double> rho = create_field<double>(g);
    Field<double> vphig = create_field<double>(g);

    dim3 threads(16,16) ;
    dim3 blocks((g.NR + 2*g.Nghost+15)/16,(g.Nphi + 2*g.Nghost+15)/16) ;
    int buff = 0;
    int vrbuff = 10;

    _set_vZ<<<blocks,threads>>>(g, wg, Sig_dot_w.get(), floor, cav);
    _set_v_bounds<<<blocks,threads>>>(g, wg, bound, 3, 0);
    
    // Calc v_phi from true profile

    _calc_p<<<blocks,threads>>>(g, wg, cs2, p);
    _calc_vphi<<<blocks,threads>>>(g, p, wg, vphig, star.GM, floor, buff, cav);
    _set_v_bounds<<<blocks,threads>>>(g, wg, bound, 2, buff);    

    // Calc v_R from parametrised profiles

    _calc_rho<<<blocks,threads>>>(g, Sig_g.get(), cs2, star.GM, star.L, rho);
    _calc_p<<<blocks,threads>>>(g, rho, nu.get(), alpha, star.GM, p);
    _calc_vphi<<<blocks,threads>>>(g, p, rho, vphig, star.GM, floor, buff, cav);
    _set_vphi_bounds<<<blocks,threads>>>(g, vphig, bound);        
    _calc_T<<<blocks,threads>>>(g, Trphi, TZphi, vphig, drvphidr, dvphidZ, wg, rho, nu.get(), 2);
    _calc_vr<<<blocks,threads>>>(g, Trphi, TZphi, vphig, drvphidr, dvphidZ, wg, rho, floor, vrbuff, cav);
    _correct_vr_cav<<<blocks,threads>>>(g, wg, cav);
    _set_v_bounds<<<blocks,threads>>>(g, wg, bound, 1, vrbuff);
}

// 2-population dust model Birnstiel et al. 2012 https://www.aanda.org/articles/aa/pdf/2012/03/aa18136-11.pdf

__global__
void _calc_P(GridRef g, double* sig_g, double* P, double GMstar, double Lstar) {
    
    double mu = 2.4;
    double T0 = std::pow(6.25e-3 * Lstar / (M_PI *au*au * sigma_SB), 0.25);
    double q = 0.5;

    double T, v_k, c_s, Sig_g_e;

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int istride = gridDim.x * blockDim.x ;

    for (int i=iidx; i<g.NR+2*g.Nghost+1; i+=istride) {
        
        T = T0 * pow(g.Re(i)/au, -q);
        v_k = sqrt(GMstar/g.Re(i));
        c_s = sqrt(k_B*T/(mu*m_H));

        if (i==0) {
            P[i] = sig_g[i] * c_s * v_k / (g.Re(i) * sqrt(2.*M_PI));
        }

        else if (i==g.NR+2*g.Nghost) {
            P[i] = sig_g[i-1] * c_s * v_k / (g.Re(i) * sqrt(2.*M_PI));
        }

        else {
            Sig_g_e = sig_g[i] - (g.Rc(i)-g.Re(i))*(sig_g[i]-sig_g[i-1]) / g.dRc(i-1);

            P[i] = Sig_g_e * c_s * v_k / (g.Re(i) * sqrt(2.*M_PI));
        }
    
    }
}

__global__
void _calc_ubar(GridRef g, double* ubar, double* sig, double* sig_g, double* u_gas, double* P,
                    double u_f, double rho_s, double alpha, double t, double a0, int buff, double GMstar, double Lstar) {

    double f_f = 0.37;
    double f_d = 0.55;
    double f_grow = 1.;
    // double p = 1.;
    double q = 0.5;
    // double gamma = -p - q/2. - 3./2.;
    double mu = 2.4;
    double T0 = std::pow(6.25e-3 * Lstar / (M_PI *au*au * sigma_SB), 0.25);
    double N = 0.5;
    double t0 = 0.;

    double T, v_k, c_s, a_frag, a_drift, St_df, a_df, a_min, a1, tau_grow, u0, u1, f_m, u_drift, St0, St1, dlnPdlnR;

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int istride = gridDim.x * blockDim.x ;

    for (int i=iidx+buff; i<g.NR+2*g.Nghost-buff; i+=istride) {

        // if (sig[i] < 10.e-15) { 
        //     ubar[i] = 0.;
        //     continue; 
        // } 

        dlnPdlnR = (log(P[i+1]) - log(P[i])) / (log(g.Re(i+1)) - log(g.Re(i))); 

        // if (i>250 && i<280){
        //     printf("%d %g\n", i, dlnPdlnR);
        // }

        T = T0 * pow(g.Rc(i)/au, -q);
        v_k = sqrt(GMstar/g.Rc(i));
        c_s = sqrt(k_B*T/(mu*m_H));

        a_frag = f_f * (2./(3.*M_PI)) * (sig_g[i]/(rho_s*alpha)) * (u_f*u_f)/(c_s*c_s);
        a_drift = f_d * (2*sig[i]/(M_PI*rho_s)) * (v_k*v_k/(c_s*c_s)) * (1./abs(dlnPdlnR));
        
        St_df = (u_f * v_k)/(abs(dlnPdlnR)*c_s*c_s*(1-N)); 
        a_df = St_df * (2.*sig_g[i]/(M_PI*rho_s));

        a_min = min(min(a_drift, a_frag), a_df);
        // if (a_min == a_frag && g.Rc(i)/au<5 ) {
        //     printf("%g\n", g.Rc(i)/au);
        // }
        tau_grow = f_grow * sig_g[i]/(sig[i]*(v_k/g.Rc(i)));
        a1 = min(a_min, a0*exp((t-t0)/tau_grow));

        u_drift = min(c_s*c_s / (2.*v_k) * dlnPdlnR, c_s);

        St0 = (a0*rho_s*M_PI)/(sig_g[i]*2.);
        St1 = (a1*rho_s*M_PI)/(sig_g[i]*2.);


        u0 = u_gas[i]/(1.+St0*St0) + (2.*u_drift)/(St0 + 1./St0);
        u1 = u_gas[i]/(1.+St1*St1) + (2.*u_drift)/(St1 + 1./St1);

        // if (i>240 && i<260){
        //     printf("%d %g %g %g %g %g %g\n", i, u0,u1,St0,St1,a_drift,a_frag);
        // }

        if (a_min == a_drift) {f_m = 0.97;}
        else {f_m = 0.75;}

        ubar[i] = (1 - f_m)*u0 + f_m*u1;
    }
}

__device__
double vl(GridRef& g, double* Qty, int i) {

    double Rc = g.Rc(i);

    double cF = (g.Rc(i+1) - Rc) / (g.Re(i+1)-Rc) ;
    double cB = (g.Rc(i-1) - Rc) / (g.Re(i)-Rc) ;

    double dQF = (Qty[i+1] - Qty[i]) / (g.Rc(i+1) - Rc) ;
    double dQB = (Qty[i-1] - Qty[i]) / (g.Rc(i-1) - Rc) ;

    return _vl_slope(dQF, dQB, cF, cB) ;
}

__device__
double compute_diff_flux(GridRef& g, double* sig, double* sig_g, double* D, int i) {

    if (sig_g[i] < 1.01e-10) {
        return 0 ;
    }

    double dRc = g.dRc(i-1);
    double dRe = g.Re(i)-g.Rc(i-1);

    double D_e = D[i-1] + dRe*(D[i]-D[i-1]) / dRc;
    double Sig_g_e = sig_g[i-1] + dRe*(sig_g[i]-sig_g[i-1]) / dRc;
    double ddtg = (sig[i]/sig_g[i] - sig[i-1]/sig_g[i-1]) / dRc;

    // printf("diff ddtg %d = %g \n", i,ddtg);


    return - D_e * Sig_g_e * ddtg;
}

__device__
void construct_diff_fluxes(GridRef& g, double v_l, double v_r, double v_av, 
                  double sig_l, double sig_r, double* flux, double diff_flux, int i) {

    if (v_l < 0 && v_r > 0) {
        flux[i] = diff_flux; 
    }
    
    if (v_av > 0.) {  
        flux[i] = sig_l*v_l + diff_flux;
    }

    if (v_av < 0.) {   
        flux[i] = sig_r*v_r + diff_flux;
    }

    if (v_av == 0.) {
        flux[i] = 0.5*(sig_l*v_l + sig_r*v_r + 2*diff_flux);
    }
}

__device__
void dust_diff_flux(GridRef& g, double* sig, double* ubar, int i, double* flux, double diff_flux) {

    double sig_l = sig[i-1];
    double sig_r = sig[i];

    double v_l = ubar[i-1];
    double v_r = ubar[i];
    double rhorat, v_av;

    rhorat = std::pow(sig_r, 0.5)/std::pow(sig_l, 0.5);
    v_av = (v_l + rhorat * v_r) / (1. + rhorat);

    // Construct fluxes depending on sign of interface velocities

    construct_diff_fluxes(g, v_l, v_r, v_av, sig_l, sig_r, flux, diff_flux, i);
}

__device__
void dust_diff_flux_vl(GridRef& g, double* sig, double* ubar, int i, double* flux, double diff_flux) {

    double dR_l = g.Re(i)-g.Rc(i-1);
    double dR_r = g.Re(i)-g.Rc(i);

    double sig_l = sig[i-1] + vl(g,sig,i-1)*dR_l;
    double sig_r = sig[i] + vl(g,sig,i)*dR_r;

    double v_l = ubar[i-1] + vl(g,ubar,i-1)*dR_l;
    double v_r = ubar[i] + vl(g,ubar,i)*dR_r;

    double rhorat, v_av;

    rhorat = std::pow(sig_r, 0.5)/std::pow(sig_l, 0.5);
    v_av = (v_l + rhorat * v_r) / (1. + rhorat);

    // Construct fluxes depending on sign of interface velocities

    construct_diff_fluxes(g, v_l, v_r, v_av, sig_l, sig_r, flux, diff_flux, i);
}

__global__ void _calc_diff_flux(GridRef g, double* sig, double* sig_g, double* ubar, double* flux, double* D) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int istride = gridDim.x * blockDim.x ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost+1; i+=istride) { 

        double diff_flux = compute_diff_flux(g, sig, sig_g, D, i);

        //printf("diff flux %d i = %g \n", i,diff_flux);

        dust_diff_flux(g, sig, ubar, i, flux, diff_flux);
    }
}

__global__ void _calc_diff_flux_vl(GridRef g, double* sig, double* sig_g, double* ubar, double* flux, double* D) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int istride = gridDim.x * blockDim.x ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost+1; i+=istride) { 

        double diff_flux = compute_diff_flux(g, sig, sig_g, D, i);

        dust_diff_flux_vl(g, sig, ubar, i, flux, diff_flux);

        // if (i<30) {
        //     printf("%d %g %g\n", i, diff_flux, flux[i]);
        // }
    }
}


void _set_bounds_d(Grid& g, double* Sig_d, int bound, double floor) {

    if (bound & BoundaryFlags::open_R_inner) {
        Sig_d[g.Nghost-1] = Sig_d[g.Nghost];
        Sig_d[g.Nghost-2] = Sig_d[g.Nghost];
    }
    else if (bound & BoundaryFlags::set_ext_R_inner) {} //set externally (e.g. inflow)
    else {
        Sig_d[g.Nghost-1] = floor;
        Sig_d[g.Nghost-2] = floor;
    }


    if (bound & BoundaryFlags::open_R_outer) {
        Sig_d[g.NR+g.Nghost] = Sig_d[g.NR+g.Nghost-1];
        Sig_d[g.NR+g.Nghost+1] = Sig_d[g.NR+g.Nghost-1];
    }
    else if (bound & BoundaryFlags::set_ext_R_inner) {} //set externally (e.g. inflow)
    else {
        Sig_d[g.NR+g.Nghost] = floor;
        Sig_d[g.NR+g.Nghost+1] = floor;
    }

}

__global__ void _update_mid_quants(GridRef g, double* sig_mid, double* sig, double dt, double* flux) {
    
    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int istride = gridDim.x * blockDim.x ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {

        double dRf = g.Re(i) * flux[i] - g.Re(i+1) * flux[i+1];
        double dV = 0.5 * (g.Re(i+1)*g.Re(i+1) - g.Re(i)*g.Re(i));

        sig_mid[i] = sig[i] + (0.5*dt/dV)*dRf;
    }

}

__global__ void _update_quants(GridRef g, double* sig, double dt, double* flux) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int istride = gridDim.x * blockDim.x ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {

        double dRf = g.Re(i) * flux[i] - g.Re(i+1) * flux[i+1];
        double dV = 0.5 * (g.Re(i+1)*g.Re(i+1) - g.Re(i)*g.Re(i));

        sig[i] = sig[i] + (dt/dV)*dRf;
        
        if (sig[i] < 10.e-15) { sig[i] = 1.e-15; } 
    }
}

void _set_ubar_bounds(Grid& g, CudaArray<double>& ubar, int buff) {

    for (int i=0; i<buff; i++) {
        ubar[i] = ubar[buff];
    }

    for (int i=g.NR+2*g.Nghost-buff; i<g.NR+2*g.Nghost; i++) {
        ubar[i] = ubar[g.NR+2*g.Nghost-buff-1];
    }
}


void calculate_ubar(Grid& g, CudaArray<double>& sig, CudaArray<double>& sig_g, 
                    CudaArray<double>& ubar, CudaArray<double>& u_gas,
                    double t, double u_f, double rho_s, double alpha, double a0, Star& star, int bound,int boundg) {

    size_t threads = 64 ;
    size_t blocks = (g.NR + 2*g.Nghost+63)/64;
    int buff = 3;

    CudaArray<double> P = make_CudaArray<double>(g.NR+2*g.Nghost+1);
    
    _calc_P<<<blocks,threads>>>(g, sig_g.get(), P.get(), star.GM, star.L);
    _calc_ubar<<<blocks, threads>>>(g, ubar.get(), sig.get(), sig_g.get(), u_gas.get(), P.get(), u_f, rho_s, alpha, t, a0, buff, star.GM, star.L);
    cudaDeviceSynchronize();
    _set_ubar_bounds(g,ubar,buff);
}

void update_dust_sigma(Grid& g, CudaArray<double>& sig, CudaArray<double>& sig_g, 
                    CudaArray<double>& ubar, CudaArray<double>& D, double dt, int bound) {
    size_t threads = 64;
    size_t blocks = (g.NR + 2*g.Nghost+63)/64 ;
    
    // calculate advection-diffusion

    CudaArray<double> sig_mid = make_CudaArray<double>(g.NR+2*g.Nghost);
    CudaArray<double> flux = make_CudaArray<double>(g.NR+2*g.Nghost);

    _set_bounds_d(g, sig.get(), bound, 1.e-15);

    // Calc donor cell flux

    _calc_diff_flux<<<blocks,threads>>>(g, sig.get(), sig_g.get(), ubar.get(), flux.get(), D.get());

    // Update quantities a half time step

    _update_mid_quants<<<blocks,threads>>>(g, sig_mid.get(), sig.get(), dt, flux.get());

    _set_bounds_d(g, sig_mid.get(), bound, 1.e-15);

    // Compute fluxes with Van Leer

    _calc_diff_flux_vl<<<blocks,threads>>>(g, sig_mid.get(), sig_g.get(), ubar.get(), flux.get(), D.get());

    _update_quants<<<blocks,threads>>>(g, sig.get(), dt, flux.get());
}

double compute_CFL(Grid& g, CudaArray<double>& ubar, CudaArray<double>& D,
                        double CFL_adv, double CFL_diff) {
    double dt = 1e308;
    for (int i=0; i<g.NR+2*g.Nghost; i++) {

        double dtR = abs(g.dRe(i) / ubar[i]);

        double dtR_diff = abs(g.dRe(i)*g.dRe(i) / D[i]);

        double CFL_i;

        if (dtR < dtR_diff) {
            CFL_i= CFL_adv * dtR;
            // std::cout << "adv"<<"\n";
        }
        else {
            CFL_i = CFL_diff * dtR_diff;
            // std::cout << "diff"<<"\n";
        }  

        dt = std::min(dt, CFL_i);
    } 
    return dt;
}

// Calculate wind launch surface

__global__ void _calc_wind_surface(GridRef g, FieldConstRef<Quants> wg) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) {
        }
    }

}

__global__ void _calc_nH(GridRef g, FieldConstRef<Prims> wg, FieldRef<double> nH) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) {

            nH(i,j) = wg(i,j).rho/(2.8*m_H) * g.dre(i,j);

        }
    }
}

void calc_wind_surface(Grid& g, const Field<Prims>& wg, CudaArray<double>& h_w, double col) {

    Field<double> nH = create_field<double>(g);

    dim3 threads(32,32) ;
    dim3 blocks((g.NR + 2*g.Nghost+31)/32,(g.Nphi + 2*g.Nghost+31)/32) ;

    _calc_nH<<<blocks,threads>>>(g, wg, nH);

    Reduction::scan_R_sum(g, nH);
    cudaDeviceSynchronize();

    for (int i=0; i<g.NR+2*g.Nghost; i++) {
        h_w[i] = g.Zc(i,g.Nphi+2*g.Nghost-1);
        
        for (int j=g.Nghost; j<g.Nphi+2*g.Nghost; j++) {
            // if (i == 40) { std::cout << nH(i,j) << "\n";}
            if (nH(i,j) < col) {
                h_w[i] = g.Zc(i,j); 
                break;
            }
        }
    }

}

// Gas updates for Prims1D object

void update_gas_sigma(Grid& g, Field<Prims1D>& W_g, double dt, const CudaArray<double>& nu, int bound, double floor) {

    CudaArray<double> RF = make_CudaArray<double>(g.NR+2*g.Nghost);
    CudaArray<double> Sig_g = make_CudaArray<double>(g.NR+2*g.Nghost);

    size_t threads = 256 ;
    size_t blocks = (g.NR + 2*g.Nghost+255)/256 ;

    for (int i=0; i<g.NR + 2*g.Nghost; i++) {
        Sig_g[i] = W_g(i,g.Nghost).Sig;
    }

    _set_bounds(g, Sig_g.get(), bound, floor);
    _calc_Rflux<<<blocks, threads>>>(g, Sig_g.get(), RF.get(), nu.get());
    _update_Sig<<<blocks, threads>>>(g, Sig_g.get(), RF.get(), dt, floor);
    cudaDeviceSynchronize();

    for (int i=0; i<g.NR + 2*g.Nghost; i++) {
        W_g(i,g.Nghost).Sig = Sig_g[i];
    }
}

__global__ 
void _calc_v_gas(GridRef g, FieldRef<Prims1D> W_g, double GMstar, FieldConstRef<double> cs, double gas_floor, double* nu) {
    
    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int istride = gridDim.x * blockDim.x ;

    int j = g.Nghost;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {

        double P1 = W_g(i+1,j).Sig / sqrt(2.*M_PI) * cs(i+1,j) * sqrt(GMstar/(g.Rc(i+1)*g.Rc(i+1)*g.Rc(i+1)));
        double P0 = W_g(i-1,j).Sig / sqrt(2.*M_PI) * cs(i-1,j) * sqrt(GMstar/(g.Rc(i-1)*g.Rc(i-1)*g.Rc(i-1)));

        double vk = sqrt(GMstar/g.Rc(i));
        double eta = - cs(i,j)*cs(i,j) / (vk*vk) * log(P1/P0) / log(g.Rc(i+1)/g.Rc(i-1));

        W_g(i,j).v_phi = vk*sqrt(1.-eta);

        if (W_g(i,j).Sig < 10.*gas_floor) { 
            W_g(i,j).v_R  = -100; 
            continue;
        }

        double f1 = sqrt(g.Rc(i+1)) * W_g(i+1,j).Sig * nu[i+1];
        double f0 = sqrt(g.Rc(i-1)) * W_g(i-1,j).Sig * nu[i-1];

        W_g(i,j).v_R = -3./(W_g(i,j).Sig * sqrt(g.Rc(i))) * (f1-f0)/(g.Rc(i+1)-g.Rc(i-1));

        if (W_g(i,j).v_R  < -100) {
            W_g(i,j).v_R  = -100;
        }

    }
}

void calc_v_gas(Grid& g, Field<Prims1D>& W_g, const Field<double>& cs, CudaArray<double>& nu, double GMstar, double gasfloor) {

    size_t threads = 256 ;
    size_t blocks = (g.NR + 2*g.Nghost+255)/256 ;

    _calc_v_gas<<<blocks,threads>>>(g, W_g, GMstar, cs, gasfloor, nu.get());
    check_CUDA_errors("_calc_v_gas");
    cudaDeviceSynchronize();

    for (int i=0; i<g.Nghost; i++) {
        W_g(i,g.Nghost).v_R = W_g(g.Nghost,g.Nghost).v_R;
        W_g(i,g.Nghost).v_phi = W_g(g.Nghost,g.Nghost).v_phi;
    }
    for (int i=g.NR+g.Nghost; i<g.NR+2*g.Nghost; i++) {
        W_g(i,g.Nghost).v_R = W_g(g.NR+g.Nghost-1,g.Nghost).v_R;
        W_g(i,g.Nghost).v_phi = W_g(g.NR+g.Nghost-1,g.Nghost).v_phi;
    }
}