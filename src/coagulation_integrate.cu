#include <stdexcept>

#include "timing.h"
#include "utils.h"
#include "coagulation/coagulation.h"
#include "coagulation/fragments.h"
#include "coagulation/integration.h"
#include "dustdynamics.h"

#include <iostream>

__global__ void _compute_ytot(GridRef g, Field3DConstRef<double> y, 
                              FieldRef<double> yscale, double scale) {

    int j = threadIdx.x + blockIdx.x * blockDim.x ;
    int i = threadIdx.y + blockIdx.y * blockDim.y ;

    if (i < g.NR + 2*g.Nghost && j < g.Nphi + 2*g.Nghost) {
        double res = 0 ;
        for (int k=0; k<y.Nd; k++)
            res += y(i,j,k) ;

        yscale(i,j) = (res+1e-100)*scale ;

    }
}

template<typename T>
__device__ double& _density(T& value) {
    return value[0];
}

template<>
__device__ double& _density<double>(double& value) {
    return value;
}

template<typename T>
__device__ double _density(const T& value) {
    return value[0];
}

template<>
__device__ double _density<double>(const double& value) {
    return value;
}

// Compute the maximum error scaled in each block. 
// The result is stored in the errtot(i,j) corresponding to threadIdx.{x,y} = 0.

template<bool debug, typename T>
__global__ void _compute_error_norm(GridRef g,
                                    Field3DConstRef<double> y, Field3DConstRef<double> ynew,
                                    FieldConstRef<double> yabs, FieldConstRef<T> wg,
                                    double floor, double rel_tol,
                                    Field3DConstRef<double> err, FieldRef<double> errtot,
                                    Field3DRef<int> idxs) {

    int j = threadIdx.x + blockIdx.x * blockDim.x ;
    int i = threadIdx.y + blockIdx.y * blockDim.y ;

    // Get the total scaled error for each cell.
    if (i < g.NR + 2*g.Nghost && j < g.Nphi + 2*g.Nghost) {
        double res = 0 ;
        double scale ;
        for (int k=0; k<y.Nd; k++) {
            double floor_density = floor * _density(wg(i,j));
            scale = yabs(i,j) + rel_tol * max(max(abs(y(i,j,k)), abs(ynew(i,j,k))),
                                               floor_density) ;
            res += err(i,j,k)*err(i,j,k) / (scale*scale) ;
        }
        errtot(i,j) = res ;
        if constexpr (debug) {
            idxs(i,j,0) = i;
            idxs(i,j,1) = j;
        }
    }
    __syncthreads() ;

    // Compute the max error over each cell
    //   1. Reduce over y
    int size = blockDim.x / 2 ;
    while (size > 0) {
        if (threadIdx.x < size && (i < g.NR + 2*g.Nghost && j + size < g.Nphi + 2*g.Nghost)) {
            if (errtot(i, j+size) > errtot(i,j)) {
                errtot(i,j) = errtot(i, j+size);
                if constexpr (debug) {
                    idxs(i,j,0) = idxs(i,j+size,0);
                    idxs(i,j,1) = idxs(i,j+size,1);
                }
            }
        }
        
        size /= 2 ;
        __syncthreads() ;
    }

    //   2. Reduce over x
    size = blockDim.y / 2 ;
    if (blockIdx.x * blockDim.x < g.Nphi + 2*g.Nghost) {        
        while (size > 0) {
            if (threadIdx.x == 0 && threadIdx.y < size && i + size < g.NR + 2*g.Nghost) {
                if (errtot(i+size, j) > errtot(i,j)) {
                    errtot(i,j) = errtot(i+size, j);
                    if constexpr (debug) {
                        idxs(i,j,0) = idxs(i+size,j,0);
                        idxs(i,j,1) = idxs(i+size,j,1);
                    }
                }
            }

            size /= 2 ;
            __syncthreads() ;
        }
    }
}

template<bool debug, typename T>
double TimeIntegration::take_step_impl(Grid& g, Field3D<double>& y,
                      Field<T>& wg, double& dtguess, int* idxs,
                      Field<bool>& active, double floor) const {

    CodeTiming::BlockTimer block =
        timer->StartNewTimer("TimeIntegation::take_step");
  
    Field3D<double> ynew  = create_field3D<double>(g, y.Nd) ;
    Field3D<double> error = create_field3D<double>(g, y.Nd) ;

    Field<double> yabs    = create_field<double>(g) ;
    Field<double> err_tot = create_field<double>(g) ;

    Field3D<int> idxgrid = create_field3D<int>(g, 2);

    double dt ;
    if (dtguess > 0) {
        dt = dtguess ;
    }
    else {
        dt = 1 ;
    }

    dim3 threads(32,32,1) ;
    dim3 blocks((g.Nphi+2*g.Nghost+31)/32,(g.NR+2*g.Nghost+31)/32,1) ;

    _compute_ytot<<<blocks,threads>>>(g, y, yabs, _abs_tol) ;
    check_CUDA_errors("_compute_ytot") ;

    bool success = false ;
    while (not success) {
        if (dt == 0)
            throw std::runtime_error("Error time-step of zero was assigned");

        do_step(dt, g, y, ynew, error, active) ;

        _compute_error_norm<debug,T><<<blocks,threads>>>(g, y, ynew, yabs,
            FieldConstRef<T>(wg), floor, _rel_tol, error, err_tot, idxgrid) ;
        check_CUDA_errors("_compute_error_norm") ;

        double err_norm = 0 ;
        for (int i=0; i < g.NR + 2*g.Nghost; i += 32)
            for (int j=0; j < g.Nphi + 2*g.Nghost; j += 32)
                if (err_tot(i,j) > err_norm) {
                    err_norm = err_tot(i,j);
                    if constexpr (debug) {
                        idxs[0] = idxgrid(i,j,0);
                        idxs[1] = idxgrid(i,j,1);
                    }
                }

        if (err_norm < 1) {
            success = true ;

            dtguess = dt * std::min(_MAX_FACTOR,
                                std::max(1., _SAFETY * std::pow(err_norm, -0.5 / _order)));
        } else {
            dt  = dt * std::max(_MIN_FACTOR, _SAFETY * std::pow(err_norm, -0.5 / _order)) ;
        }
    }

    copy_field(g, ynew, y) ;

    return dt ;
}

template<typename T>
double TimeIntegration::take_step(Grid& g, Field3D<double>& y, Field<T>& wg, double& dtguess,
                                  Field<bool>& active, double floor) const {
    return take_step_impl<false>(g, y, wg, dtguess, nullptr, active, floor);
}

template<typename T>
double TimeIntegration::take_step_debug(Grid& g, Field3D<double>& y, Field<T>& wg, double& dtguess,
                                        int* idxs, Field<bool>& active, double floor) const {
    return take_step_impl<true>(g, y, wg, dtguess, idxs, active, floor);
}

template<typename T>
__global__ void _copy_rho_forwards(GridRef g, Field3DRef<T> ws, FieldRef<T> wg, Field3DRef<double> rhos, double floor) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.z * blockDim.z ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost; j+=jstride) { 
            for (int k=kidx; k<ws.Nd; k+=kstride) { 
                rhos(i,j,k) = max(_density(ws(i,j,k)) - floor*_density(wg(i,j)), 0.);
            }
        }
    }
}

template<typename T>
__global__ void _copy_rho_backwards(GridRef g, Field3DRef<T> ws, FieldRef<T> wg, Field3DRef<double> rhos, double floor) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.z * blockDim.z ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost; j+=jstride) { 
            for (int k=kidx; k<ws.Nd; k+=kstride) { 
                _density(ws(i,j,k)) = rhos(i,j,k) + floor*_density(wg(i,j));
            }
        }
    }
}

template<bool debug, typename T>
int TimeIntegration::integrate_impl(Grid& g, Field3D<T>& ws, Field<T>& wg, double tmax, double& dt_coag, double floor) const {
    double dt = dt_coag ;
    if (dt_coag < tmax && dt_coag > _SAFETY*tmax)
        dt /= 2 ;

    double t = 0 ;

    Field3D<double> rhos = create_field3D<double>(g, ws.Nd);
    Field<bool> active = create_field<bool>(g); 
    set_all(g, rhos, 0.);

    dim3 threads(16,8,8);
    dim3 blocks((g.NR + 2*g.Nghost+15)/16,(g.Nphi + 2*g.Nghost+7)/8, (ws.Nd+7)/8) ;

    _copy_rho_forwards<<<blocks,threads>>>(g, Field3DRef<T>(ws), FieldRef<T>(wg), rhos, floor);
    _check_active<<<blocks,threads>>>(g, FieldRef<T>(wg), rhos, active, floor);
    cudaDeviceSynchronize();
    int count = 0;
    int idxs[2] = {0,0};

    while (t < tmax) {
        dt = std::min(dt, tmax-t) ;
        if constexpr (debug) {
            t += take_step_debug(g, rhos, wg, dt, idxs, active, floor) ;
        }
        else {
            t += take_step(g, rhos, wg, dt, active, floor) ;
        }
        count += 1;

        bool print_progress = (count % 100) == 0 && (debug || _verbose);
        if (print_progress) {
            std::cout << "Coagulation Steps = " << count << ", dt_coag = " << dt/year << " years, t = " << t/year << " years \n";
            if constexpr (debug) {
                std::cout << "i index = " << idxs[0] << ", j index = " << idxs[1] << "\n";
            }
        }
    }
    if (debug || _verbose) {
        std::cout << "Coagulation Steps = " << count << ", dt_coag = " << dt/year << " years, t = " << t/year << " years \n";
        if constexpr (debug) {
            std::cout << "i index = " << idxs[0] << ", j index = " << idxs[1] << "\n";
        }
    }
    
    dt_coag = dt;

    _copy_rho_backwards<<<blocks,threads>>>(g, Field3DRef<T>(ws), FieldRef<T>(wg), rhos, floor);

    return count ;
}

template<typename T>
int TimeIntegration::integrate(Grid& g, Field3D<T>& ws, Field<T>& wg,
                               double tmax, double& dt_coag, double floor) const {
    return integrate_impl<false>(g, ws, wg, tmax, dt_coag, floor);
}

template<typename T>
int TimeIntegration::integrate_debug(Grid& g, Field3D<T>& ws, Field<T>& wg,
                                     double tmax, double& dt_coag, double floor) const {
    return integrate_impl<true>(g, ws, wg, tmax, dt_coag, floor);
}

__global__ void _Rk2_update1(GridRef g, Field3DConstRef<double> y, 
                             Field3DConstRef<double> rate, double dt, Field3DRef<double> y_new) {
   
    int i = threadIdx.z + blockIdx.z * blockDim.z ;
    int j = threadIdx.y + blockIdx.y * blockDim.y ;
    int k = threadIdx.x + blockIdx.x * blockDim.x ;

    if (i < g.NR + 2*g.Nghost && j < g.Nphi + 2*g.Nghost) {

        for (/**/; k < y.Nd; k += blockDim.x*gridDim.x) 
            y_new(i,j,k) = max(y(i,j,k) + rate(i,j,k) * dt, 0.0) ;
    }
}

// Compute Heun's method update. 
//   Note that error is used as the rate on input
__global__ void _Rk2_update2(GridRef g, Field3DConstRef<double> y, 
                             double dt, Field3DRef<double> y_new, Field3DRef<double> error) {

    int i = threadIdx.z + blockIdx.z * blockDim.z ;
    int j = threadIdx.y + blockIdx.y * blockDim.y ;
    int k = threadIdx.x + blockIdx.x * blockDim.x ;

    if (i < g.NR + 2*g.Nghost && j < g.Nphi + 2*g.Nghost) {
        for (/**/; k < y.Nd; k += blockDim.x*gridDim.x) {
            double ys = y_new(i,j,k) ;
            double y1 = max(0.5*(ys + y(i,j,k) + error(i,j,k) * dt), 0.);
        
            y_new(i,j,k) = y1 ;
            error(i,j,k) = y1-ys ;
        }
    }
}


template<class Rate>
void Rk2Integration<Rate>::do_step(double dt, Grid& g, const Field3D<double>& y,
                                   Field3D<double>& ynew, Field3D<double>& error, Field<bool>& active) const {

    CodeTiming::BlockTimer block =
        timer->StartNewTimer("Rk2Integration::do_step") ;

    Field3D<double>& rate = error ;

    // Compute the rate
    this->operator()(y, rate, active) ;

    dim3 threads(32,8,4) ;
    dim3 blocks((y.Nd+31)/32, (g.Nphi +2*g.Nghost + 7)/8, (g.NR + 2*g.Nghost + 3)/4);

    // 1st guess (Euler's method)
    _Rk2_update1<<<blocks, threads>>>(g, y, rate, dt, ynew) ;
    check_CUDA_errors("_Rk2_update1") ;

    // Compute the rate, correction, and error (Heun's method)
    this->operator()(ynew, rate, active) ;

    _Rk2_update2<<<blocks, threads>>>(g, y, dt, ynew, error) ;
    check_CUDA_errors("_Rk2_update2") ;

}

__global__ void _BS32_update1(GridRef g, Field3DConstRef<double> y, 
                             Field3DConstRef<double> rate, double dt, Field3DRef<double> y_new) {
   
    int i = threadIdx.z + blockIdx.z * blockDim.z ;
    int j = threadIdx.y + blockIdx.y * blockDim.y ;
    int k = threadIdx.x + blockIdx.x * blockDim.x ;

    if (i < g.NR + 2*g.Nghost && j < g.Nphi + 2*g.Nghost) {

        for (/**/; k < y.Nd; k += blockDim.x*gridDim.x) 
            y_new(i,j,k) = max(y(i,j,k) + 0.5*rate(i,j,k) * dt, 0.0) ;
    }
}
__global__ void _BS32_update2(GridRef g, Field3DConstRef<double> y, Field3DConstRef<double> rate2, double dt, Field3DRef<double> y_new) {
   
    int i = threadIdx.z + blockIdx.z * blockDim.z ;
    int j = threadIdx.y + blockIdx.y * blockDim.y ;
    int k = threadIdx.x + blockIdx.x * blockDim.x ;

    if (i < g.NR + 2*g.Nghost && j < g.Nphi + 2*g.Nghost) {

        for (/**/; k < y.Nd; k += blockDim.x*gridDim.x) 
            y_new(i,j,k) = max(y(i,j,k) + dt*(3./4. * rate2(i,j,k)), 0.0) ;
    }
}

__global__ void _BS32_update3(GridRef g, Field3DConstRef<double> y, 
                             Field3DConstRef<double> rate1, Field3DConstRef<double> rate2, Field3DConstRef<double> rate3, double dt, Field3DRef<double> y_new) {
   
    int i = threadIdx.z + blockIdx.z * blockDim.z ;
    int j = threadIdx.y + blockIdx.y * blockDim.y ;
    int k = threadIdx.x + blockIdx.x * blockDim.x ;

    if (i < g.NR + 2*g.Nghost && j < g.Nphi + 2*g.Nghost) {

        for (/**/; k < y.Nd; k += blockDim.x*gridDim.x) 
            y_new(i,j,k) = max(y(i,j,k) + dt*(2./9. * rate1(i,j,k) + 1./3. * rate2(i,j,k) + 4./9. * rate3(i,j,k)), 0.0) ;
    }
}


__global__ void _BS32_update4(GridRef g, Field3DConstRef<double> y, 
                             double dt, Field3DRef<double> y_new, Field3DConstRef<double> rate1, Field3DConstRef<double> rate2, 
                             Field3DConstRef<double> rate3, Field3DConstRef<double> rate4, Field3DRef<double> error) {

    int i = threadIdx.z + blockIdx.z * blockDim.z ;
    int j = threadIdx.y + blockIdx.y * blockDim.y ;
    int k = threadIdx.x + blockIdx.x * blockDim.x ;

    if (i < g.NR + 2*g.Nghost && j < g.Nphi + 2*g.Nghost) {
        for (/**/; k < y.Nd; k += blockDim.x*gridDim.x) {

            double ys = y_new(i,j,k);
            double y1 = max(y(i,j,k) + dt*(7./24. * rate1(i,j,k) + 1./4. * rate2(i,j,k) + 1./3. * rate3(i,j,k) + 1./8. * rate4(i,j,k)), 0.);
        
            y_new(i,j,k) = y1 ;
            error(i,j,k) = y1-ys ;
        }
    }
}


template<class Rate>
void BS32Integration<Rate>::do_step(double dt, Grid& g, const Field3D<double>& y,
                                   Field3D<double>& ynew, Field3D<double>& error, Field<bool>& active) const {
                                
    // Bogacki-Shampine embedded Runge-Kutta 3(2) method: https://www.sciencedirect.com/science/article/pii/0893965989900797
    
    CodeTiming::BlockTimer block =
        timer->StartNewTimer("BS32Integration::do_step") ;

    Field3D<double> k1 = create_field3D<double>(g, y.Nd);
    Field3D<double> k2 = create_field3D<double>(g, y.Nd);
    Field3D<double> k3 = create_field3D<double>(g, y.Nd);
    Field3D<double> k4 = create_field3D<double>(g, y.Nd);

    dim3 threads(32,8,4) ;
    dim3 blocks((y.Nd+31)/32, (g.Nphi +2*g.Nghost + 7)/8, (g.NR + 2*g.Nghost + 3)/4);

    this->operator()(y, k1, active) ;

    _BS32_update1<<<blocks, threads>>>(g, y, k1, dt, ynew) ;

    this->operator()(ynew, k2, active) ;

    _BS32_update2<<<blocks, threads>>>(g, y, k2, dt, ynew) ;

    this->operator()(ynew, k3, active) ;

    _BS32_update3<<<blocks, threads>>>(g, y, k1, k2, k3, dt, ynew) ;

    this->operator()(ynew, k4, active) ;

    _BS32_update4<<<blocks, threads>>>(g, y, dt, ynew, k1, k2, k3, k4, error) ;

}


template<typename T>
__global__ void _check_active(GridRef g, FieldRef<T> wg, Field3DRef<double> rhos, FieldRef<bool> active, double floor) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost; j+=jstride) {
            active(i,j) = false;
            for (int k=0; k<rhos.Nd; k++) {
                if (rhos(i,j,k) > 10.*floor*_density(wg(i,j))) {
                    active(i,j) = true;
                    break;
                }
            }
        }
    }
}


template class Rk2Integration<CoagulationRate<BirnstielKernel<true>,SimpleErosion>> ;
template class Rk2Integration<CoagulationRate<BirnstielKernel<false>,SimpleErosion>> ;
template class Rk2Integration<CoagulationRate<BirnstielKernelVertInt<false>,SimpleErosion>> ;
template class Rk2Integration<CoagulationRate<BirnstielKernelVertInt<true>,SimpleErosion>> ;
template class Rk2Integration<CoagulationRate<ConstantKernel,SimpleErosion>> ;

template class BS32Integration<CoagulationRate<BirnstielKernel<true>,SimpleErosion>> ;
template class BS32Integration<CoagulationRate<BirnstielKernel<false>,SimpleErosion>> ;
template class BS32Integration<CoagulationRate<BirnstielKernelVertInt<false>,SimpleErosion>> ;
template class BS32Integration<CoagulationRate<BirnstielKernelVertInt<true>,SimpleErosion>> ;
template class BS32Integration<CoagulationRate<ConstantKernel,SimpleErosion>> ;


template int TimeIntegration::integrate_debug<Prims>(Grid& g, Field3D<Prims>& ws, Field<Prims>& wg, double tmax, double& dt_coag, double floor) const;
template int TimeIntegration::integrate_debug<Prims1D>(Grid& g, Field3D<Prims1D>& ws, Field<Prims1D>& wg, double tmax, double& dt_coag, double floor) const;
template int TimeIntegration::integrate_debug<double>(Grid& g, Field3D<double>& ws, Field<double>& wg, double tmax, double& dt_coag, double floor) const;

template int TimeIntegration::integrate<Prims>(Grid& g, Field3D<Prims>& ws, Field<Prims>& wg, double tmax, double& dt_coag, double floor) const;
template int TimeIntegration::integrate<Prims1D>(Grid& g, Field3D<Prims1D>& ws, Field<Prims1D>& wg, double tmax, double& dt_coag, double floor) const;
template int TimeIntegration::integrate<double>(Grid& g, Field3D<double>& ws, Field<double>& wg, double tmax, double& dt_coag, double floor) const;