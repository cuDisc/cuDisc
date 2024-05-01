#include <stdexcept>

#include "timing.h"
#include "utils.h"
#include "coagulation/coagulation.h"
#include "coagulation/fragments.h"
#include "coagulation/integration.h"
#include "dustdynamics.h"

#include <iostream>

__global__ void _compute_ytot(GridRef g, Field3DConstRef<double> y, 
                              FieldRef<double> yscale, double scale, FieldRef<Prims> wg) {

    int j = threadIdx.x + blockIdx.x * blockDim.x ;
    int i = threadIdx.y + blockIdx.y * blockDim.y ;

    if (i < g.NR + 2*g.Nghost && j < g.Nphi + 2*g.Nghost) {
        double res = 0 ;
        for (int k=0; k<y.Nd; k++)
            res += y(i,j,k) ;
        if (res > y.Nd*10*1e-40*wg(i,j).rho) {
            yscale(i,j) = (res+1e-100)*scale ;
        }
        else {
            yscale(i,j) = 1.;
        }
    }
}

// Compute the maximum error scaled in each block. 
// The result is stored in the errtot(i,j) corresponding to threadIdx.{x,y} = 0.
__global__ void _compute_error_norm(GridRef g, 
                                    Field3DConstRef<double> y, Field3DConstRef<double> ynew, 
                                    FieldConstRef<double> yabs, double rel_tol,
                                    Field3DConstRef<double> err, FieldRef<double> errtot) {

    int j = threadIdx.x + blockIdx.x * blockDim.x ;
    int i = threadIdx.y + blockIdx.y * blockDim.y ;

    // Get the total scaled error for each cell.
    if (i < g.NR + 2*g.Nghost && j < g.Nphi + 2*g.Nghost) {
        double res = 0 ;
        double scale ;
        for (int k=0; k<y.Nd; k++) {
            scale = yabs(i,j) + max(abs(y(i,j,k)), abs(ynew(i,j,k))) * rel_tol ;
            res += err(i,j,k)*err(i,j,k) / (scale*scale) ;
        }
        errtot(i,j) = res ;
    }
    __syncthreads() ;

    // Compute the max error over each cell
    //   1. Reduce over y
    int size = blockDim.x / 2 ;
    while (size > 0) {
        if (threadIdx.x < size && (i < g.NR + 2*g.Nghost && j + size < g.Nphi + 2*g.Nghost))
            errtot(i,j) = max(errtot(i,j), errtot(i, j+size)) ;
        
        size /= 2 ;
        __syncthreads() ;
    }

    //   2. Reduce over x
    size = blockDim.y / 2 ;
    if (blockIdx.x * blockDim.x < g.Nphi + 2*g.Nghost) {        
        while (size > 0) {
            if (threadIdx.x == 0 && threadIdx.y < size && i + size < g.NR + 2*g.Nghost)
                errtot(i,j) = max(errtot(i,j), errtot(i+size, j)) ;

            size /= 2 ;
            __syncthreads() ;
        }
    }
} 
__global__ void _compute_error_norm_debug(GridRef g, 
                                    Field3DConstRef<double> y, Field3DConstRef<double> ynew, 
                                    FieldConstRef<double> yabs, double rel_tol,
                                    Field3DConstRef<double> err, FieldRef<double> errtot, Field3DRef<int> idxs) {

    int j = threadIdx.x + blockIdx.x * blockDim.x ;
    int i = threadIdx.y + blockIdx.y * blockDim.y ;

    // Get the total scaled error for each cell.
    if (i < g.NR + 2*g.Nghost && j < g.Nphi + 2*g.Nghost) {
        double res = 0 ;
        double scale ;
        for (int k=0; k<y.Nd; k++) {
            scale = yabs(i,j) + max(abs(y(i,j,k)), abs(ynew(i,j,k))) * rel_tol ;
            res += err(i,j,k)*err(i,j,k) / (scale*scale) ;
        }
        errtot(i,j) = res ;
        idxs(i,j,0) = i ;
        idxs(i,j,1) = j ;
    }
    __syncthreads() ;

    // Compute the max error over each cell
    //   1. Reduce over y
    int size = blockDim.x / 2 ;
    while (size > 0) {
        if (threadIdx.x < size && (i < g.NR + 2*g.Nghost && j + size < g.Nphi + 2*g.Nghost)) {
            if ( errtot(i, j+size) > errtot(i,j)) {
                errtot(i,j) = errtot(i, j+size);
                idxs(i,j,0) = idxs(i,j+size, 0);
                idxs(i,j,1) = idxs(i,j+size, 1);
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
                if ( errtot(i+size, j) > errtot(i,j)) {
                    errtot(i,j) = errtot(i+size, j);
                    idxs(i,j,0) = idxs(i+size, j, 0) ;
                    idxs(i,j,1) = idxs(i+size, j, 1) ;
                } 
            }
                 

            size /= 2 ;
            __syncthreads() ;
        }
    }
} 


double TimeIntegration::take_step(Grid& g, Field3D<double>& y, Field<Prims>& wg, double& dtguess) const {

    CodeTiming::BlockTimer block =
        timer->StartNewTimer("TimeIntegation::take_step");
  
    Field3D<double> ynew  = create_field3D<double>(g, y.Nd) ;
    Field3D<double> error = create_field3D<double>(g, y.Nd) ;

    Field<double> yabs    = create_field<double>(g) ;
    Field<double> err_tot = create_field<double>(g) ;

    
    double dt ;
    if (dtguess > 0) {
        // Use guess provided
        dt = dtguess ;
    }
    else {
        dt = 1 ;
    }

    dim3 threads(32,32,1) ;
    dim3 blocks((g.Nphi+2*g.Nghost+31)/32,(g.NR+2*g.Nghost+31)/32,1) ;

    // Compute the total density for the error estimation
    _compute_ytot<<<blocks,threads>>>(g, y, yabs, _abs_tol, wg) ; 
    check_CUDA_errors("_compute_ytot") ;
      
    bool success = false ;

    while (not success) {
        if (dt == 0)
            throw std::runtime_error("Error time-step of zero was assigned");
          
        do_step(dt, g, y, ynew, error) ;

        // Compute the normalized error
        _compute_error_norm<<<blocks,threads>>>(g, y, ynew, yabs, _rel_tol, 
                                                error, err_tot) ;
        check_CUDA_errors("_compute_error_norm") ;

        double err_norm = 0 ;
        for (int i=0; i < g.NR + 2*g.Nghost; i += 32)
            for (int j=0; j < g.Nphi + 2*g.Nghost; j += 32)
                err_norm = std::max(err_norm, err_tot(i,j)) ;

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
double TimeIntegration::take_step_debug(Grid& g, Field3D<double>& y, Field<Prims>& wg, double& dtguess, int* idxs) const {

    CodeTiming::BlockTimer block =
        timer->StartNewTimer("TimeIntegation::take_step");
  
    Field3D<double> ynew  = create_field3D<double>(g, y.Nd) ;
    Field3D<double> error = create_field3D<double>(g, y.Nd) ;

    Field<double> yabs    = create_field<double>(g) ;
    Field<double> err_tot = create_field<double>(g) ;

    Field3D<int> idxgrid = create_field3D<int>(g, 2);

    
    double dt ;
    if (dtguess > 0) {
        // Use guess provided
        dt = dtguess ;
    }
    else {
        dt = 1 ;
    }

    dim3 threads(32,32,1) ;
    dim3 blocks((g.Nphi+2*g.Nghost+31)/32,(g.NR+2*g.Nghost+31)/32,1) ;

    // Compute the total density for the error estimation
    _compute_ytot<<<blocks,threads>>>(g, y, yabs, _abs_tol, wg) ; 
    check_CUDA_errors("_compute_ytot") ;
      
    bool success = false ;

    while (not success) {
        if (dt == 0)
            throw std::runtime_error("Error time-step of zero was assigned");
          
        do_step(dt, g, y, ynew, error) ;

        // Compute the normalized error
        _compute_error_norm_debug<<<blocks,threads>>>(g, y, ynew, yabs, _rel_tol, 
                                                error, err_tot, idxgrid) ;
        check_CUDA_errors("_compute_error_norm") ;

        double err_norm = 0 ;
        for (int i=0; i < g.NR + 2*g.Nghost; i += 32)
            for (int j=0; j < g.Nphi + 2*g.Nghost; j += 32)
                if (err_tot(i,j) > err_norm) {
                    err_norm = std::max(err_norm, err_tot(i,j)) ;
                    idxs[0] = idxgrid(i,j,0);
                    idxs[1] = idxgrid(i,j,1);
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

void TimeIntegration::integrate(Grid& g, Field3D<double>& y, double tmax) const {
    double dt = tmax ;
    double t = 0 ;
    Field<Prims> wg = create_field<Prims>(g);
    while (t < tmax) {
        dt = std::min(dt, tmax-t) ;
        t += take_step(g, y, wg, dt) ;
    }
}

__global__ void _copy_rho_forwards(GridRef g, Field3DRef<Prims> ws, FieldRef<Prims> wg, Field3DRef<double> rhos, double floor) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.z * blockDim.z ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost; j+=jstride) { 
            for (int k=kidx; k<ws.Nd; k+=kstride) { 
                rhos(i,j,k) = max(ws(i,j,k).rho-floor*wg(i,j).rho, 0.);
            }
        }
    }
}

__global__ void _copy_rho_backwards(GridRef g, Field3DRef<Prims> ws, FieldRef<Prims> wg, Field3DRef<double> rhos, double floor) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.z * blockDim.z ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) { 
            for (int k=kidx; k<ws.Nd; k+=kstride) { 
                ws(i,j,k).rho = rhos(i,j,k) + floor*wg(i,j).rho; 
            }
        }
    }
}

double calc_mass(Grid& g, Field3D<double>& q) {

    double mass=0;

    for (int i=g.Nghost; i<g.NR+g.Nghost; i++) {
        for (int j=g.Nghost; j<g.Nphi+g.Nghost; j++) {
            for (int k=0; k<q.Nd; k++) {
                mass += 4*M_PI* q(i,j,k) * g.volume(i,j);
            }
        }
    }

    return mass;
}
double calc_mass_cell(Grid& g, Field3D<double>& q) {

    double mass=0;

    for (int k=0; k<q.Nd; k++) {
        mass += q(52,2,k);
    }


    return mass;
}



void TimeIntegration::integrate(Grid& g, Field3D<Prims>& ws, Field<Prims>& wg, double tmax, double& dt_coag, double floor) const {
    double dt = dt_coag ;
    if (dt_coag < tmax && dt_coag > _SAFETY*tmax)
        dt /= 2 ;

    double t = 0 ;

    Field3D<double> rhos = create_field3D<double>(g, ws.Nd);
    set_all(g, rhos, 0.);

    dim3 threads(16,8,8);
    dim3 blocks((g.NR + 2*g.Nghost+15)/16,(g.Nphi + 2*g.Nghost+7)/8, (ws.Nd+7)/8) ;

    _copy_rho_forwards<<<blocks,threads>>>(g, ws, wg, rhos, floor);
    cudaDeviceSynchronize();
    int count = 0;

    while (t < tmax) {
        dt = std::min(dt, tmax-t) ;
        t += take_step(g, rhos, wg, dt) ;
        count += 1;
        if ((count%100) == 0) {
            std::cout << "Coagulation Steps = " << count << ", dt_coag = " << dt/year << " years, t = " << t/year << " years \n";
        }
    }
    std::cout << "Coagulation Steps = " << count << ", dt_coag = " << dt/year << " years, t = " << t/year << " years \n";
    
    dt_coag = dt;

    _copy_rho_backwards<<<blocks,threads>>>(g, ws, wg, rhos, floor);
}

void TimeIntegration::integrate_debug(Grid& g, Field3D<Prims>& ws, Field<Prims>& wg, double tmax, double& dt_coag, double floor) const {
    double dt = dt_coag ; 
    if (dt_coag < tmax && dt_coag > _SAFETY*tmax)
        dt /= 2 ;
    double t = 0 ;

    Field3D<double> rhos = create_field3D<double>(g, ws.Nd);
    set_all(g, rhos, 0.);
    
    dim3 threads(16,8,8);
    dim3 blocks((g.NR + 2*g.Nghost+15)/16,(g.Nphi + 2*g.Nghost+7)/8, (ws.Nd+7)/8) ;

    _copy_rho_forwards<<<blocks,threads>>>(g, ws, wg, rhos, floor);
    cudaDeviceSynchronize();
    int count = 0;
    int idxs[2] = {0,0};

    while (t < tmax) {


        dt = std::min(dt, tmax-t) ;
        t += take_step_debug(g, rhos, wg, dt, idxs) ;
        if (!(count%100) && count) {
            std::cout << "Count = " << count << ", dt_coag = " << dt/year << " years, t = " << t/year << " years \n";
            std::cout << "i index = " << idxs[0] << ", j index = " << idxs[1] << "\n";
        }
        count += 1;
        // printf("%1.12g %1.12g %g\n", calc_mass(g,rhos), calc_mass_cell(g,rhos), dt);
        // printf("%1.12g\n", calc_mass_cell(g,rhos));
    }
    std::cout << "Count = " << count << ", dt_coag = " << dt/year << " years, t = " << t/year << " years \n";
    std::cout << "i index = " << idxs[0] << ", j index = " << idxs[1] << "\n";

    dt_coag = dt;

    _copy_rho_backwards<<<blocks,threads>>>(g, ws, wg, rhos, floor);
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
                                   Field3D<double>& ynew, Field3D<double>& error) const {

    CodeTiming::BlockTimer block =
        timer->StartNewTimer("Rk2Integration::do_step") ;

    Field3D<double>& rate = error ;

    // Compute the rate
    this->operator()(y, rate) ;

    dim3 threads(32,8,4) ;
    dim3 blocks((y.Nd+31)/32, (g.Nphi +2*g.Nghost + 7)/8, (g.NR + 2*g.Nghost + 3)/4);

    // 1st guess (Euler's method)
    _Rk2_update1<<<blocks, threads>>>(g, y, rate, dt, ynew) ;
    check_CUDA_errors("_Rk2_update1") ;

    // Compute the rate, correction, and error (Heun's method)
    this->operator()(ynew, rate) ;

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
                                   Field3D<double>& ynew, Field3D<double>& error) const {
                                
    // Bogacki-Shampine embedded Runge-Kutta 3(2) method: https://www.sciencedirect.com/science/article/pii/0893965989900797
    
    CodeTiming::BlockTimer block =
        timer->StartNewTimer("BS32Integration::do_step") ;

    Field3D<double> k1 = create_field3D<double>(g, y.Nd);
    Field3D<double> k2 = create_field3D<double>(g, y.Nd);
    Field3D<double> k3 = create_field3D<double>(g, y.Nd);
    Field3D<double> k4 = create_field3D<double>(g, y.Nd);

    dim3 threads(32,8,4) ;
    dim3 blocks((y.Nd+31)/32, (g.Nphi +2*g.Nghost + 7)/8, (g.NR + 2*g.Nghost + 3)/4);

    this->operator()(y, k1) ;

    _BS32_update1<<<blocks, threads>>>(g, y, k1, dt, ynew) ;

    this->operator()(ynew, k2) ;

    _BS32_update2<<<blocks, threads>>>(g, y, k2, dt, ynew) ;

    this->operator()(ynew, k3) ;

    _BS32_update3<<<blocks, threads>>>(g, y, k1, k2, k3, dt, ynew) ;

    this->operator()(ynew, k4) ;

    _BS32_update4<<<blocks, threads>>>(g, y, dt, ynew, k1, k2, k3, k4, error) ;

}



template class Rk2Integration<CoagulationRate<BirnstielKernel,SimpleErosion>> ;
template class Rk2Integration<CoagulationRate<ConstantKernel,SimpleErosion>> ;
template class Rk2Integration<CoagulationRate<BirnstielKernelVertInt,SimpleErosion>> ;
template class BS32Integration<CoagulationRate<BirnstielKernel,SimpleErosion>> ;
template class BS32Integration<CoagulationRate<ConstantKernel,SimpleErosion>> ;
template class BS32Integration<CoagulationRate<BirnstielKernelVertInt,SimpleErosion>> ;