
#include <cuda_runtime.h>

#include "grid.h"
#include "super_stepping.h"


__global__ void _super_stepping_update_solution(
        GridRef g,
        Field3DConstRef<double> u0, 
        Field3DConstRef<double> u_current,
        Field3DRef<double> u_old_new,
        Field3DConstRef<double> l0, 
        Field3DConstRef<double> l,
        double mu, double nu, double mup, double gam,
        double dt)  {

   int i = threadIdx.x + blockIdx.x*blockDim.x ;
   int step = blockDim.x * gridDim.x ;

   int size = (g.NR + 2*g.Nghost)*u0.stride_Zd ;

    while (i < size) {
        u_old_new[i] = mu * u_current[i] + nu*u_old_new[i] + (1-mu-nu) * u0[i]
            + dt*(mup * l[i] + gam * l0[i]) ;
        i += step ;
    }

}



void SuperStepping::update_solution(Field3DConstRef<double> u0, 
                                    Field3DConstRef<double> u_current,
                                    Field3DRef<double> u_old_new,
                                    Field3DConstRef<double> l0, 
                                    Field3DConstRef<double> l,
                                    double mu, double nu, double mup, double gam,
                                    double dt) const {

    int blocks = ((_grid.NR + 2*_grid.Nghost)*u0.stride_Zd + 1023) / 1024 ;
    
    if (u0.stride_Zd != u_current.stride_Zd)
        throw std::runtime_error("Shapes of arrays do not match in "
                                 "SuperStepping::update_solution") ;

    _super_stepping_update_solution<<<blocks, 1024>>>(
        _grid, u0, u_current, u_old_new, l0, l, mu, nu, mup, gam, dt
    ) ;

    check_CUDA_errors("_super_stepping_update_solution") ;

}
