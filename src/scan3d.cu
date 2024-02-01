
#include "cuda_array.h"
#include "field.h"
#include "grid.h"
#include "utils.h"

#include "reductions.h"
#include "scan.h"

namespace Reduction {
namespace {
/* scan_Z_generic_cpu
* 
* Generic code for computing cumulative sums / products, etc, over the vertical direction.
*/
template<typename T, class Operator> 
void scan_Z_generic_cpu(const Grid& g, Field3D<T>& f) {

    for (int i = 0; i < g.NR + 2*g.Nghost; i++) {
        for (int j = 1; j < g.Nphi + 2*g.Nghost; j++) {
           for (int k=0; k < f.Nd; k++)  {
               int l1 = f.index(i,j,k) ;
               int l0 = f.index(i,j-1,k) ;
               f[l1] = Operator::apply(f[l1], f[l0]) ;
           }
        }
    }

}
 
/* scan_R_generic_cpu
 * 
 * Generic code for computing cumulative sums / products, etc, over the radial direction.
 */
template<typename T, class Operator> 
void scan_R_generic_cpu(const Grid& g, Field3D<T>& f) {

    for (int i = 1; i < g.NR + 2*g.Nghost; i++) {
        for (int j = 0; j < g.Nphi + 2*g.Nghost; j++) {
           for (int k=0; k < f.Nd; k++)  {
               int l1 = f.index(i,  j,k) ;
               int l0 = f.index(i-1,j,k) ;
               f[l1] = Operator::apply(f[l1], f[l0]) ;
           }
        }
    }

}
} // namespace
 
void scan_R_sum_cpu(const Grid& g, Field3D<double>& f) {
    scan_R_generic_cpu<double, add2<double>>(g, f) ;
}
void scan_R_mul_cpu(const Grid& g, Field3D<double>& f) {
    scan_R_generic_cpu<double, mul2<double>>(g, f) ;
}
void scan_Z_sum_cpu(const Grid& g, Field3D<double>& f) {
    scan_Z_generic_cpu<double, add2<double>>(g, f) ;
}
void scan_Z_mul_cpu(const Grid& g, Field3D<double>& f) {
     scan_Z_generic_cpu<double, mul2<double>>(g, f) ;
}

namespace {

/* scan3D_Z_generic_device
 * 
 * Parallel scan of data over Z direction for 3D arrays
 *
 */
template <class Operator, class T>
__global__ void scan3D_Z_generic_device(GridRef g, Field3DRef<T> f) {
  
    int k = threadIdx.x + blockIdx.x * blockDim.x ;
    int i = threadIdx.y + blockIdx.y * blockDim.y ;

    if (i < g.NR + 2*g.Nghost && k < f.Nd) {
        for (int j = 1; j < g.Nphi + 2*g.Nghost; j++) {
            int l1 = f.index(i,j,k) ;
            int l0 = f.index(i,j-1,k) ;
            f[l1] = Operator::apply(f[l1], f[l0]) ;
        }
    }
}  

/* scan3D_R_generic_device
 * 
 * Parallel scan of data over R direction for 3D arrays
 *
 */
template <class Operator, class T>
__global__ void scan3D_R_generic_device(GridRef g, Field3DRef<T> f) {
   
    int k = threadIdx.x + blockIdx.x * blockDim.x ;
    int j = threadIdx.y + blockIdx.y * blockDim.y ;

    if (j < g.Nphi + 2*g.Nghost && k < f.Nd) {
        for (int i = 1; i < g.NR + 2*g.Nghost; i++) {
            int l1 = f.index(i,j,k) ;
            int l0 = f.index(i-1,j,k) ;
            f[l1] = Operator::apply(f[l1], f[l0]) ;
        }
    }
}  
} // namespace


template<class OP>
void scan3D_R_OP(const Grid& g, Field3D<double>& field) {

    // Setup cuda blocks / threads and shared memory size
    int warps_per_point = (field.Nd+31)/32 ;
    int threads_per_point = std::min(32*warps_per_point, 1024) ;
    int points_per_block = 1024/threads_per_point ;

    dim3 threads(threads_per_point,points_per_block,1) ;
    dim3 blocks((field.Nd+threads_per_point-1)/threads_per_point,
                (g.Nphi+2*g.Nghost+points_per_block-1)/points_per_block,
                1) ;

    scan3D_R_generic_device<OP, double> <<<blocks, threads>>>(g, field) ;
    check_CUDA_errors("scan3D_R_OP") ;
}

template<class OP>
void scan3D_Z_OP(const Grid& g, Field3D<double>& field) {

    // Setup cuda blocks / threads and shared memory size
    int warps_per_point = (field.Nd+31)/32 ;
    int threads_per_point = std::min(32*warps_per_point, 1024) ;
    int points_per_block = 1024/threads_per_point ;

    dim3 threads(threads_per_point,points_per_block,1) ;
    dim3 blocks((field.Nd+threads_per_point-1)/threads_per_point,
                (g.NR+2*g.Nghost+points_per_block-1)/points_per_block,
                1) ;

    scan3D_Z_generic_device<OP, double> <<<blocks, threads>>>(g, field) ;
    check_CUDA_errors("scan3D_Z_OP") ;
}

void scan_R_sum(const Grid& g, Field3D<double>& field) {
    scan3D_R_OP<add2<double>>(g, field) ;
}
void scan_R_mul(const Grid& g, Field3D<double>& field){
    scan3D_R_OP<mul2<double>>(g, field) ;
}
void scan_Z_sum(const Grid& g, Field3D<double>& field){
    scan3D_Z_OP<add2<double>>(g, field) ;
}
void scan_Z_mul(const Grid& g, Field3D<double>& field){
    scan3D_Z_OP<mul2<double>>(g, field) ;
}



} // namespace Reduction