
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
void scan_Z_generic_cpu(const Grid& g, Field<T>& f) {

    int stride = f.stride ;

    for (int i = 0; i < g.NR + 2*g.Nghost; i++) {
        double tot = Operator::identity() ;
        for (int j = 0; j < g.Nphi + 2*g.Nghost; j++) {
            tot = Operator::apply(tot, f[i*stride + j]) ;
            f[i*stride + j] = tot ;
        }
    }
}

/* scan_R_generic_cpu
 * 
 * Generic code for computing cumulative sums / products, etc, over the radial direction.
 */
template<typename T, class Operator> 
void scan_R_generic_cpu(const Grid& g, Field<T>& f) {

    int stride = f.stride ;

    for (int j = 0; j < g.Nphi + 2*g.Nghost; j++) {
        double tot = Operator::identity() ;
        for (int i = 0; i < g.NR + 2*g.Nghost; i++) {
            tot = Operator::apply(tot, f[i*stride + j]) ;
            f[i*stride + j] = tot ;
        }
    }

}
} // namespace

void scan_R_sum_cpu(const Grid& g, Field<double>& f) {
    scan_R_generic_cpu<double, add2<double>>(g, f) ;
}
void scan_R_mul_cpu(const Grid& g, Field<double>& f) {
    scan_R_generic_cpu<double, mul2<double>>(g, f) ;
}
void scan_Z_sum_cpu(const Grid& g, Field<double>& f) {
    scan_Z_generic_cpu<double, add2<double>>(g, f) ;
}
void scan_Z_mul_cpu(const Grid& g, Field<double>& f) {
    scan_Z_generic_cpu<double, mul2<double>>(g, f) ;
}

namespace {


/* scan_Z_generic_device
 * 
 * Parallel scan of data over Z direction.
 *
 * Notes:
 *  - Uses one block of threads for each radial point (i.e. globalDim.x == 1).
 *
 */
template <class OP, ScanKind Kind, class T>
__global__ void scan_Z_generic_device(GridRef g, FieldRef<T> f) {
    int tid = threadIdx.x + threadIdx.y*blockDim.x ;

    extern __shared__ double tmp[] ;
    tmp[tid] = OP::identity() ;

    // Step 1: Reduce work to a block of at most blockDim.xelements.
    int worksize = (g.Nphi + 2*g.Nghost + blockDim.x - 1) / blockDim.x ;
    int i = threadIdx.y + blockIdx.y * blockDim.y ;
    for (int k=0; k < worksize; k++) {
        int j = threadIdx.x * worksize + k ;
        if (i < g.NR + 2*g.Nghost && j < g.Nphi + 2*g.Nghost) {
        
            T val = tmp[tid] ;
            tmp[tid] = OP::apply(tmp[tid], f[i*f.stride + j]) ;

            if (Kind == ScanKind::inclusive)
                f[i*f.stride + j] = tmp[tid] ;
            else
                f[i*f.stride + j] = val ;
        }
    }
    __syncthreads() ;

    // Step 2: Perform an exclusive on reduced array:
    scan_block<OP, ScanKind::exclusive,T>
        (tmp + threadIdx.y*blockDim.x, threadIdx.x) ;

    // Step3: Add the result of step 2 to 1 to get the final result
    for (int k=0; k < worksize; k++) {
        int j = threadIdx.x * worksize + k ;
        if (i < g.NR + 2*g.Nghost && j < g.Nphi + 2*g.Nghost) {
            f[i*f.stride + j] = OP::apply(f[i*f.stride + j], tmp[tid]) ;
        }
    }
}  


/* scan_R_generic_device
 * 
 * Parallel scan of data over R direction.
 *
 * Notes:
 *  - Uses one block of threads for each vertical point (i.e. globalDim.x == 1).
 *
 */
 template <class OP, ScanKind Kind, class T>
__global__ void scan_R_generic_device(GridRef g, FieldRef<T> f) {
    int tid = threadIdx.x + threadIdx.y*blockDim.x ;

    extern __shared__ double tmp[] ;
    tmp[tid] = OP::identity() ;
 
    // Step 1: Reduce work to a block of at most blockDim.xelements.
    int worksize = (g.Nphi + 2*g.Nghost + blockDim.x - 1) / blockDim.x ;
    int j = threadIdx.y + blockIdx.y * blockDim.y ;
    for (int k=0; k < worksize; k++) {
        int i = threadIdx.x * worksize + k ;
        if (i < g.NR + 2*g.Nghost && j < g.Nphi + 2*g.Nghost) {
        
            T val = tmp[tid] ;
            tmp[tid] = OP::apply(tmp[tid], f[i*f.stride + j]) ;

            if (Kind == ScanKind::inclusive)
                f[i*f.stride + j] = tmp[tid] ;
            else
                f[i*f.stride + j] = val ;
        }
    }
    __syncthreads() ;
 
    // Step 2: Perform an exclusive on reduced array:
    scan_block<OP, ScanKind::exclusive,T>
        (tmp + threadIdx.y*blockDim.x, threadIdx.x) ;
 
    // Step3: Add the result of step 2 to 1 to get the final result
    for (int k=0; k < worksize; k++) {
        int i = threadIdx.x * worksize + k ;
        if (i < g.NR + 2*g.Nghost && j < g.Nphi + 2*g.Nghost) {
            f[i*f.stride + j] = OP::apply(f[i*f.stride + j], tmp[tid]) ;
        }
    }
}

} // namespace

template<class OP, ScanKind Kind>
void scan_R_OP(const Grid& g, Field<double>& field) {

    // Setup cuda blocks / threads and shared memory size
    int warps_per_point = (g.NR + 2*g.Nghost+31)/32 ;
    int threads_per_point = std::min(32*warps_per_point, 1024) ;
    int points_per_block = 1024/threads_per_point ;

    dim3 threads(threads_per_point,points_per_block,1) ;
    dim3 blocks(1,(g.Nphi+2*g.Nghost+points_per_block-1)/points_per_block,1) ;
    int mem_size = 1024*sizeof(double) ;

    scan_R_generic_device<OP, Kind, double> <<<blocks, threads, mem_size>>>(g, field) ;
    check_CUDA_errors("scan_R_OP") ;
}

template<class OP, ScanKind Kind>
void scan_Z_OP(const Grid& g, Field<double>& field) {

    // Setup cuda blocks / threads and shared memory size
    int warps_per_point = (g.Nphi + 2*g.Nghost+31)/32 ;
    int threads_per_point = std::min(32*warps_per_point, 1024) ;
    int points_per_block = 1024/threads_per_point ;

    dim3 threads(threads_per_point,points_per_block,1) ;
    dim3 blocks(1,(g.NR+2*g.Nghost+points_per_block-1)/points_per_block,1) ;
    int mem_size = 1024*sizeof(double) ;

    scan_Z_generic_device<OP, Kind, double> <<<blocks, threads, mem_size>>>(g, field) ;
    check_CUDA_errors("scan_Z_OP") ;
}

void scan_R_sum(const Grid& g, Field<double>& field) {
    scan_R_OP<add2<double>, ScanKind::inclusive>(g, field) ;
}
void scan_R_mul(const Grid& g, Field<double>& field){
    scan_R_OP<mul2<double>, ScanKind::inclusive>(g, field) ;
}
void scan_R_min(const Grid& g, Field<double>& field) {
    scan_R_OP<min2<double>, ScanKind::inclusive>(g, field) ;
}
void scan_R_max(const Grid& g, Field<double>& field){
    scan_R_OP<max2<double>, ScanKind::inclusive>(g, field) ;
}

void scan_Z_sum(const Grid& g, Field<double>& field){
    scan_Z_OP<add2<double>, ScanKind::inclusive>(g, field) ;
}
void scan_Z_mul(const Grid& g, Field<double>& field){
    scan_Z_OP<mul2<double>, ScanKind::inclusive>(g, field) ;
}
void scan_Z_min(const Grid& g, Field<double>& field) {
    scan_Z_OP<min2<double>, ScanKind::inclusive>(g, field) ;
}
void scan_Z_max(const Grid& g, Field<double>& field){
    scan_Z_OP<max2<double>, ScanKind::inclusive>(g, field) ;
}

} // namespace Reduction
