
#include "cuda_array.h"
#include "field.h"
#include "grid.h"
#include "utils.h"

#include "reductions.h"

void Reduction::volume_integrate_Z_cpu(const Grid& g, const Field<double>& f, CudaArray<double>& result) {

    int stride = f.stride ;

    for (int i = 0; i < g.NR + 2*g.Nghost; i++) {
        double tot = 0 ;
        for (int j = 0; j < g.Nphi + 2*g.Nghost; j++)
            tot += g.volume(i,j) * f[i*stride + j] ;

        result[i] = tot ;
    }
}


namespace {
/* Parallel volume integral over z-direction.
 *
 * Notes:
 *  - This code requires globalDim.x = 1, i.e. one thread block is used to
      sum over j.
 */
__global__ void vol_integ_Z_device(GridRef g, FieldConstRef<double> f, double* result) {
    
    int tid = threadIdx.x + threadIdx.y*blockDim.x ;

    extern __shared__ double tmp[] ;
    tmp[tid] = 0 ;

    // Step 1: Serial collection over j for each thread
    int i = threadIdx.y + blockIdx.y * blockDim.y ;
    int j = threadIdx.x ;
    while (i < g.NR + 2*g.Nghost && j < g.Nphi + 2*g.Nghost) {
        tmp[tid] += g.volume(i,j) * f[i*f.stride + j] ;
        j += blockDim.x ;
    }
    __syncthreads() ;

    // Step 2: Parallel reduction of j
    j = blockDim.x / 2 ;
    while (j != 0) {
        if (threadIdx.x < j) tmp[tid] += tmp[tid + j] ;
        __syncthreads() ;
        j /= 2 ;
    }
    if (threadIdx.x == 0 && i < g.NR + 2*g.Nghost)
        result[i] = tmp[threadIdx.y*blockDim.x] ;
}

} // namespace

void Reduction::volume_integrate_Z(const Grid& g, const Field<double>& in, CudaArray<double>& result) {

    // Setup cuda blocks / threads and shared memory size
    dim3 threads(64,16,1) ;
    dim3 blocks(1,(g.NR+2*g.Nghost+15)/16,1) ;
    int mem_size = 16*64*sizeof(double) ;

    vol_integ_Z_device<<<blocks, threads, mem_size>>>(g, in, result.get()) ;
    check_CUDA_errors("volume_integrate_Z") ;
}