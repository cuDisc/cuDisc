#include <cmath>

#include <cublas_v2.h>

#include "field.h"
#include "grid.h"
#include "matrix_utils.h"
#include "reductions.h"
#include "utils.h"

namespace Reduction {

double sum_abs(const CudaArray<double>& x, int n) {
    double result = 0 ;
    if (cublasDasum(CublasHandle::get(), n, x.get(), 1, &result) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("Reduction::sum_abs: cublasDasum failed") ;
    check_CUDA_errors("Reduction::sum_abs") ;
    return result ;
}

double norm2(const CudaArray<double>& x, int n) {
    double result = 0 ;
    if (cublasDnrm2(CublasHandle::get(), n, x.get(), 1, &result) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("Reduction::norm2: cublasDnrm2 failed") ;
    check_CUDA_errors("Reduction::norm2") ;
    return result ;
}

// Each group of 32 threads (threadIdx.x) sums the species of one cell, so
// blockDim must be (32, sum_species_cells). The tree sum uses shared memory
// rather than warp shuffles so that it does not depend on the warp size.
const int sum_species_cells = 8 ;

__global__ void _sum_species(GridRef g, Field3DConstRef<double> in, FieldRef<double> out) {

    __shared__ double buf[sum_species_cells][32] ;

    int NZ = g.Nphi + 2*g.Nghost ;
    int ncell = (g.NR + 2*g.Nghost) * NZ ;

    // The loop bounds are the same for every thread in the block, as all of
    // them must reach the __syncthreads.
    for (int c0=blockIdx.x*blockDim.y; c0<ncell; c0+=blockDim.y*gridDim.x) {
        int c = c0 + threadIdx.y ;
        int i = c / NZ, j = c % NZ ;

        double sum = 0 ;
        if (c < ncell)
            for (int k=threadIdx.x; k<in.Nd; k+=blockDim.x)
                sum += in(i,j,k) ;
        buf[threadIdx.y][threadIdx.x] = sum ;
        __syncthreads() ;

        for (int n=16; n>0; n/=2) {
            if (threadIdx.x < n)
                buf[threadIdx.y][threadIdx.x] += buf[threadIdx.y][threadIdx.x + n] ;
            __syncthreads() ;
        }
        if (c < ncell && threadIdx.x == 0)
            out(i,j) = buf[threadIdx.y][0] ;
        __syncthreads() ;
    }
}

void sum_species(const Grid& g, const Field3D<double>& in, Field<double>& out) {

    int ncell = (g.NR + 2*g.Nghost) * (g.Nphi + 2*g.Nghost) ;
    dim3 threads(32, sum_species_cells) ;
    int blocks = (ncell + sum_species_cells - 1) / sum_species_cells ;

    _sum_species<<<blocks, threads>>>(g, in, out) ;
    check_CUDA_errors("Reduction::sum_species") ;
}

} // namespace Reduction
