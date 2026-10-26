#include <algorithm>
#include <cmath>

#include "errorfuncs.h"
#include "utils.h"

namespace {

const int threads = 256 ;
const int max_blocks = 1024 ;

// Sum of the values held by each thread of the block; requires blockDim.x == threads
__device__ double _block_sum(double val) {
    __shared__ double s[threads] ;
    s[threadIdx.x] = val ;
    __syncthreads() ;
    for (int k=threads/2; k>0; k/=2) {
        if (threadIdx.x < k)
            s[threadIdx.x] += s[threadIdx.x + k] ;
        __syncthreads() ;
    }
    return s[0] ;
}

// Per-block partial sums of |T2-T1|/T1 (relative) or (T2-T1)^2
template<bool relative>
__global__ void _err_partial(GridRef g, FieldConstRef<double> T1, FieldConstRef<double> T2, double* partial) {

    int NZ = g.Nphi + 2*g.Nghost ;
    int N = (g.NR + 2*g.Nghost) * NZ ;

    double sum = 0 ;
    for (int n=threadIdx.x + blockIdx.x*blockDim.x; n<N; n+=blockDim.x*gridDim.x) {
        int i = n / NZ, j = n % NZ ;
        double d = T2(i,j) - T1(i,j) ;
        sum += relative ? fabs(d) / T1(i,j) : d*d ;
    }
    sum = _block_sum(sum) ;
    if (threadIdx.x == 0)
        partial[blockIdx.x] = sum ;
}

// Sum the partial sums in a single block, storing the result in partial[n]
__global__ void _sum_partials(double* partial, int n) {
    double sum = 0 ;
    for (int k=threadIdx.x; k<n; k+=blockDim.x)
        sum += partial[k] ;
    sum = _block_sum(sum) ;
    if (threadIdx.x == 0)
        partial[n] = sum ;
}

template<bool relative>
double _err_sum(Grid& g, Field<double>& T1, Field<double>& T2) {

    int N = (g.NR + 2*g.Nghost) * (g.Nphi + 2*g.Nghost) ;
    int blocks = std::min(max_blocks, (N + threads - 1) / threads) ;

    // Plain device memory: the result is copied back without touching any
    // managed memory on the host.
    double* partial ;
    if (cudaMalloc(&partial, (blocks + 1)*sizeof(double)) != cudaSuccess)
        check_CUDA_errors("errorfuncs: cudaMalloc") ;

    _err_partial<relative><<<blocks, threads>>>(g, T1, T2, partial) ;
    _sum_partials<<<1, threads>>>(partial, blocks) ;

    double sum = 0 ;
    (void) cudaMemcpy(&sum, partial + blocks, sizeof(double), cudaMemcpyDeviceToHost) ;
    (void) cudaFree(partial) ;
    check_CUDA_errors("errorfuncs") ;

    return sum / N ;
}

} // namespace

double L2err(Grid& g, Field<double> &T1, Field<double> &T2) {
    return std::sqrt(_err_sum<false>(g, T1, T2)) ;
}

double fracerr(Grid& g, Field<double> &T1, Field<double> &T2) {
    return _err_sum<true>(g, T1, T2) ;
}

double L2err_cpu(Grid& g, Field<double> &T1, Field<double> &T2) {

    double sum=0;
    int N = (g.NR + 2*g.Nghost)*(g.Nphi + 2*g.Nghost);

    for (int i=0; i<g.NR + 2*g.Nghost; i++) {
        for (int j=0; j<g.Nphi + 2*g.Nghost; j++) {
            sum += (T2(i,j) - T1(i,j))*(T2(i,j) - T1(i,j));
        }
    }

    return std::sqrt(sum/N);
}

double fracerr_cpu(Grid& g, Field<double> &T1, Field<double> &T2) {

    double sum=0;
    int N = (g.NR + 2*g.Nghost)*(g.Nphi + 2*g.Nghost);

    for (int i=0; i<g.NR + 2*g.Nghost; i++) {
        for (int j=0; j<g.Nphi + 2*g.Nghost; j++) {
            sum += std::abs(T2(i,j) - T1(i,j))/T1(i,j);
        }
    }

    return sum/N;
}
