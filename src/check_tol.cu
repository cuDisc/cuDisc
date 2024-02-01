
#include <float.h>

#include "cuda_array.h"
#include "pcg_solver.h"
#include "reductions.h"
#include "utils.h"


// Assumptions:
//    blockDim.x must be 1024
__global__ void __max_block_wise(int size, const double* x, double* max_block) {
    
    const unsigned int idx = threadIdx.x ; 

    extern __shared__ double ptr[] ;

    // Get the max of all the elements for each thread
    int i = idx + blockIdx.x * blockDim.x ;
    double val = -DBL_MAX;
    while (i < size) {
        val = max(val, x[i]) ;
        i += blockDim.x * gridDim.x ;
    } ;
    ptr[idx] = val ;

    // Collect the values from each warp
    const unsigned int lane = idx & 31; // index of thread in warp (0..31)
    __syncthreads();
    
    if (lane < 16) ptr[idx] = max(ptr[idx + 16], ptr[idx]);
    if (lane <  8) ptr[idx] = max(ptr[idx +  8], ptr[idx]);
    if (lane <  4) ptr[idx] = max(ptr[idx +  4], ptr[idx]);
    if (lane <  2) ptr[idx] = max(ptr[idx +  2], ptr[idx]);
    if (lane <  1) ptr[idx] = max(ptr[idx +  1], ptr[idx]);

    __syncthreads() ;

    // Collect values accross the 32 warps
    if (idx < 32) {
        ptr[idx] = ptr[idx*32] ;

        if (lane < 16) ptr[idx] = max(ptr[idx + 16], ptr[idx]);
        if (lane <  8) ptr[idx] = max(ptr[idx +  8], ptr[idx]);
        if (lane <  4) ptr[idx] = max(ptr[idx +  4], ptr[idx]);
        if (lane <  2) ptr[idx] = max(ptr[idx +  2], ptr[idx]);
        if (lane <  1) ptr[idx] = max(ptr[idx +  1], ptr[idx]);
    }
    // Store the result
    if (idx == 0) max_block[blockIdx.x] = ptr[0] ;

}


double get_max_value(const DnVec& x) {

    int n_per_block = 4*1024 ;
    int threads = 1024 ;
    int blocks = (x.rows + n_per_block-1)/n_per_block ;
    int memsize = threads*sizeof(double) ;

    CudaArray<double> max_elem = make_CudaArray<double>(blocks) ;

    __max_block_wise<<<blocks, threads,memsize>>>(x.rows, x.data.get(), max_elem.get()) ;
    check_CUDA_errors("__max_block_wise") ;

    // Get the max over the final array:
    double final_max = max_elem[0] ;
    for (int i=1; i < blocks; i++) 
        final_max = std::max(final_max, max_elem[i]) ;

    return final_max ;
}


__global__ void __compute_error(
    DnVecConstRef x, DnVecConstRef res,  DnVecConstRef rhs, DnVecRef err) {
        
    int i = threadIdx.x + blockIdx.x*blockDim.x ;
    int step = blockDim.x * gridDim.x ;
        
    while (i < x.rows) {
        err.data[i] = fabs(res.data[i]) / (max(fabs(rhs.data[i]), fabs(x.data[i])) + 1e-300) ;
        i += step ;
    }
}


bool CheckResidual::operator()(const DnVec& x, const DnVec& residual, const DnVec& rhs) const {
              
    if (x.rows != err.rows)
        err = DnVec(x.rows) ;


    int blocks = (x.rows + 1023)/1024;
    __compute_error<<<blocks, 1024>>>(x,residual, rhs, err) ;
    check_CUDA_errors("__compute_error") ;

    return get_max_value(err) < _tol ;
}


__global__ void __compute_error_blocks(
    DnVecConstRef x, DnVecConstRef res,  DnVecConstRef rhs, DnVecRef err, 
    double tol1, double tol2, int block_size) {
        
    int i = threadIdx.x + blockIdx.x*blockDim.x ;
    int step = blockDim.x * gridDim.x ;
        
    while (i < x.rows) {
        double tmp  = fabs(res.data[i]) / (max(fabs(rhs.data[i]), fabs(x.data[i])) + 1e-300) ;
        if (i % block_size == 0)
            err.data[i] = tmp / tol1 ;
        else 
            err.data[i] = tmp / tol2 ;
        i += step ;
    }
}

bool CheckTemperatureResidual::operator()(const DnVec& x, const DnVec& residual, const DnVec& rhs) const {
         
    if (x.rows != err.rows)
        err = DnVec(x.rows) ;

    int blocks = (x.rows + 1023)/1024;
    __compute_error_blocks<<<blocks, 1024>>>(x,residual, rhs, err, _tol, _tolJ, _num_vars) ;
    check_CUDA_errors("__compute_error_blocks") ;

    return get_max_value(err) < 1 ;
}


