/* cuda_array.h
 *
 *  This header provides a simple class for managing memory that exists on both 
 *  the CPU (host) and GPU (device). Since cudaMallocManaged is used there is no
 *  need to manually transfer memory between the host and device.
 */
#ifndef _CUDISC_CUDA_ARRAY_H_
#define _CUDISC_CUDA_ARRAY_H_

#include <memory>
#include <stdexcept>

#include <cuda_runtime.h>

#include "utils.h"

/* CudaArray_deleter
 * 
 *  Free memory from both host and device
 */ 
template <typename T>
struct CudaArray_deleter {
    void operator()(T* p) const {
      cudaDeviceSynchronize();
      cudaFree(p);

      check_CUDA_errors("CudaArray_deleter") ;
    }
};


/* CudaArray
 *
 * Managed pointer to memory that is shared between the host and the device
 */
template <typename T>
using CudaArray = std::unique_ptr<T[], CudaArray_deleter<T>>;

/* make_CudaArray
 *
 * Create an array of size n with space allocated on both the host and device
 * using cudaMallocManaged. 
 * 
 * The CUDA runtime will automaticly handle any copies between the host and device 
 * as needed.
 */
template <typename T>
CudaArray<T> make_CudaArray(std::size_t n) {

    // Always allocate 1 byte at least (c++ standard for operator new)
    std::size_t size = n*sizeof(T) ;
    if (size == 0)
        size = 1 ;

    T* ptr;
    cudaError_t status = cudaMallocManaged(&ptr, size);
    if (status != cudaSuccess) {
        throw std::runtime_error("CUDA Failed to allocate memory") ;
    }
    cudaDeviceSynchronize();
    return CudaArray<T>(ptr);
}

#endif//_CUDISC_CUDA_ARRAY_H_