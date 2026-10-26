/* cuda_array.h
 *
 *  This header provides a simple class for managing memory that exists on both
 *  the CPU (host) and GPU (device). Since cudaMallocManaged is used there is no
 *  need to manually transfer memory between the host and device.
 */
#ifndef _CUDISC_CUDA_ARRAY_H_
#define _CUDISC_CUDA_ARRAY_H_

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "utils.h"

/* CudaMemoryCache
 *
 * Freed arrays are kept and handed out again to later allocations of the same
 * size, rather than returned to the driver. Allocating managed memory, first
 * touching it on the GPU (page faults) and freeing it costs a few ms for a
 * large field, and many routines create the same temporary fields every call.
 *
 * Reused blocks are zeroed, like freshly allocated managed memory.
 * Set CUDISC_NO_MEMORY_CACHE=1 to disable the cache.
 */
namespace CudaMemoryCache {
    struct Cache {
        std::mutex lock ;
        std::unordered_map<std::size_t, std::vector<void*>> blocks ;
        std::size_t bytes = 0 ;
        bool enabled ;

        Cache() {
            const char* env = std::getenv("CUDISC_NO_MEMORY_CACHE") ;
            enabled = not (env && *env && *env != '0') ;
        }
    } ;

    // Never destroyed (a static destructor could run after the CUDA runtime has
    // shut down); cached blocks are freed by release_at_exit instead.
    inline Cache& get() {
        static Cache* cache = new Cache ;
        return *cache ;
    }

    inline void* take(std::size_t size) {
        Cache& c = get() ;
        std::lock_guard<std::mutex> guard(c.lock) ;
        auto it = c.blocks.find(size) ;
        if (it == c.blocks.end() || it->second.empty())
            return nullptr ;
        void* p = it->second.back() ;
        it->second.pop_back() ;
        c.bytes -= size ;
        return p ;
    }

    // Free all cached blocks; returns the number of bytes freed
    inline std::size_t free_all() {
        Cache& c = get() ;
        std::lock_guard<std::mutex> guard(c.lock) ;
        for (auto& entry : c.blocks)
            for (void* p : entry.second)
                (void) cudaFree(p);
        std::size_t freed = c.bytes ;
        c.blocks.clear() ;
        c.bytes = 0 ;
        return freed ;
    }

    // Free the cache at exit so leak checkers only report real leaks. It is
    // registered once the CUDA runtime is up, so it runs before the runtime's
    // own shutdown; errors are ignored as nothing can be done about them.
    inline void release_at_exit() {
        free_all() ;
        (void) cudaGetLastError();
    }

    // Returns false if the block should be freed instead
    inline bool give(void* p, std::size_t size) {
        Cache& c = get() ;
        if (not c.enabled || size == 0)
            return false ;
        static bool registered = (std::atexit(release_at_exit) == 0) ;
        (void) registered ;
        std::lock_guard<std::mutex> guard(c.lock) ;
        c.blocks[size].push_back(p) ;
        c.bytes += size ;
        return true ;
    }
}

/* clear_cuda_memory_cache
 *
 * Free all cached blocks, returning the memory to the driver. Returns the
 * number of bytes freed. Arrays still in use are not affected.
 *
 * Must be called before anything that destroys the CUDA context (e.g.
 * cudaDeviceReset), since cached blocks would otherwise outlive it and could
 * be handed out again. As with live arrays, cuDisc never does this itself.
 */
inline std::size_t clear_cuda_memory_cache() {
    (void) cudaDeviceSynchronize();
    std::size_t freed = CudaMemoryCache::free_all() ;
    check_CUDA_errors("clear_cuda_memory_cache") ;
    return freed ;
}

/* cuda_memory_cache_size
 *
 * Number of bytes currently held in the cache.
 */
inline std::size_t cuda_memory_cache_size() {
    CudaMemoryCache::Cache& c = CudaMemoryCache::get() ;
    std::lock_guard<std::mutex> guard(c.lock) ;
    return c.bytes ;
}

/* CudaArray_deleter
 *
 *  Return memory to the cache, or free it from both host and device.
 *
 *  No synchronisation is needed: all work runs on the default stream, so a
 *  later kernel using a cached block is ordered after any still reading it,
 *  make_CudaArray synchronises before handing a reused block to the host, and
 *  cudaFree synchronises itself.
 */
template <typename T>
struct CudaArray_deleter {
    std::size_t size = 0 ;

    void operator()(T* p) const {
      if (CudaMemoryCache::give(p, size))
          return ;

      // Called from a noexcept destructor, so report rather than throw. Errors
      // from earlier kernels are left for the next check_CUDA_errors.
      cudaError_t status = cudaFree(p);
      if (status != cudaSuccess && status != cudaErrorCudartUnloading)
          std::fprintf(stderr, "CudaArray_deleter: cudaFree failed: %s\n",
                       cudaGetErrorString(status)) ;
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
 * using cudaMallocManaged, or reused from the cache.
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

    // Managed memory is allocated in 2 MiB pages, so rounding large requests
    // up costs nothing and lets similar sizes (e.g. staggered fields) share.
    const std::size_t page = 2 << 20 ;
    if (size > page / 2)
        size = page * ((size + page - 1) / page) ;

    T* ptr = static_cast<T*>(CudaMemoryCache::take(size)) ;
    if (ptr) {
        // The memset is asynchronous, so wait before the host can touch it
        (void) cudaMemset(ptr, 0, size);
        (void) cudaDeviceSynchronize();
    }
    else {
        cudaError_t status = cudaMallocManaged(&ptr, size);
        if (status != cudaSuccess) {
            // Release cached blocks and try again before giving up
            (void) cudaGetLastError();
            clear_cuda_memory_cache() ;
            status = cudaMallocManaged(&ptr, size);
        }
        if (status != cudaSuccess) {
            throw std::runtime_error("CUDA Failed to allocate memory") ;
        }
    }
    return CudaArray<T>(ptr, CudaArray_deleter<T>{size});
}

#endif//_CUDISC_CUDA_ARRAY_H_
