
#ifndef _CUDISC_SCAN_H_
#define _CUDISC_SCAN_H_

template<typename T>
struct add2 {
    __host__ __device__ static T apply(T a, T b)  { return a + b ; }
    __host__ __device__ static T identity() { return 0 ; } ;
} ;

template<typename T>
struct mul2 {
    __host__ __device__ static T apply(T a, T b)  { return a * b ; }
    __host__ __device__ static T identity() { return 1 ; } ;
} ;

template<typename T>
struct min2 {
    __host__ __device__ static T apply(T a, T b)  { return a < b ? a : b ; }
    __host__ __device__ static T identity() { return 1e308 ; } ;
} ;

template<typename T>
struct max2 {
    __host__ __device__ static T apply(T a, T b)  { return a > b ? a : b ; }
    __host__ __device__ static T identity() { return -1e308 ; } ;
} ;


  
enum class ScanKind {
    inclusive, exclusive
} ;

/* scan_warp
 * 
 * Parallel scan of one warp of data (32 elements). 
 *
 * Reference: Sengupta, Harris, & Garland
 */
template <class OP, ScanKind Kind, class T>
__device__ T scan_warp(volatile T *ptr) {
    const unsigned int idx=threadIdx.x;
    const unsigned int lane = idx & 31; // index of thread in warp (0..31)
    T v = ptr[idx] ;
    if (lane >=  1) v = OP::apply(ptr[idx -  1], ptr[idx]); __syncwarp() ;
    ptr[idx] = v; __syncwarp() ;
    if (lane >=  2) v = OP::apply(ptr[idx -  2], ptr[idx]); __syncwarp() ; 
    ptr[idx] = v; __syncwarp() ;
    if (lane >=   4) v = OP::apply(ptr[idx -  4], ptr[idx]); __syncwarp() ; 
    ptr[idx] = v; __syncwarp() ;
    if (lane >=  8) v = OP::apply(ptr[idx -  8], ptr[idx]); __syncwarp() ; 
    ptr[idx] = v; __syncwarp() ;
    if (lane >= 16) v = OP::apply(ptr[idx -  16], ptr[idx]); __syncwarp() ; 
    ptr[idx] = v; __syncwarp() ;
    
    if (Kind == ScanKind::inclusive) 
        return  ptr[idx];
    else                             
        return (lane >0) ? ptr[idx -1] : OP::identity();
}

/* scan_block
 * 
 * Parallel scan of one block of data (up to 1024 elements). 
 *
 * Reference: Sengupta, Harris, & Garland
 */
template <class OP, ScanKind Kind, class T>
__device__ T scan_block(volatile T *ptr) {
    const unsigned int idx=threadIdx.x;
    const unsigned int lane = idx & 31;
    const unsigned int warpid = idx  >> 5;
    
    // Step 1: Intra-warp scan in each warp
    T val = scan_warp <OP,Kind>(ptr);
    __syncthreads();
    
    // Step 2: Collect per-warp partial results
    if (lane == 31) ptr[warpid] = ptr[idx];
    __syncthreads();
    
    // Step 3: Use 1st warp to scan per-warp results
    if(warpid == 0) scan_warp <OP,ScanKind::inclusive>(ptr);
    __syncthreads();
    
    // Step 4:  Accumulate  results  from  Steps 1 and 3
    if (warpid  > 0) val = OP::apply(ptr[warpid -1], val);
    __syncthreads();
    
    // Step 5: Write and return the final result
    ptr[idx] = val;
    __syncthreads();
    
    return  val;
}


#endif//_CUDISC_SCAN_H_