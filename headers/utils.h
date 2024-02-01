
#ifndef _CUDISC_UTILS_H_
#define _CUDISC_UTILS_H_

#include <cstdio>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

class Grid;
template<typename T> class Field;
template<typename T> class Field3D;

/* check_CUDA_errors
 * 
 * Checks that the last cuda operation succeeded. If not, it raises a 
 * runtime_error.
 */
inline void check_CUDA_errors(std::string fn_name) {
  cudaDeviceSynchronize();
  
  cudaError_t cudaError = cudaGetLastError();
  if (cudaError != cudaSuccess) {
    const char *error = cudaGetErrorString(cudaError);
    
    throw std::runtime_error("CUDA error in " + fn_name + 
                             + ": " + std::string(error) + "\n") ;
  }
}

void copy_field(const Grid&, const Field<double>&, Field<double>&) ;
void copy_field(const Grid&, const Field3D<double>&, Field3D<double>&) ;

void copy_field_cpu(const Grid&, const Field<double>&, Field<double>&) ;
void copy_field_cpu(const Grid&, const Field3D<double>&, Field3D<double>&) ;

void zero_boundaries(const Grid& g, Field<double>& f) ;
void zero_boundaries(const Grid& g, Field3D<double>& f) ;

void zero_boundaries_cpu(const Grid& g, Field<double>& f) ;
void zero_boundaries_cpu(const Grid& g, Field3D<double>& f) ;

void zero_midplane_boundary(const Grid& g, Field<double>& f) ;
void zero_midplane_boundary(const Grid& g, Field3D<double>& f) ;

void zero_midplane_boundary_cpu(const Grid& g, Field<double>& f) ;
void zero_midplane_boundary_cpu(const Grid& g, Field3D<double>& f) ;

void set_all(const Grid& g, Field<double>& f, double val) ;
void set_all(const Grid& g, Field3D<double>& f, double val) ;

void set_all_cpu(const Grid& g, Field<double>& f, double val) ;
void set_all_cpu(const Grid& g, Field3D<double>& f, double val) ;


inline void check_CUDA_devices() {

  int nDevices;
  cudaGetDeviceCount(&nDevices);
  
  printf("Number of devices: %d\n", nDevices);
  
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (MHz): %d\n",
           prop.memoryClockRate/1024);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %.1f\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    printf("  Total global memory (Gbytes) %.1f\n",(float)(prop.totalGlobalMem)/1024.0/1024.0/1024.0);
    printf("  Shared memory per block (Kbytes) %.1f\n",(float)(prop.sharedMemPerBlock)/1024.0);
    printf("  minor-major: %d-%d\n", prop.minor, prop.major);
    printf("  Warp-size: %d\n", prop.warpSize);
    printf("  Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
    printf("  Concurrent computation/communication: %s\n\n",prop.deviceOverlap ? "yes" : "no");
  }
  fflush(stdout) ;
}


#endif//_CUDISC_UTILS_H_
