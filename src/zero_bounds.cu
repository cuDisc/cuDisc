
#include <iostream>
#include <stdexcept>
#include <string>

#include "cuda_array.h"
#include "field.h"
#include "grid.h"
#include "utils.h"


void zero_boundaries_cpu(const Grid& g, Field<double>& f) {

    for (int i=0; i < g.NR + 2*g.Nghost; i++) 
        for (int j=0; j < g.Nghost; j++) {
            f[f.index(i,j)] = 0 ;
            f[f.index(i,j+g.Nphi+g.Nghost)] = 0 ;
        }

    for (int j=0; j < g.Nphi + 2*g.Nghost; j++) 
        for (int i=0; i < g.Nghost; i++) {
            f[f.index(i,j)] = 0 ;
            f[f.index(i+g.NR+g.Nghost,j)] = 0 ;
        }
}

void zero_boundaries_cpu(const Grid& g, Field3D<double>& f) {

    for (int i=0; i < g.NR + 2*g.Nghost; i++) 
        for (int j=0; j < g.Nghost; j++)
            for (int k=0; k < f.Nd; k++) {
                f[f.index(i,j,k)] = 0 ;
                f[f.index(i,j+g.Nphi+g.Nghost, k)] = 0 ;
            }

    for (int j=0; j < g.Nphi + 2*g.Nghost; j++) 
        for (int i=0; i < g.Nghost; i++) 
            for (int k=0; k < f.Nd; k++) {
                f[f.index(i,j,k)] = 0 ;
                f[f.index(i+g.NR+g.Nghost,j,k)] = 0 ;
            }
}


void zero_midplane_boundary_cpu(const Grid& g, Field<double>& f) {

    for (int i=0; i < g.NR + 2*g.Nghost; i++) 
        for (int j=0; j < g.Nghost; j++) {
            f[f.index(i,j)] = 0 ;
        }
}

void zero_midplane_boundary_cpu(const Grid& g, Field3D<double>& f) {

    for (int i=0; i < g.NR + 2*g.Nghost; i++) 
        for (int j=0; j < g.Nghost; j++)
            for (int k=0; k < f.Nd; k++) {
                f[f.index(i,j,k)] = 0 ;
            }
 }



namespace {

template<typename T>
__global__ void zero_boundary_Z_device(GridRef g, FieldRef<T> f) {
    int i = threadIdx.x + blockIdx.x * blockDim.x ;
    int j = threadIdx.y ; // Just 1 block in y

    if (j >= g.Nghost) j += g.Nphi ;

    if (i < g.NR + 2*g.Nghost && j < g.Nphi + 2*g.Nghost)
        f[f.index(i,j)] = 0 ;
}

template<typename T>
__global__ void zero_boundary_R_device(GridRef g, FieldRef<T> f) {
    int j = threadIdx.x + blockIdx.x * blockDim.x ;
    int i = threadIdx.y ; // Just 1 block in y

    if (i >= g.Nghost) i += g.NR ;

    if (i < g.NR + 2*g.Nghost && j < g.Nphi + 2*g.Nghost)
        f[f.index(i,j)] = 0 ;
}

template<typename T>
__global__ void zero_boundary_Z_3D_device(GridRef g, Field3DRef<T> f) {
    int k = threadIdx.x + blockIdx.x * blockDim.x ;
    int i = threadIdx.y + blockIdx.y * blockDim.y ;
    int j = threadIdx.z ;

    if (j >= g.Nghost) j += g.Nphi ;

    if (i < g.NR + 2*g.Nghost && j < g.Nphi + 2*g.Nghost && k < f.Nd)
        f[f.index(i,j,k)] = 0 ;
}

template<typename T>
__global__ void zero_boundary_R_3D_device(GridRef g, Field3DRef<T> f) {
    int k = threadIdx.x + blockIdx.x * blockDim.x ;
    int j = threadIdx.y + blockIdx.y * blockDim.y ;
    int i = threadIdx.z ;

    if (i >= g.Nghost) i += g.NR ;

    if (i < g.NR + 2*g.Nghost && j < g.Nphi + 2*g.Nghost && k < f.Nd)
        f[f.index(i,j,k)] = 0 ;
}

} //namespace

void zero_boundaries(const Grid& g, Field<double>& f) {

    if (g.Nghost > 512) { 
        std::string msg = 
            "zero_boundaries does not support Nghost > 512.\n"
            "Note the CPU version does not have this limitation";
        throw std::invalid_argument(msg) ;
    }   

    int nt = 512/g.Nghost ;
    dim3 threads(nt, 2*g.Nghost) ;

    dim3 block_r((g.Nphi + 2*g.Nghost+nt-1)/nt) ;
    dim3 block_z((g.NR + 2*g.Nghost+nt-1)/nt) ;

    zero_boundary_R_device<double><<<block_r, threads>>>(g, f) ;
    //check_CUDA_errors("zero_boundaries") ;
    zero_boundary_Z_device<double><<<block_z, threads>>>(g, f) ;
    check_CUDA_errors("zero_boundaries") ;
}


void zero_boundaries(const Grid& g, Field3D<double>& f) {

    if (g.Nghost > 512) { 
        std::string msg = 
            "zero_boundaries does not support Nghost > 512.\n"
            "Note the CPU version does not have this limitation";
        throw std::invalid_argument(msg) ;
    }   
     

    int nk = std::min(512/g.Nghost, f.Nd) ;
    int nt = 512/(g.Nghost*nk) ;

    dim3 threads(nk,nt, 2*g.Nghost) ;

    dim3 block_r((f.Nd+nk-1)/nk, (g.Nphi + 2*g.Nghost+nt-1)/nt) ;
    dim3 block_z((f.Nd+nk-1)/nk, (g.NR + 2*g.Nghost+nt-1)/nt);

    zero_boundary_R_3D_device<double><<<block_r, threads>>>(g, f) ;
    //check_CUDA_errors("zero_boundaries") ;
    zero_boundary_Z_3D_device<double><<<block_z, threads>>>(g, f) ;
    check_CUDA_errors("zero_boundaries") ;
}

void zero_midplane_boundary(const Grid& g, Field<double>& f) {

    if (g.Nghost > 1024) { 
        std::string msg = 
            "zero_midplane_boundary does not support Nghost > 1024.\n"
            "Note the CPU version does not have this limitation";
        throw std::invalid_argument(msg) ;
    }   

    int nt = 1024/g.Nghost ;

    dim3 threads(nt, g.Nghost) ;
    dim3 block_z((g.NR + 2*g.Nghost+nt-1)/nt) ;

    zero_boundary_Z_device<double><<<block_z, threads>>>(g, f) ;
    check_CUDA_errors("zero_boundaries") ;
}


void zero_midplane_boundary(const Grid& g, Field3D<double>& f) {

    if (g.Nghost > 1024) { 
        std::string msg = 
            "zero_midplane_boundary does not support Nghost > 1024.\n"
            "Note the CPU version does not have this limitation";
        throw std::invalid_argument(msg) ;
    }   
     

    int nk = std::min(1024/g.Nghost, f.Nd) ;
    int nt = 1024/(g.Nghost*nk) ;

    dim3 threads(nk,nt, g.Nghost) ;
    dim3 block_z((f.Nd+nk-1)/nk, (g.NR + 2*g.Nghost+nt-1)/nt);

    zero_boundary_Z_3D_device<double><<<block_z, threads>>>(g, f) ;
    check_CUDA_errors("zero_boundaries") ;
}


void set_all(Grid&g, Field<double>& f, double val) {

}



__global__ void set_all_device(GridRef g, 
                               FieldRef<double> f, double val) {

    int i = threadIdx.x + blockIdx.x*blockDim.x ;
    int step = blockDim.x * gridDim.x ;

    int size = (g.NR + 2*g.Nghost)*f.stride ;

    while (i < size) {
        f[i] = val ;
        i += step ;
    }
}

__global__ void set_all_device(GridRef g, 
                               Field3DRef<double> f, double val) {

    int i = threadIdx.x + blockIdx.x*blockDim.x ;
    int step = blockDim.x * gridDim.x ;

    int size = (g.NR + 2*g.Nghost)*f.stride_Zd ;

    while (i < size) {
        f[i] = val ;
        i += step ;
    }
}

void set_all(const Grid& g, Field<double>& f, double val) {

    int blocks = ((g.NR + 2*g.Nghost)*f.stride + 1023) / 1024 ;

    set_all_device<<<blocks, 1024>>>(g, f, val) ;
    check_CUDA_errors("set_all_device") ;
}

void set_all(const Grid& g, Field3D<double>& f, double val) {

    int blocks = ((g.NR + 2*g.Nghost)*f.stride_Zd + 1023) / 1024 ;

    set_all_device<<<blocks, 1024>>>(g, f, val) ;
    check_CUDA_errors("set_all_device") ;
}

void set_all_cpu(const Grid& g, Field<double>& f, double val) {

    int size = (g.NR + 2*g.Nghost)*f.stride ;

    for (int i=0; i < size; i++)
        f[i] = val ;
}

void set_all_cpu(const Grid& g, Field3D<double>& f, double val) {

    int size = (g.NR + 2*g.Nghost)*f.stride_Zd ;

    for (int i=0; i < size; i++)
        f[i] = val ;
}
