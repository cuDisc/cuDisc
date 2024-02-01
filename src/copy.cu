
#include "grid.h"
#include "utils.h"

__global__ void copy_field_device(GridRef g, FieldConstRef<double> in, 
                                  FieldRef<double> out) {

    int i = threadIdx.x + blockIdx.x*blockDim.x ;
    int step = blockDim.x * gridDim.x ;

    int size = (g.NR + 2*g.Nghost)*in.stride ;

    while (i < size) {
        out[i] = in[i] ;
        i += step ;
    }
}

__global__ void copy_field3D_device(GridRef g, Field3DConstRef<double> in, 
                                    Field3DRef<double> out) {

    int i = threadIdx.x + blockIdx.x*blockDim.x ;
    int step = blockDim.x * gridDim.x ;

    int size = (g.NR + 2*g.Nghost)*in.stride_Zd ;

    while (i < size) {
        out[i] = in[i] ;
        i += step ;
    }
}

void copy_field(const Grid& g, const Field<double>& in, Field<double>& out) {

    int blocks = ((g.NR + 2*g.Nghost)*in.stride + 1023) / 1024 ;

    copy_field_device<<<blocks, 1024>>>(g, in, out) ;
    check_CUDA_errors("copy_field") ;
}

void copy_field(const Grid& g, const Field3D<double>& in, Field3D<double>& out) {

    int blocks = ((g.NR + 2*g.Nghost)*in.stride_Zd + 1023) / 1024 ;

    copy_field3D_device<<<blocks, 1024>>>(g, in, out) ;
    check_CUDA_errors("copy_field") ;
}

void copy_field_cpu(const Grid& g, const Field<double>& in, Field<double>& out) {
    int size = (g.NR + 2*g.Nghost)*in.stride * sizeof(double) ;
    memcpy(out.get().get(), in.get().get(), size) ;
}

void copy_field_cpu(const Grid& g, const Field3D<double>& in, Field3D<double>& out) {

    int size = (g.NR + 2*g.Nghost)*in.stride_Zd * sizeof(double) ;
    memcpy(out.get().get(), in.get().get(), size) ;
}