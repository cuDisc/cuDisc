
#include <cuda_runtime.h>
#include <fstream>
#include <filesystem>

#include <stdio.h>
#include "bins.h"
#include "utils.h"

__device__ __host__
inline 
void aggregate_data(int i, 
                    int n_in, const double* input,
                    int n_out, double* output, bool MEAN) {

    auto MIN = [](int i, int j) {
        if (i < j)
            return i ;
        else 
            return j ;
    } ;


    int n_sub = n_in / n_out ;
    int n_rem = n_in - n_out*n_sub ;

    if (i < n_out) {
        int l0 =   i  *n_sub + MIN( i , n_rem) ;
        int l1 = (i+1)*n_sub + MIN(i+1, n_rem) ;
        
        double val = 0 ;
        for (int l=l0; l < l1; l++)
            val += input[l] ;

        if (MEAN)
            val /= (l1 - l0) ;

        output[i] = val ;
        l0 = l1 ;
    }
}
__device__ __host__
inline 
void aggregate_planck(int i, 
                    int n_in, const double* input, const double* wle_in,
                    int n_out, double* output, 
                    const PlanckIntegralRef& planck, const double T) {

    auto MIN = [](int i, int j) {
        if (i < j)
            return i ;
        else 
            return j ;
    } ;

    auto MAX = [](double i, double j) {
        if (i < j)
            return j ;
        else 
            return i ;
    } ;


    int n_sub = (n_in-1) / n_out ;
    int n_rem = (n_in-1) - n_out*n_sub ;

    if (i < n_out) {
        int l0 =   i  *n_sub + MIN( i , n_rem) ;
        int l1 = (i+1)*n_sub + MIN(i+1, n_rem) ;
        double val1 = 0 ;
        double val2 = 0 ;
        
        for (int l=l0; l < l1; l++) {
            double B = MAX(planck_factor(planck, T, l, n_in, wle_in), 1e-100);
            val1 += B*input[l] ;
            val2 += B ;
        }
        output[i] = val1/val2 ;

        l0 = l1 ;
    }
}

CudaArray<double> WavelengthBinner::bin_data(const CudaArray<double>& input,
                                             int mode) const {
    
    CudaArray<double> result = make_CudaArray<double>(num_bands) ;

    for (int i=0; i < num_bands; i++)
        aggregate_data(i, 
            _nwle_in, input.get(), num_bands, result.get(), mode == MEAN) ;

    return result ;
}

__global__ void _bin_field(GridRef g, 
                           Field3DConstRef<double> input, Field3DRef<double> output,
                           bool MEAN) {

    int k = threadIdx.x + blockIdx.x*blockDim.x ;
    int j = threadIdx.y + blockIdx.y*blockDim.y ;
    int i = threadIdx.z + blockIdx.z*blockDim.z ;

    if (i < g.NR + 2*g.Nghost && j < g.Nphi + 2*g.Nghost && k < output.Nd) {
        aggregate_data(k, 
                       input.Nd, input.get() + input.index(i,j,0), 
                       output.Nd, output.get() + output.index(i,j,0),
                       MEAN) ;
    }
}

Field3D<double> WavelengthBinner::bin_field(const Grid& g, const Field3D<double>& input,
                                            int mode) const {

    Field3D<double> result = create_field3D<double>(g, num_bands) ;

    dim3 threads(4,16,16) ;
    dim3 blocks((num_bands+3)/4, (g.Nphi+2*g.Nghost+15)/16, (g.NR+2*g.Nghost+15)/16) ;

    _bin_field<<<blocks, threads>>>(g, input, result, mode == MEAN) ;
    check_CUDA_errors("_bin_field") ;

    return result ;
}

CudaArray<double> WavelengthBinner::bin_planck_data(const CudaArray<double>& input,
                                             const double T) const {
    
    CudaArray<double> result = make_CudaArray<double>(num_bands) ;

    for (int i=0; i < num_bands; i++) {
        //printf("band %d:\n", i);
        aggregate_planck(i, 
            _nwle_in, input.get(), _wle_in_e.get(), num_bands, result.get(), _planck, T) ;
    }

    return result ;
}

__global__ void _bin_planck(GridRef g, 
                           Field3DConstRef<double> input, Field3DRef<double> output,
                           PlanckIntegralRef planck, FieldConstRef<double> T, const double* wle) {

    int k = threadIdx.x + blockIdx.x*blockDim.x ;
    int j = threadIdx.y + blockIdx.y*blockDim.y ;
    int i = threadIdx.z + blockIdx.z*blockDim.z ;

    if (i < g.NR + 2*g.Nghost && j < g.Nphi + 2*g.Nghost && k < output.Nd) {
        aggregate_planck(k, 
                       input.Nd, input.get() + input.index(i,j,0), wle,
                       output.Nd, output.get() + output.index(i,j,0),
                       planck, T[T.index(i,j)]) ;
    }
}

Field3D<double> WavelengthBinner::bin_planck(const Grid& g, const Field3D<double>& input,
                                            const Field<double>& T) const {

    Field3D<double> result = create_field3D<double>(g, num_bands) ;

    dim3 threads(4,16,16) ;
    dim3 blocks((num_bands+3)/4, (g.Nphi+2*g.Nghost+15)/16, (g.NR+2*g.Nghost+15)/16) ;

    _bin_planck<<<blocks, threads>>>(g, input, result, _planck, T, _wle_in_e.get()) ;
    check_CUDA_errors("_bin_planck") ;

    return result ;
}

Field<double> WavelengthBinner::planck_mean(const Grid& g, const Field3D<double>& input,
                                             const Field<double>& T) const {

    Field<double> result = create_field<double>(g) ;

    dim3 threads(4,16,16) ;
    dim3 blocks((num_bands+3)/4, (g.Nphi+2*g.Nghost+15)/16, (g.NR+2*g.Nghost+15)/16) ;

    _bin_planck<<<blocks, threads>>>(g, input, Field3DRef<double>(result), _planck, T, _wle_in_e.get()) ;
    check_CUDA_errors("_bin_planck") ;

    return result ;
}

void WavelengthBinner::write_wle(std::filesystem::path folder) const {

        std::ofstream f(folder / ("wle_grid.dat"), std::ios::binary);

        int Nlam = _nwle_in, Nbands = num_bands;
        
        f.write((char*) &Nlam, sizeof(int));
        f.write((char*) &Nbands, sizeof(int));

        for (int i=0; i < Nlam; i++) {  
            double lam = _wle_in_c[i];
            f.write((char*) &lam, sizeof(double));
        }

        f.close();
    }