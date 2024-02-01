
#ifndef _CUDISC_HEADERS_BINS_H_
#define _CUDISC_HEADERS_BINS_H_

#include <stdio.h>
#include <cassert>
#include <cmath>

#include "cuda_array.h"
#include "field.h"
#include "grid.h"
#include "planck.h"

/* class WavelengthBinner
 * 
 * Creates a grid of reduced wavelengths from a set of input 
 * wavelengths and provides a series of ultility functions to bin
 * data from the input grid to the reduced grid.
 */
class WavelengthBinner {

  public:

    template<class T>
    WavelengthBinner(int nwle, const T& wle_in, int num_bands) 
      : num_bands(num_bands),
        bands(make_CudaArray<double>(num_bands)),
        edges(make_CudaArray<double>(num_bands-1)),
        _nwle_in(nwle),
        _wle_in_e(make_CudaArray<double>(nwle-1)),
        _wle_in_c(make_CudaArray<double>(nwle))
    { 
        for (int i=0; i < nwle; i++) {
            _wle_in_c[i] = wle_in[i];
        }

        int n_sub = (nwle-1) / num_bands ;
        int n_rem = (nwle-1) - num_bands*n_sub ;
        
        int l = 0; 
        for (int i=0; i < num_bands-1; i++) {
            //printf("wavelength l: %d\n", l);
            l += n_sub + (i < n_rem) ;
            edges[i] = std::sqrt(wle_in[l] * wle_in[l+1]) ;
        }
        assert(l + n_sub == nwle-1) ; 
        
        // Estimates for the bin centres.
        for (int i=1; i < num_bands-1; i++) 
            bands[i] = sqrt(edges[i-1]*edges[i]) ;

        bands[0] = edges[0] * edges[0] / bands[1] ;
        bands[num_bands-1] = edges[num_bands-2] * edges[num_bands-2] / bands[num_bands-2] ;

        // Store the edges of the old wavelength bins
        for (int i=0; i < nwle-1; i++) 
            _wle_in_e[i] = std::sqrt(wle_in[i]*wle_in[i+1]) ;
        
    }

    int num_bands ;
    CudaArray<double> bands, edges ;

    CudaArray<double> bin_data(const CudaArray<double>& input,
                              int mode) const ;

    Field3D<double> bin_field(const Grid& g, const Field3D<double>& input,
                              int mode) const ;

    Field3D<double> bin_planck(const Grid& g, const Field3D<double>& input,
                              const Field<double>& T) const ;

    CudaArray<double> bin_planck_data(const CudaArray<double>& input,
                                      const double T) const ;

    
    
    void write_wle(std::filesystem::path folder) const ;

    static const int SUM = 0 ;
    static const int MEAN = 1 ;

  private:
    int _nwle_in ;
    CudaArray<double> _wle_in_e ;
    CudaArray<double> _wle_in_c ;
    PlanckInegral _planck ;
} ;

inline void bin_central(const Grid& g, Field3D<double>& input_field, Field3D<double>& binned_field, int num_wle, int num_bands) {

    int n = num_wle/num_bands;

    for (int i=0; i<g.NR + 2*g.Nghost; i++) {
        for (int j=0; j<g.Nphi + 2*g.Nghost; j++) { 
            for (int k=0; k<num_bands; k++) {
                int idx = binned_field.index(i,j,k);
                int idx0 = input_field.index(i,j,(k*n)+(n/2));
                binned_field[idx] = input_field[idx0];
            }
        }
    }
}

#endif// _CUDISC_HEADERS_BINS_H_