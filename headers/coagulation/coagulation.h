
#ifndef _CUDISC_HEADERS_COAGULATION_COAGULATION_H_
#define _CUDISC_HEADERS_COAGULATION_COAGULATION_H_

#include "cuda_array.h"
#include "field.h"
#include "grid.h"
#include "star.h"

#include "coagulation/size_grid.h"
#include "coagulation/kernels.h"
#include "coagulation/fragments.h"

class CoagulationCacheRef ;

class CoagulationCache {
  public:

    CoagulationCache(int size, int block_size=-1) {   
        if (block_size < 0) { // Use a power of 2, up to 16 bytes
            block_size = 1 ; 
            int max_size = 16 ;
            while (block_size < size && block_size < max_size)
                block_size *= 2 ;
        }
        // Setup storage, padding the storage in j to be a multiple of block_size:
        stride = block_size * ((size + block_size-1) / block_size) ;

        _id = make_CudaArray<id>(size * stride) ;
        _Cijk = make_CudaArray<coeff>(size * stride) ;
        _Cijk_frag = make_CudaArray<double>(size * stride) ;
    }

    struct id {
        int coag, frag, remnant ;
    } ;
    struct coeff {
        double coag, remnant, eps ;
    } ;

    id& index(int i, int j) {
        return _id[i*stride + j] ;
    }
    id index(int i, int j) const {
        return _id[i*stride + j] ;
    }

    coeff& Cijk(int i, int j) {
        return _Cijk[i*stride + j] ;
    }
    coeff Cijk(int i, int j) const {
        return _Cijk[i*stride + j] ;
    }

    double& Cijk_frag(int i, int j) {
        return _Cijk_frag[i*stride + j] ;
    } 
    double Cijk_frag(int i, int j) const {
        return _Cijk_frag[i*stride + j] ;
    }


  private:
    CudaArray<id> _id ;
    CudaArray<coeff> _Cijk ;
    CudaArray<double> _Cijk_frag ;

    int stride ;

    friend class CoagulationCacheRef ;
} ;

template<class Kernel, class Fragments>
class CoagulationRate {
  public:

    CoagulationRate(SizeGrid& grains, Kernel kernel, Fragments fragments)
     : _kernel(kernel), _grain_sizes(grains), _cache(grains.size())
    {
        _set_coagulation_properties() ;
        _set_fragment_properties(fragments) ;
    }

    void set_kernel(Kernel kernel) {
        _kernel = kernel ;
    }

    void set_fragments(Fragments fragments) {
        _set_fragment_properties(fragments) ;
    }

    void operator()(const Field3D<double>& dust_density, Field3D<double>& rate) const ;

    struct id {
        int coag, frag, remnant, eps ;
    } ;
    struct coeff {
        double coag, remnant ;
    } ;

  private:
    Kernel _kernel ;
    SizeGrid& _grain_sizes ;

    // Helpers for coagulation / fragmentation
    CoagulationCache _cache ;

    void _set_coagulation_properties() ;
    void _set_fragment_properties(Fragments) ;
} ;

template<class Kernel, class Fragments>
CoagulationRate<Kernel, Fragments> create_coagulation_rate(SizeGrid& sizes, Kernel kernel, Fragments fragments) {
    return CoagulationRate<Kernel, Fragments>(sizes, std::move(kernel), std::move(fragments)) ;
}


#endif//_CUDISC_HEADERS_COAGULATION_COAGULATION_H_