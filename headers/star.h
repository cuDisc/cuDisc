#ifndef _CUDISC_STAR_H_
#define _CUDISC_STAR_H_

#include <cuda_array.h>

class Star {
  public:
    Star(double GM_, double L_, double T_eff) 
      : GM(GM_), L(L_), Teff(T_eff)
    { } ;

    Star(double GM, double R, int num_wle, 
         const CudaArray<double>& wle, const CudaArray<double>& Lband) ;

    template<typename T>
    void set_wavelengths(int num_wle_, const T& wle_) {
        num_wle = num_wle_ ;

        wle = make_CudaArray<double>(num_wle) ;
        for(int i = 0; i < num_wle; i++)
           wle[i] = wle_[i] ;
    }
    void set_blackbody_fluxes() ;

    double GM ;
    double L ;
    double Teff ;

    int num_wle = 0;
    CudaArray<double> wle ;
    CudaArray<double> Lband ;
} ;

 

#endif //_CUDISC_STAR_H_