

#ifndef _CUDISC_HEADERS_PLANCK_H_
#define _CUDISC_HEADERS_PLANCK_H_

#include <cmath>

#include "cuda_array.h"
#include "interpolate.h"

class PlanckIntegralRef ;

/* class PlanckInegral
 *
 * Provides an approximate interpolant for the normalized Planck integral
 * in frequency, i.e.
 *
 *     P(x) = \frac{15}{\pi^4}\int_0^x \frac{t^3}{\exp(t) - 1} dt
 *
 */
class PlanckInegral { 
  public:
    PlanckInegral()
      : _lgx0(std::log10(1e-6)), _dlgx(0.03125), Nmax(512)
    {
        int fac = 128 ;
        std::vector<double> x(Nmax*fac+1) ;
        std::vector<double> B(Nmax*fac+1) ;

        for (int i=0; i <= fac*Nmax; i++) {
            x[i] = std::pow(10, _lgx0 + i*_dlgx/fac) ;
            B[i] = eval(x[i]) ;
        }

        PchipInterpolator<1> interp(x, B) ;

        _val = make_CudaArray<double>(Nmax) ;
        _val[0] = _norm * x[0]*x[0]*x[0]/3;
        for (int i=1; i < Nmax; i++) {
            _val[i] = _val[i-1] + 
                interp.integrate(std::pow(10, _lgx0 + _dlgx*(i-1)),
    		  			         std::pow(10, _lgx0 + _dlgx*i));

            _val[i-1] = std::log(_val[i-1]) ;
        }
        _val[Nmax-1] = std::log(_val[Nmax-1]) ;

        // normalize so that integral is exactly 1
        double norm = _val[Nmax-1] ;
        for (int i=0; i < Nmax; i++)
            _val[i] -= norm ;
    }

    double operator()(double x) const {
        double lgx = std::log10(x) ;

        double f = (lgx - _lgx0) / _dlgx ;
        int i = std::floor(f)  ;
        f -= i ;
        
        if (i < 0)
            return _norm * x*x*x / 3. ;   
        else if (i >= Nmax-1)
            return (1 - std::exp(-x)*(6 + x*(6 + x*(3 + x)))) ;
        else 
            return std::exp(_val[i]*(1-f) + _val[i+1]*f) ;
    }

    // Wavelength is in microns, T in Kelvin
    double WienParameter(double wle, double T) const {
        return 1.438775e4 / (wle * T) ;
    }

private:
    friend class PlanckIntegralRef ;

    double eval(double x) const {
       return _norm*x*x*x/std::expm1(x) ;
    }
    double _norm = 0.15398973382026507 ;

      
    CudaArray<double> _val ;
    double _lgx0, _dlgx ;
    int Nmax ;
} ;



/* class PlanckIntegralRef
 *
 * Reference type for PlanckInegral, allowing it to be used on the GPU.
 *
 */
class PlanckIntegralRef {
  public:
    PlanckIntegralRef(const PlanckInegral& pl) 
      : _val(pl._val.get()), _lgx0(pl._lgx0), _dlgx(pl._dlgx),
        Nmax(pl.Nmax)
    { } ;

    __device__ __host__ double operator()(double x) const {
        double lgx = log10(x) ;

        double f = (lgx - _lgx0) / _dlgx ;
        int i = floor(f)  ;
        f -= i ;
        
        if (i < 0)
            return _norm * x*x*x / 3. ;   
        else if (i >= Nmax-1)
            return (1 - exp(-x)*(6 + x*(6 + x*(3 + x)))) ;
        else 
            return exp(_val[i]*(1-f) + _val[i+1]*f) ;
    }

    // Wavelength is in microns, T in Kelvin
     __device__  __host__ double WienParameter(double wle, double T) const {
        return 1.438775e4 / (wle * T) ;
    }

  private:
    double _norm = 0.15398973382026507 ;

    double* _val ;
    double _lgx0, _dlgx ;
    int Nmax ;
} ;

/* planck_factor
 *
 * Evaualtes the fraction of thermal energy radiated into a given band
*/
inline __device__ __host__
double planck_factor(const PlanckIntegralRef& planck, double T, int i, int num_bands,
                     const double* wavelengths) {

    double f0, f1 ;
    double x ;
    if (i == 0)
        f1 = 1 ;
    else {
        x = planck.WienParameter(wavelengths[i-1], T) ;
        f1 = planck(x) ;
    }
    
    if (i == num_bands-1)
        f0 = 0 ;
    else {
        x = planck.WienParameter(wavelengths[i], T) ;
        f0 = planck(x) ;
    }

    return (f1 - f0) ;
}

#endif//_CUDISC_HEADERS_PLANCK_H_