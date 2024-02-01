#ifndef _CUDISC_HEADERS_OPACITY_H_
#define _CUDISC_HEADERS_OPACITY_H_

#include <math.h>
#include <cuda.h>


/* class ConstantOpacity
 *
 * Provides a wavelength-independent opacity.
 */
class ConstantOpacity {
  public:
    ConstantOpacity(double kappa) 
     : _kappa(kappa)
    { } ;

    __device__ __host__ 
    double kappa_abs(double) const {
        return _kappa ;
    }

    __device__ __host__ 
    double kappa_sca(double) const {
        return 0 ;
    }

    __device__ __host__ 
    double mean(double) const {
        return _kappa ;
    }


  private:
    double _kappa ;
} ;

/* class SingleGrainOpacity
 *
 * Provides a simple model for the opacity of a single grain.
 *
 * Units:
 *    s    : micron         (grain size)
 *    rho  : g cm^{-3}      (grain internal density)
 *    wle  : micron         (wavelength)
 *    beta : dimensionless  (Rayleigh index)
 */
class SingleGrainOpacity {
  public:
    SingleGrainOpacity(double s, double beta=1, double rho=1) 
     : _s2pi(2*M_PI*s), _k0(1e4 * 3/(4*rho*s)), _beta(beta)
    { } ;

    __device__ __host__ 
    double kappa_abs(double wle) const {
        if (wle > _s2pi)
            return _k0 * pow(_s2pi/wle, _beta) ;
        else 
            return _k0 ;
    }

    __device__ __host__ 
    double kappa_sca(double) const {
        return 0 ;
    }

    __device__ __host__ 
    double mean(double T) const {
        return _k0 / (1 + pow(3.475e-4*_s2pi*T, -_beta)) ;
    }

  private:
    double _s2pi, _k0, _beta ;
} ;


/* class GrainDistributionOpacity
 *
 * Provides a simple model for the opacity of a power-law distribution of
 * grain sizes
 *
 * Units:
 *    s1  : micron          (maximum grain size)
 *    s0  : micron          (minimum grain size)
 *    rho : g cm^{-3}       (grain internal density)
 *    wle : micron          (wavelength)
 *    beta : dimensionless  (Rayleigh index)
 *    q    : dimensionless  (Power-law index of the grain-size distribution)
 */
class GrainDistributionOpacity {
  public:
    GrainDistributionOpacity(double s1, double s0=0.01, double q=3.5, 
                             double beta=1, double rho=1.)
      : _s0(s0), _s1(s1), _q(q), _beta(beta)
    { 
        // 1e4 corrects for grain size being in microns
        double m_inv ;
        if (_q != 4)
            m_inv = 1e4 * 3 / (4 * rho * (pow(s1, 4-q) - pow(s0, 4-q))) ;
        else
            m_inv = 1e4 * 3 / (4 * rho * log(s1/s0)) ;


        if (q - _beta != 3)
            _k0 = m_inv * pow(_s0,  3 - q + _beta) / (3 - q + _beta) ;
        else 
            _k0 = m_inv ;

        if (_q != 3)
            _k1 = m_inv * pow(s1, 3 - _q) / (3 - _q) ;
        else
            _k1 = m_inv ;
    }

    __device__ __host__ 
    double kappa_abs(double wle) const {
        double l = wle/(2*M_PI) ;
        if      (l > _s1) l = _s1 ;
        else if (l < _s0) l = _s0 ;

        double kappa = 0 ;

        // Handle grains with Q(s,wle) = 1
        if (_s1 > l) {
            if (_q != 3)
                kappa += _k1 * (1 - pow(l/_s1, 3-_q)) ;
            else
                kappa += _k1 * log(_s1/l) ;
        }

        // Handle grains with Q(s,wle) = (2pi*s/wle)^beta
        if (_s0 < l) {
            double f = pow(2*M_PI/wle, _beta) ;
            double a = 3 - _q + _beta ;
            if (a != 0)
                kappa += f * _k0 * (pow(l/_s0, a) - 1) ;
            else
                kappa += f * _k0 * log(l/_s0) ;
        }

        return kappa ;
    }
    
    __device__ __host__ 
    double kappa_sca(double) const {
        return 0 ;
    }
    
    __device__ __host__ 
    double mean(double T) const {
        double wle = 2.88e3 / T ;
        return kappa_abs(wle) ;
    }

  private:
    double _s0, _s1, _q, _beta ;
    double _k0, _k1 ;
} ;



#endif//_CUDISC_HEADERS_OPACITY_H_