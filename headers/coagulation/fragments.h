#ifndef _CUDISC_HEADERS_COAGULATION_FRAGMENTS_H_
#define _CUDISC_HEADERS_COAGULATION_FRAGMENTS_H_

#include <algorithm>
#include <cmath>

/* SimpleErosion
 *
 * Power law distribution of fragments for an erosion/fragmentation model.
 *
 * Params:
 *   Xi    : A mass of Xi*m_proj is removed from the target, up to m_target.
 *   eta   : Index of the number density of fragments, n(m) dm ~ m^-eta dm
 *   m_min : Minimum fragment mass, default = 0. 
 *              Note m_min > 0 is required if eta > 2.
 */
class SimpleErosion {
 public:
  SimpleErosion(double Xi=1, double eta=11/6., double m_min=0)
   : _Xi(Xi), _eta(eta), _m_min(m_min)
    {} ;

  // Mass of the remnant body
  double remnant_mass(double m_i, double m_j) const {
    // Maxmimum mass of erosion/fragmentation products
    double m_max = std::max(m_i, m_j) ;
    double m_erode = std::min(_Xi * std::min(m_i, m_j), m_max) ;

    // Remnant mass
    return m_max - m_erode ;
  }
  
  // Scale parameter for mass distribution
  double mass_scale(double m_i, double m_j) const {
    // Here we just use the largest possible fragment.
    double m_max = std::max(m_i, m_j) ;
    double m_erode = std::min(_Xi * std::min(m_i, m_j), m_max) ;
    
    return m_erode ;
  }

  // Largest possible fragment mass
  double max_fragment_mass(double m_scale) const  {
    return m_scale ;
  }

  // Fraction of fragments with mass < m for the mass scale provided, m_scale
  double cumulative_fragment_distribution(double m, double m_scale) const {
    return _integ(std::min(m, m_scale)) / _integ(m_scale) ;
  }

 private:
  // Integral over fragment mass distribution
  double _integ(double m) const {
    return std::pow(m, 2-_eta) - std::pow(_m_min, 2-_eta);
  }
  
  double _Xi, _eta, _m_min ;
} ;




#endif// _CUDISC_HEADERS_COAGULATION_FRAGMENTS_H_