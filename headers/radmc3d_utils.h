#ifndef _CUDISC_RADMC3D_UTILS_H_
#define _CUDISC_RADMC3D_UTILS_H_

#include <string>
#include <vector>

#include "grid.h"
#include "field.h"
#include "star.h"

namespace Radmc3d {

class DiscModel {
  public:
    DiscModel(std::string folder) ;

    Grid create_grid(int nghost_y) const ;
    Field<double> read_quantity(std::string filename, const Grid& g) const ;


    int nr() const {
        return _radius.size() - 1;
    }
    int nt() const {
        return _colatitude.size() - 1 ;
    }

    double r_c(int i) const {
        return 0.5*(_radius[i] + _radius[i+1]) ;
    }
    double theta_c(int i) const {
        return 0.5*(_colatitude[i] + _colatitude[i+1]) ;
    }

  private:
    std::vector<double> _radius, _colatitude ;
    std::string _folder ;

    void _parse_amr_grid(std::string name="amr_grid.inp") ;
    double interp_qty(const std::vector<double>& qty, double r, int j) const ;

} ;

class DustOpacity {
  public:
    DustOpacity(std::string folder, std::string filename="dustkappa_silicate.inp") ;

    int num_wle() const { return _wle.size() ; }
    bool has_scattering() const { return _ksca.size() > 0; }
    bool has_anisotropic_scattering() const { return _gsca.size() > 0 ;} 

    double wle(int i) const       { return _wle[i] ; }
    double kappa_abs(int i) const { return _kabs[i] ; }
    double kappa_sca(int i) const { return _ksca[i] ; }
    double g_sca(int i) const     { return _gsca[i] ; }


    double kappa_abs(double wle) const { return _interp(wle,_kabs, 1) ; }
    double kappa_sca(double wle) const { return _interp(wle,_ksca, 1) ; }
    double g_sca(double wle) const     { return _interp(wle,_gsca, 0) ; }


  private:
    void _parse_dust_opacity() ;

    double _interp(double l, const std::vector<double>& qty, bool extrap) const ;  

    std::string _folder, _file ;
    std::vector<double> _wle, _kabs, _ksca, _gsca ;
} ;

Star load_stellar_properties(std::string folder, std::string filename) ;

} // namespace Radmc3d

#endif//_CUDISC_RADMC3D_UTILS_H_
