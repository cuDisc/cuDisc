
#ifndef _CUDISC_ADVECTION_H_
#define _CUDISC_ADVECTION_H_

#include "cuda_array.h"
#include "field.h"
#include "grid.h"


class VanLeerAdvection {
  public:
    VanLeerAdvection(double CFL=0.4)
      : _CFL(CFL)
    { } ;

    void operator()(const Grid& g, Field<double>& Qty,
                    const Field<double>& v_R, const Field<double>& v_phi, double dt) ;
    void operator()(const Grid& g, Field3D<double>& Qty,
                    const Field3D<double>& v_R, const Field3D<double>& v_phi, double dt) ;

    double get_CFL_limit(const Grid& g,
                         const Field<double>& v_R, const Field<double>& v_phi) ;
    double get_CFL_limit(const Grid& g,
                         const Field3D<double>& v_R, const Field3D<double>& v_phi) ;

  private:
    double _CFL ;
} ;

class Diffusion {
  public:
    Diffusion(double CFL=0.25)
      : _CFL(CFL)
    { } ;


    void operator()(const Grid& g, Field<double>& rho_dust,
                    const Field<double>& rho_gas, const Field<double>& D, double dt) ;
    void operator()(const Grid& g, Field3D<double>& rho_dust,
                    const Field<double>& rho_gas, const Field<double>& D, double dt) ;

    /* Update diffusion using super time stepping 
    *  Returns: the number of internal steps used.
    */
    int update_sts(const Grid& g, Field<double>& rho_dust,
                    const Field<double>& rho_gas, const Field<double>& D, double dt) ;
    int update_sts(const Grid& g, Field3D<double>& rho_dust,
                    const Field<double>& rho_gas, const Field<double>& D, double dt) ;

    double get_CFL_limit(const Grid& g, const Field<double>& D) ;

private:
    double _CFL ;
} ;

#endif