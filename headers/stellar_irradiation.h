#ifndef _CUDISC_HEADERS_STELLAR_IRRADIATION_H_
#define _CUDISC_HEADERS_STELLAR_IRRADIATION_H_

#include "field.h"
#include "grid.h"
#include "star.h"
#include "dustdynamics.h"
#include "DSHARP_opacs.h"

template<class Opacity>
void compute_stellar_heating(const Star& star, const Grid& g, 
                             const Opacity& kappa, const Field<double>& density, 
                             Field<double>& heating) ;
template<class Opacity>
void compute_stellar_heating(const Star& star, const Grid& g, 
                             const Opacity& kappa, const Field<double>& density, 
                             Field<double>& heating, Field3D<double>& scattering) ;

void compute_stellar_heating(const Star& star, const Grid&, 
                            const Field3D<double>& kappa, 
                            const Field<double>& density, 
                            Field<double>& heating) ;

void compute_stellar_heating_with_scattering(const Star& star, const Grid&, 
                                             const Field3D<double>& kappa_abs, 
                                             const Field3D<double>& kappa_sca,
                                             const Field<double>& density, 
                                             Field<double>& heating, 
                                             Field3D<double>& scattering) ;

void compute_stellar_heating_with_scattering(const Star& star, const Grid& g, 
                                             const Field3D<double>& rhok_abs,
                                             const Field3D<double>& rhok_sca, 
                                             Field<double>& heating,
                                             Field3D<double>& scattering) ;  

void compute_radiation_pressure_with_scattering(const Star& star, const Grid& g, 
                                             const Field3D<double>& rhok_abs, const Field3D<double>& rhok_sca, 
                                             Field3D<double>& f_pressure, const Field3D<Prims>& qd, DSHARP_opacs& opacs) ; 

void compute_stellar_heating_with_scattering_with_inner_disc(const Star& star, const Grid& g, 
                                             const Field3D<double>& rhok_abs,
                                             const Field3D<double>& rhok_sca, 
                                             Field<double>& heating,
                                             Field3D<double>& scattering,
                                             Field3D<double>& tau_inner, double t, CudaArray<double>& ts, int NZ, int Nt) ; 

void compute_radiation_pressure_with_scattering_with_inner_disc(const Star& star, const Grid& g, 
                                             const Field3D<double>& rhok_abs, const Field3D<double>& rhok_sca, 
                                             Field3D<double>& f_pressure, const Field3D<Prims>& qd, DSHARP_opacs& opacs, 
                                             Field<double>& tau_inner, int NZ) ;  

void _interpolate_tau_inner(Field3D<double>& tau_inner, Field<double>& tau_inner_interp, double* ts, double t, int NZ, int Nt) ;                                   

void add_viscous_heating(const Star&star, const Grid& grid, double alpha, 
                         const Field<double>&rho, const Field<double>&cs2, 
                         Field<double>& heating) ;

void add_viscous_heating(const Star& star, const Grid &grid, double alpha, 
                         const Field<Prims>& w_g, const Field<double>&cs2, 
                         Field<double>& heating) ;
void add_viscous_heating(const Star& star, const Grid &grid, 
                         const Field<Prims>& w_g, const CudaArray<double>& nu, 
                         Field<double>& heating) ;
void add_viscous_heating(const Star& star, const Grid &grid, 
                         const CudaArray<double>& Sig, const CudaArray<double>& nu, 
                         Field<double>& heating) ;

#endif//_CUDISC_HEADERS_STELLAR_IRRADIATION_H_
