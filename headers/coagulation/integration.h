#ifndef _CUDISC_HEADERS_COAGULATION_INTEGRATION_H_
#define _CUDISC_HEADERS_COAGULATION_INTEGRATION_H_

#include <iostream>
#include <fstream>

#include "field.h"
#include "grid.h"
#include "dustdynamics.h"

/* Class TimeIntegration
 *
 * Base class for Runge-Kutta (RK) type embedded methods.
 *
 * Provides automatic step-size control for RK-type methods based upon the
 * provided relative and absolute tolerences. The error control maintains the
 * condition
 *     sum( (error / scale)^2 ) < 1,
 * where
 *   scale[i] = rel_tol * max(y^{n}[i], y^{n+1}[i]) + abs_tol*sum(y^{n}).
 *
 * Child classes must implement the step and error estimate.
 */
class TimeIntegration {
  public:
    /* Constructor
     *
     * args:
     *     order   : int, order of the integration method. Used for step-size 
     *               control.
     *     rel_tol : double, default=1e-02, relative error tolerence.
     *     abs_tol : double, default=1e-10, absolute error tolerence (scaled to
     *               the total density.
     */
    TimeIntegration(int order, double rel_tol=1e-2, double abs_tol=1e-10)
      : _rel_tol(rel_tol), _abs_tol(abs_tol), _order(order) 
    { } ;


  template<typename T>
  double take_step(Grid& g, Field3D<double>& y, Field<T>& wg, double& dtguess) const ;
  
  template<typename T>
  double take_step_debug(Grid& g, Field3D<double>& y, Field<T>& wg, double& dtguess, int* idxs) const ;

  template<typename T>
  double take_step_tracers(Grid& g, Field3D<double>& y, Field<T>& wg, double& dtguess, Field3D<double>& tracers, int* idxs) const ;

  void integrate(Grid& g, Field3D<double>& y, double tmax) const ;
  
  template<typename T>
  void integrate(Grid& g, Field3D<T>& ws, Field<T>& wg, double tmax, double& dt_coag, double floor = 1.e-40) const ;

  template<typename T>
  void integrate_debug(Grid& g, Field3D<T>& ws, Field<T>& wg, double tmax, double& dt_coag, double floor) const ;

  template<typename T>
  void integrate_tracers(Grid& g, Field3D<T>& ws, Field<T>& wg, Molecule& mol, double tmax, double& dt_coag, double floor) const ;


protected:

  /* do_step
   *
   * Take a single step of fixed length dt, returning the new density (rho_new)
   * and error estimate (error).
   *
   * This function must be provided by child classes.
   */
  virtual void do_step(double dt, Grid& g, const Field3D<double>& rho,
		       Field3D<double>& rho_new, Field3D<double>& error) const = 0 ;

private:
  double _rel_tol, _abs_tol ;

  static constexpr double _MAX_FACTOR = 10. ;
  static constexpr double _MIN_FACTOR = 0.2 ;
  static constexpr double _SAFETY     = 0.9 ;

  int _order ;    
} ;

template<class Rate>
class Rk2Integration :
  public Rate, public TimeIntegration
{
  public:
    using Rate::set_kernel ;
    using Rate::operator() ;

    Rk2Integration(Rate rate, double rel_tol=1e-2, double abs_tol=1e-10)
      : Rate(std::move(rate)), TimeIntegration(2, rel_tol, abs_tol)
    { } ;

  protected:
    virtual void do_step(double dt, Grid& g, const Field3D<double>& rho,
	                     Field3D<double>& rho_new, Field3D<double>& error) const ;

} ;
template<class Rate>
class BS32Integration :
  public Rate, public TimeIntegration
{
  public:
    using Rate::set_kernel ;
    using Rate::operator() ;

    BS32Integration(Rate rate, double rel_tol=1e-2, double abs_tol=1e-10)
      : Rate(std::move(rate)), TimeIntegration(3, rel_tol, abs_tol)
    { } ;

  protected:
    virtual void do_step(double dt, Grid& g, const Field3D<double>& rho,
	                     Field3D<double>& rho_new, Field3D<double>& error) const ;
} ;

#endif//_CUDISC_HEADERS_COAGULATION_INTEGRATION_H_

