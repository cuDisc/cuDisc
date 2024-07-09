#ifndef _CUDISC_HEADERS_FLD_H_
#define _CUDISC_HEADERS_FLD_H_

#include "grid.h"
#include "field.h"
#include "flags.h"
#include "pcg_solver.h"

/* class FLD_Solver
 *
 * Solve the hybrid flux-limited diffusion equations for radiation transfer.
 *
 * Two methods are provided, solving either the single-band or multi-band
 * equations.
 * 
 * The diffusion equation is solved using the method outlined in 
 *    Wu, Gao & Dai (2012), https://doi.org/10.1016/j.jcp.2012.06.042
 */
class FLD_Solver {
  public:
    FLD_Solver(double T_ext=0, double tol=1e-4, int max_iter=10000)
      : _T_ext(T_ext), _tol(tol), _max_iter(max_iter)
    { } ;

    /* Set the boundary type for radiation
     *
     * The boundary will be closed unless the flag is set to an
     * open boundary
     */
    void set_boundaries(int flag) {
        _boundary = flag ;
    }
    int get_boundaries() const {
       return _boundary ;
    }

    /* Set the size of ILU(k) preconditioner 
     * The size will be A**k where A*x = b is the system to solve
     * 
     * Note: k < 0 will skip pre-conditioning.
     */
    void set_precond_level(int k) {
        _ILU_order = k ;
    }

    void set_tolerance(double tol) {
        _tol = tol;
    }

    void operator()(const Grid& g, double dt, double Cv, 
                    const Field<double>& rhokappa_P, const Field<double>& rhokappa_R,
                    const Field<double>& rho, 
                    const Field<double>& heat, 
                    Field<double>& T, Field<double>& J);


    void solve_multi_band(const Grid& g, double dt, double Cv, 
                                 const Field3D<double>& rhokappa_abs,
                                 const Field3D<double>& rhokappa_sca,
                                 const Field<double>& rho,
                                 const Field<double>& heat, 
                                 const Field3D<double>& scattering,
                                 const CudaArray<double>& wle,
                                 Field<double>& T, Field3D<double>& J) ;

  private:
    double _T_ext ;
    double _tol ;
    int _max_iter ;
    int _boundary = 
        BoundaryFlags::open_R_inner | BoundaryFlags::open_R_outer;
    int _ILU_order = 0 ;
} ;


#endif//_CUDISC_HEADERS_FLD_H_