
#ifndef _CUDISC_HEADERS_COAGULATION_KERNELS_H_
#define _CUDISC_HEADERS_COAGULATION_KERNELS_H_

#include <cmath>

#include "constants.h"
#include "grid.h"
#include "field.h"
#include "dustdynamics.h"
#include "dustdynamics1D.h"

#include "coagulation/size_grid.h"

struct KernelResult {
    double K, p_coag, p_frag ;
} ;

struct vec3 {
    RealType R, Z, phi ;
} ;

template<bool use_full_stokes=false>
class BirnstielKernel {
  public:
    BirnstielKernel(Grid&g, SizeGrid& sizes, const Field3D<Prims>& wd, 
                    const Field<Prims>& wg, const Field<double>& sound_speed, 
                    const Field<double>& alpha, const Field<double>& mu, double Mstar=1)
      : _g(g), _cs(sound_speed), _sizes(sizes),
        _wd(wd), _wg(wg),
        _alpha_t(alpha), _mu(mu) , _GMstar(Mstar*GMsun)
    { } ;

    /* The kernel is evaluated in two stages, so that the quantities that
     * depend on only one cell, or on one grain size in one cell, are computed
     * once rather than once for every pair of sizes:
     *   CellProps : quantities of the cell
     *   SizeProps : quantities of one grain size in the cell
     */
    // The types follow those used when the kernel is evaluated directly
    struct CellProps {
        RealType rho, cs, R, Omega, sqrtRe_1 ;
        double alpha, mu ;
    } ;
    struct SizeProps {
        RealType a, St, St_star, m, i_to_t_rat ;
        double v_R, v_Z, v_phi ;
    } ;

    __device__ __host__
    CellProps cell_props(int i, int j) const ;
    __device__ __host__
    SizeProps size_props(int i, int j, int k, const CellProps& cell) const ;

    // Compute the kernel for a pair of sizes in a cell.
    __device__ __host__
    KernelResult operator()(const CellProps& cell, const SizeProps& s1, const SizeProps& s2) const ;

    // Compute the kernel for cell i,j and species k1 and k2.
    __device__ __host__
    KernelResult operator()(int i, int j, int k1, int k2) const {
        CellProps cell = cell_props(i,j) ;
        return (*this)(cell, size_props(i,j,k1,cell), size_props(i,j,k2,cell)) ;
    }

    __device__ __host__
    int NR() const {
        return _g.NR + 2*_g.Nghost ;
    }
    __device__ __host__
    int Nphi() const {
        return _g.Nphi + 2*_g.Nghost ;
    }
    
    void set_fragmentation_threshold(double v_frag_base) {
        _v_frag_b = v_frag_base ;
    }
    void set_icy_fragmentation_threshold(double v_frag_ice) {
        _v_frag_i = v_frag_ice ;
    }

  private:
    GridRef _g ;
    FieldConstRef<double> _cs ;
    SizeGridRef _sizes;
    Field3DConstRef<Prims> _wd ;
    FieldConstRef<Prims> _wg ;
    FieldConstRef<double> _alpha_t ;
    FieldConstRef<double> _mu ;

    RealType _GMstar;
    RealType _v_frag_b=1000., _v_frag_i=1000.;

} ;

template<bool use_full_stokes=false>
class BirnstielKernelVertInt {
  public:
    BirnstielKernelVertInt(Grid&g, SizeGrid& sizes, const Field3D<Prims1D>& wd,
                    const Field<Prims1D>& wg, const Field<double>& sound_speed, 
                    const Field<double>& alpha, double mu, double Mstar=1)
      : _g(g), _cs(sound_speed), _sizes(sizes),
      _wd(wd), _wg(wg),
      _alpha_t(alpha), _GMstar(Mstar*GMsun), _mu(mu)
    { } ;

    // Two-stage evaluation, as for BirnstielKernel
    struct CellProps {
        RealType Sig_g, cs, R, Omega, sqrtRe_1 ;
        double alpha, Hp2 ;
    } ;
    struct SizeProps {
        RealType a, St, St_star, m, i_to_t_rat ;
        double v_R, v_phi, h2, v_settle ;
    } ;

    __device__ __host__
    CellProps cell_props(int i, int j) const ;
    __device__ __host__
    SizeProps size_props(int i, int j, int k, const CellProps& cell) const ;

    // Compute the kernel for a pair of sizes in a cell.
    __device__ __host__
    KernelResult operator()(const CellProps& cell, const SizeProps& s1, const SizeProps& s2) const ;

    // Compute the kernel for cell i,j and species k1 and k2.
    __device__ __host__
    KernelResult operator()(int i, int j, int k1, int k2) const {
        CellProps cell = cell_props(i,j) ;
        return (*this)(cell, size_props(i,j,k1,cell), size_props(i,j,k2,cell)) ;
    }

    __device__ __host__
    int NR() const {
        return _g.NR + 2*_g.Nghost ;
    }
    __device__ __host__
    int Nphi() const {
        return _g.Nphi + 2*_g.Nghost ;
    }

    void set_fragmentation_threshold(double v_frag_base) {
      _v_frag_b = v_frag_base ;
    }
    void set_icy_fragmentation_threshold(double v_frag_ice) {
      _v_frag_i = v_frag_ice ;
    }

  private:
    GridRef _g ;
    FieldConstRef<double> _cs ;
    SizeGridRef _sizes;
    Field3DConstRef<Prims1D> _wd ;
    FieldConstRef<Prims1D> _wg ;
    FieldConstRef<double> _alpha_t ;

    RealType  _GMstar, _mu;
    RealType _v_frag_b=1000., _v_frag_i=1000.;

} ;

class ConstantKernel {
  public:
    ConstantKernel(const Grid& g)
     : _g(g)
    { } 

    // Two-stage evaluation, as for BirnstielKernel
    struct CellProps { } ;
    struct SizeProps { } ;

    __device__ __host__
    CellProps cell_props(int, int) const {
        return {} ;
    }
    __device__ __host__
    SizeProps size_props(int, int, int, const CellProps&) const {
        return {} ;
    }

    __device__ __host__
    KernelResult operator()(const CellProps&, const SizeProps&, const SizeProps&) const {
        return {4., 1., 0.} ;
    }

    // Compute the kernel for cell i,j and species k1 and k2.
    __device__ __host__
    KernelResult operator()(int, int, int, int) const {
        return {4., 1., 0.} ;
    }

    __device__ __host__
    int NR() const {
        return _g.NR + 2*_g.Nghost ;
    }
    __device__ __host__
    int Nphi() const {
        return _g.Nphi + 2*_g.Nghost ;
    }

  private:
    GridRef _g ;
} ;

#endif//_CUDISC_HEADERS_COAGULATION_KERNELS_H_