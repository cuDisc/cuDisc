
#ifndef _CUDISC_HEADERS_COAGULATION_KERNELS_H_
#define _CUDISC_HEADERS_COAGULATION_KERNELS_H_

#include <cmath>

#include "constants.h"
#include "grid.h"
#include "field.h"
#include "dustdynamics.h"

#include "coagulation/size_grid.h"

struct KernelResult {
    double K, p_coag, p_frag ;
} ;

struct vec3 {
    RealType R, Z, phi ;
} ;

class BirnstielKernel {
  public:
    BirnstielKernel(Grid&g, SizeGrid& sizes, const Field3D<Prims>& wd,
                    const Field<Prims>& wg, const Field<double>& sound_speed, 
                    const Field<double>& alpha, double mu, double Mstar=1)
      : _g(g), _cs(sound_speed), _grain_sizes(sizes.grain_sizes()), _grain_masses(sizes.grain_masses()),
        _wd(wd), _wg(wg),
        _alpha_t(alpha), _GMstar(Mstar*GMsun), _mu(mu),
        _rho_grain(sizes.solid_density())
    { } ;

    // Compute the kernel for cell i,j and species k1 and k2.
    __device__ __host__
    KernelResult operator()(int i, int j, int k1, int k2) const ;

    __device__ __host__
    int NR() const {
        return _g.NR + 2*_g.Nghost ;
    }
    __device__ __host__
    int Nphi() const {
        return _g.Nphi + 2*_g.Nghost ;
    }

    void set_turbulence_Reynolds_Number(double Re) {
        _sqrtRe = std::sqrt(Re) ;
    }
    void set_fragmentation_threshold(double v_frag) {
        _v_frag = v_frag ;
    }

  private:
    GridRef _g ;
    FieldConstRef<double> _cs ;
    const RealType* _grain_sizes ;
    const RealType* _grain_masses ;
    Field3DConstRef<Prims> _wd ;
    FieldConstRef<Prims> _wg ;
    FieldConstRef<double> _alpha_t ;

    RealType _sqrtRe=1e4, _GMstar, _mu;
    RealType _v_frag=1e3, _rho_grain ;

} ;

class BirnstielKernelVertInt {
  public:
    BirnstielKernelVertInt(Grid&g, SizeGrid& sizes, const Field3D<Prims>& wd,
                    const Field<Prims>& wg, const Field<double>& sound_speed, 
                    const Field<double>& alpha, double mu, double Mstar=1)
      : _g(g), _cs(sound_speed), _grain_sizes(sizes.grain_sizes()), _grain_masses(sizes.grain_masses()),
        _wd(wd), _wg(wg),
        _alpha_t(alpha), _GMstar(Mstar*GMsun), _mu(mu),
        _rho_grain(sizes.solid_density())
    { } ;

    // Compute the kernel for cell i,j and species k1 and k2.
    __device__ __host__
    KernelResult operator()(int i, int j, int k1, int k2) const ;

    __device__ __host__
    int NR() const {
        return _g.NR + 2*_g.Nghost ;
    }
    __device__ __host__
    int Nphi() const {
        return _g.Nphi + 2*_g.Nghost ;
    }

    void set_turbulence_Reynolds_Number(double Re) {
        _sqrtRe = std::sqrt(Re) ;
    }
    void set_fragmentation_threshold(double v_frag) {
        _v_frag = v_frag ;
    }

  private:
    GridRef _g ;
    FieldConstRef<double> _cs ;
    const RealType* _grain_sizes ;
    const RealType* _grain_masses ;
    Field3DConstRef<Prims> _wd ;
    FieldConstRef<Prims> _wg ;
    FieldConstRef<double> _alpha_t ;

    RealType _sqrtRe=1e4, _GMstar, _mu;
    RealType _v_frag=1e3, _rho_grain ;

} ;


class ConstantKernel {
  public:
    ConstantKernel(const Grid& g)
     : _g(g)
    { } 

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