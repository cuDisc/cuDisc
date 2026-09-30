#ifndef _CUDISC_DENSITY_VIEW_H_
#define _CUDISC_DENSITY_VIEW_H_

#include <cstddef>

#include "field.h"

struct Prims;
struct Prims1D;
struct Quants;

// Density accessor for plain density fields. The overloads for Prims, Quants
// and Prims1D live next to those structs (dustdynamics.h, dustdynamics1D.h).
inline __host__ __device__ double get_rho(const double& rho) { return rho ; }
inline __host__ __device__ double& get_rho(double& rho) { return rho ; }

/* class DensityView
 *
 * Device-copyable view of the density component of a Field3D, whatever its
 * element type (double, Prims, Prims1D, Quants). The element type is reduced
 * to a pointer to the first density and the element size in bytes, so kernels
 * that only need the density do not need to be templated on the element type.
 * The element may contain members of any type; only the density itself must
 * be stored as a double.
 *
 * Construct with density_view(field). Adding support for a new element type
 * only requires a new density_view overload.
 */
struct DensityView {
    char* ptr ;               // address of the first density
    std::size_t elem_bytes ;  // sizeof(element)
    int Nd ;
    int stride_Zd ;
    int stride_d ;

    __host__ __device__
    double& operator()(int i, int j, int k) const {
        std::size_t idx = std::size_t(i*stride_Zd + j*stride_d + k) ;
        return *reinterpret_cast<double*>(ptr + idx*elem_bytes) ;
    }
} ;

// Field3D<T> converts implicitly to Field3DRef<T>, so these accept either.
DensityView density_view(Field3DRef<double> f) ;
DensityView density_view(Field3DRef<Prims> f) ;
DensityView density_view(Field3DRef<Prims1D> f) ;
DensityView density_view(Field3DRef<Quants> f) ;

#endif//_CUDISC_DENSITY_VIEW_H_
