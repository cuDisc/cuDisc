#ifndef _CUDISC_HEADERS_DISC_HELPERS_H_
#define _CUDISC_HEADERS_DISC_HELPERS_H_

#include "cuda_array.h"
#include "field.h"
#include "grid.h"

struct Prims ;

/* Per-step helpers for setting up the disc's thermal and transport
 * properties, over all cells (including ghosts):
 *
 *   compute_cs2   : cs2 = R_gas T / mu
 *   cs2_to_cs     : cs = sqrt(cs2)
 *   compute_nu    : nu = alpha cs2 / Omega_K (cs2 at j=2), or
 *                   nu = alpha nu0 (R/au) / sqrt(Mstar)
 *   compute_alpha : alpha2D = nu Omega_K / cs2, i.e. the local alpha for which
 *                   alpha2D cs2 / Omega_K equals the (vertically constant) nu
 *   compute_D     : D = rho_g alpha cs2 / (Sc Omega_K), or D = rho_g nu / Sc
 *                   (the same for every dust species)
 *   compute_Cv    : Cv = (dof/2) R_gas / mu, for dof degrees of freedom
 *                   (default 5, a diatomic gas)
 *
 * The default versions run on the GPU, so the fields are not migrated to the
 * host. The _cpu versions loop over the fields on the host.
 */
void compute_cs2(const Grid& g, const Field<double>& T, Field<double>& cs2, const Field<double>& mu) ;
void compute_cs2(const Grid& g, const Field<double>& T, Field<double>& cs2, double mu) ;
void cs2_to_cs(const Grid& g, Field<double>& cs, const Field<double>& cs2) ;
void compute_nu(const Grid& g, CudaArray<double>& nu, const Field<double>& cs2, double Mstar, double alpha) ;
void compute_nu(const Grid& g, CudaArray<double>& nu, double nu0, double Mstar, double alpha) ;
void compute_alpha(const Grid& g, const CudaArray<double>& nu, Field<double>& alpha2D, const Field<double>& cs2, double Mstar) ;
void compute_D(const Grid& g, Field3D<double>& D, const Field<Prims>& wg, const Field<double>& cs2, double Mstar, double alpha, double Sc) ;
void compute_D(const Grid& g, Field3D<double>& D, const Field<Prims>& wg, const CudaArray<double>& nu, double Sc) ;
void compute_Cv(const Grid& g, Field<double>& Cv, const Field<double>& mu, double dof=5) ;

void compute_cs2_cpu(const Grid& g, const Field<double>& T, Field<double>& cs2, const Field<double>& mu) ;
void compute_cs2_cpu(const Grid& g, const Field<double>& T, Field<double>& cs2, double mu) ;
void cs2_to_cs_cpu(const Grid& g, Field<double>& cs, const Field<double>& cs2) ;
void compute_nu_cpu(const Grid& g, CudaArray<double>& nu, const Field<double>& cs2, double Mstar, double alpha) ;
void compute_nu_cpu(const Grid& g, CudaArray<double>& nu, double nu0, double Mstar, double alpha) ;
void compute_alpha_cpu(const Grid& g, const CudaArray<double>& nu, Field<double>& alpha2D, const Field<double>& cs2, double Mstar) ;
void compute_D_cpu(const Grid& g, Field3D<double>& D, const Field<Prims>& wg, const Field<double>& cs2, double Mstar, double alpha, double Sc) ;
void compute_D_cpu(const Grid& g, Field3D<double>& D, const Field<Prims>& wg, const CudaArray<double>& nu, double Sc) ;
void compute_Cv_cpu(const Grid& g, Field<double>& Cv, const Field<double>& mu, double dof=5) ;

#endif//_CUDISC_HEADERS_DISC_HELPERS_H_
