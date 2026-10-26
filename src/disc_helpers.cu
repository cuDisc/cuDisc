#include <cmath>

#include "disc_helpers.h"
#include "constants.h"
#include "dustdynamics.h"
#include "utils.h"

// The contiguous index (j for 2D fields, k for 3D fields) is mapped to
// threadIdx.x.

namespace {

__global__ void _compute_cs2(GridRef g, FieldConstRef<double> T, FieldRef<double> cs2, FieldConstRef<double> mu) {

    int jidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int iidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int jstride = gridDim.x * blockDim.x ;
    int istride = gridDim.y * blockDim.y ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride)
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride)
            cs2(i,j) = R_gas * T(i,j) / mu(i,j);
}

__global__ void _compute_cs2(GridRef g, FieldConstRef<double> T, FieldRef<double> cs2, double mu) {

    int jidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int iidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int jstride = gridDim.x * blockDim.x ;
    int istride = gridDim.y * blockDim.y ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride)
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride)
            cs2(i,j) = R_gas * T(i,j) / mu;
}

__global__ void _cs2_to_cs(GridRef g, FieldRef<double> cs, FieldConstRef<double> cs2) {

    int jidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int iidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int jstride = gridDim.x * blockDim.x ;
    int istride = gridDim.y * blockDim.y ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride)
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride)
            cs(i,j) = sqrt(cs2(i,j));
}

__global__ void _compute_nu(GridRef g, double* nu, FieldConstRef<double> cs2, double Mstar, double alpha) {

    int i = threadIdx.x + blockIdx.x*blockDim.x ;
    int istride = gridDim.x * blockDim.x ;

    for (; i<g.NR+2*g.Nghost; i+=istride) {
        double Om = sqrt(GMsun * Mstar / (g.Rc(i)*g.Rc(i)*g.Rc(i)));
        nu[i] = alpha * cs2(i,2) / Om;
    }
}

__global__ void _compute_nu(GridRef g, double* nu, double nu0, double Mstar, double alpha) {

    int i = threadIdx.x + blockIdx.x*blockDim.x ;
    int istride = gridDim.x * blockDim.x ;

    for (; i<g.NR+2*g.Nghost; i+=istride)
        nu[i] = alpha * nu0 * (g.Rc(i)/au) / sqrt(Mstar);
}

__global__ void _compute_alpha(GridRef g, const double* nu, FieldRef<double> alpha2D, FieldConstRef<double> cs2, double Mstar) {

    int jidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int iidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int jstride = gridDim.x * blockDim.x ;
    int istride = gridDim.y * blockDim.y ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        double Om = sqrt(GMsun*Mstar/g.Rc(i))/g.Rc(i);
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride)
            alpha2D(i,j) = nu[i] * Om / cs2(i,j);
    }
}

__global__ void _compute_D(GridRef g, Field3DRef<double> D, FieldConstRef<Prims> wg, FieldConstRef<double> cs2, double Mstar, double alpha, double Sc) {

    int kidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int iidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int kstride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;
    int istride = gridDim.z * blockDim.z ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        double Om = sqrt(GMsun * Mstar / (g.Rc(i)*g.Rc(i)*g.Rc(i)));
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) {
            double D_ij = wg(i,j).rho * alpha * cs2(i,j) / (Sc*Om) ;
            for (int k=kidx; k<D.Nd; k+=kstride)
                D(i,j,k) = D_ij ;
        }
    }
}

__global__ void _compute_D(GridRef g, Field3DRef<double> D, FieldConstRef<Prims> wg, const double* nu, double Sc) {

    int kidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int iidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int kstride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;
    int istride = gridDim.z * blockDim.z ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) {
            double D_ij = wg(i,j).rho * nu[i] / Sc ;
            for (int k=kidx; k<D.Nd; k+=kstride)
                D(i,j,k) = D_ij ;
        }
    }
}

__global__ void _compute_Cv(GridRef g, FieldRef<double> Cv, FieldConstRef<double> mu, double dof) {

    int jidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int iidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int jstride = gridDim.x * blockDim.x ;
    int istride = gridDim.y * blockDim.y ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride)
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride)
            Cv(i,j) = 0.5*dof*R_gas / mu(i,j);
}

const dim3 threads2D(32,8) ;
dim3 blocks2D(const Grid& g) {
    return dim3((g.Nphi+2*g.Nghost+31)/32, (g.NR+2*g.Nghost+7)/8) ;
}

const int threads1D = 128 ;
int blocks1D(const Grid& g) {
    return (g.NR+2*g.Nghost+threads1D-1)/threads1D ;
}

const dim3 threads3D(32,4,2) ;
dim3 blocks3D(const Grid& g, const Field3D<double>& D) {
    return dim3((D.Nd+31)/32, (g.Nphi+2*g.Nghost+3)/4, (g.NR+2*g.Nghost+1)/2) ;
}

} // namespace

void compute_cs2(const Grid& g, const Field<double>& T, Field<double>& cs2, const Field<double>& mu) {
    _compute_cs2<<<blocks2D(g), threads2D>>>(g, T, cs2, mu) ;
    check_CUDA_errors("compute_cs2") ;
}

void compute_cs2(const Grid& g, const Field<double>& T, Field<double>& cs2, double mu) {
    _compute_cs2<<<blocks2D(g), threads2D>>>(g, T, cs2, mu) ;
    check_CUDA_errors("compute_cs2") ;
}

void cs2_to_cs(const Grid& g, Field<double>& cs, const Field<double>& cs2) {
    _cs2_to_cs<<<blocks2D(g), threads2D>>>(g, cs, cs2) ;
    check_CUDA_errors("cs2_to_cs") ;
}

void compute_nu(const Grid& g, CudaArray<double>& nu, const Field<double>& cs2, double Mstar, double alpha) {
    _compute_nu<<<blocks1D(g), threads1D>>>(g, nu.get(), cs2, Mstar, alpha) ;
    check_CUDA_errors("compute_nu") ;
}

void compute_nu(const Grid& g, CudaArray<double>& nu, double nu0, double Mstar, double alpha) {
    _compute_nu<<<blocks1D(g), threads1D>>>(g, nu.get(), nu0, Mstar, alpha) ;
    check_CUDA_errors("compute_nu") ;
}

void compute_alpha(const Grid& g, const CudaArray<double>& nu, Field<double>& alpha2D, const Field<double>& cs2, double Mstar) {
    _compute_alpha<<<blocks2D(g), threads2D>>>(g, nu.get(), alpha2D, cs2, Mstar) ;
    check_CUDA_errors("compute_alpha") ;
}

void compute_D(const Grid& g, Field3D<double>& D, const Field<Prims>& wg, const Field<double>& cs2, double Mstar, double alpha, double Sc) {
    _compute_D<<<blocks3D(g, D), threads3D>>>(g, D, wg, cs2, Mstar, alpha, Sc) ;
    check_CUDA_errors("compute_D") ;
}

void compute_D(const Grid& g, Field3D<double>& D, const Field<Prims>& wg, const CudaArray<double>& nu, double Sc) {
    _compute_D<<<blocks3D(g, D), threads3D>>>(g, D, wg, nu.get(), Sc) ;
    check_CUDA_errors("compute_D") ;
}

void compute_Cv(const Grid& g, Field<double>& Cv, const Field<double>& mu, double dof) {
    _compute_Cv<<<blocks2D(g), threads2D>>>(g, Cv, mu, dof) ;
    check_CUDA_errors("compute_Cv") ;
}

void compute_cs2_cpu(const Grid& g, const Field<double>& T, Field<double>& cs2, const Field<double>& mu) {
    for (int i=0; i<g.NR + 2*g.Nghost; i++)
        for (int j=0; j<g.Nphi + 2*g.Nghost; j++)
            cs2(i,j) = R_gas * T(i,j) / mu(i,j);
}

void compute_cs2_cpu(const Grid& g, const Field<double>& T, Field<double>& cs2, double mu) {
    for (int i=0; i<g.NR + 2*g.Nghost; i++)
        for (int j=0; j<g.Nphi + 2*g.Nghost; j++)
            cs2(i,j) = R_gas * T(i,j) / mu;
}

void cs2_to_cs_cpu(const Grid& g, Field<double>& cs, const Field<double>& cs2) {
    for (int i=0; i<g.NR + 2*g.Nghost; i++)
        for (int j=0; j<g.Nphi + 2*g.Nghost; j++)
            cs(i,j) = std::sqrt(cs2(i,j));
}

void compute_nu_cpu(const Grid& g, CudaArray<double>& nu, const Field<double>& cs2, double Mstar, double alpha) {
    for (int i=0; i<g.NR + 2*g.Nghost; i++) {
        double Om = std::sqrt(GMsun * Mstar / (g.Rc(i)*g.Rc(i)*g.Rc(i)));
        nu[i] = alpha * cs2(i,2) / Om;
    }
}

void compute_alpha_cpu(const Grid& g, const CudaArray<double>& nu, Field<double>& alpha2D, const Field<double>& cs2, double Mstar) {
    for (int i=0; i<g.NR + 2*g.Nghost; i++) {
        double Om = std::sqrt(GMsun*Mstar/g.Rc(i))/g.Rc(i);
        for (int j=0; j<g.Nphi + 2*g.Nghost; j++)
            alpha2D(i,j) = nu[i] * Om / cs2(i,j);
    }
}

void compute_nu_cpu(const Grid& g, CudaArray<double>& nu, double nu0, double Mstar, double alpha) {
    for (int i=0; i<g.NR + 2*g.Nghost; i++)
        nu[i] = alpha * nu0 * (g.Rc(i)/au) / std::sqrt(Mstar);
}

void compute_D_cpu(const Grid& g, Field3D<double>& D, const Field<Prims>& wg, const Field<double>& cs2, double Mstar, double alpha, double Sc) {
    for (int i=0; i<g.NR + 2*g.Nghost; i++) {
        double Om = std::sqrt(GMsun * Mstar / (g.Rc(i)*g.Rc(i)*g.Rc(i)));
        for (int j=0; j<g.Nphi + 2*g.Nghost; j++)
            for (int k=0; k<D.Nd; k++)
                D(i,j,k) = wg(i,j).rho * alpha * cs2(i,j) / (Sc*Om) ;
    }
}

void compute_D_cpu(const Grid& g, Field3D<double>& D, const Field<Prims>& wg, const CudaArray<double>& nu, double Sc) {
    for (int i=0; i<g.NR + 2*g.Nghost; i++)
        for (int j=0; j<g.Nphi + 2*g.Nghost; j++)
            for (int k=0; k<D.Nd; k++)
                D(i,j,k) = wg(i,j).rho * nu[i] / Sc ;
}

void compute_Cv_cpu(const Grid& g, Field<double>& Cv, const Field<double>& mu, double dof) {
    for (int i=0; i<g.NR + 2*g.Nghost; i++)
        for (int j=0; j<g.Nphi + 2*g.Nghost; j++)
            Cv(i,j) = 0.5*dof*R_gas / mu(i,j);
}
