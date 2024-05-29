

#include <iostream>
#include <stdexcept>

#include "coagulation/coagulation.h"
#include "coagulation/fragments.h"
#include "dustdynamics.h"

// Ormel & Cuzzi Turbulent squared relative velocity.
//   Scaled to the R.M.S. turbulent speed (\sqrt{alpha} c_s)
//   Here St1 and St2 are the particle Stokes number and sqrtRe is
//   the square root of the Reynolds number
__host__ __device__
RealType Vrel_sqd_OC07(RealType St1, RealType St2, RealType sqrtRe_1) {
    
    // An quadratic approximation to OC07's St*. Good to 2%
    auto St_star = [](RealType St) -> RealType {
        if (St > 1)
            return 1 ;

        RealType a = 0.51505382481121010 ;
        RealType b = 0.12876345620280252 ;

        return St*(1 - b + sqrt(b*b + a*(1-St)/(1+St))) ;
    } ; 

    RealType Sts = max(St_star(max(St1, St2)), sqrtRe_1) ;

    auto V1 = [Sts](RealType St) -> RealType {
        return St * (St / (Sts + St) - St / (1 + St)) ;
    } ;
    auto V2 = [Sts, sqrtRe_1](RealType St) -> RealType {
        // return Sts*Sts / (St + Sts) - sqrtRe_1*sqrtRe_1/(St + sqrtRe_1) ;
        return (Sts-sqrtRe_1) +  (St*St) / (St + Sts) - (St*St) /(St + sqrtRe_1) ;
    } ;

    return max(((St1 - St2) / (St1 + St2)) * (V1(St1) - V1(St2)) + V2(St1) + V2(St2),0.) ;
}

__device__ __host__
KernelResult BirnstielKernel::operator()(int i, int j, int k1, int k2) const {

    // Step 0: Compute the geometric cross-section

    RealType a1 = _grain_sizes[k1] ;
    RealType a2 = _grain_sizes[k2] ;

    RealType xsec = M_PI * (a1 + a2)*(a1 + a2) ;

    // Step 1: Compute the turbulent velocity:
    //   1a. Get the Stokes number (a*tmp)
    RealType rho = _wg(i,j).rho, cs = _cs(i,j), R = _g.Rc(i) ;
    RealType tmp = 0.6266570686577501f * _rho_grain*sqrt(_GMstar/R)/R ; // Ordering needed to prevent over/underflow. 
    //     (4.51351666838205 = 8/sqrt(pi) -> should be sqrt(pi/8) = 0.627...)
    tmp /= (rho*cs) ;

    a1 *= tmp ;
    a2 *= tmp ;

    RealType sqrtRe = sqrt(_alpha_t(i,j) * cs * rho / (sqrt(_GMstar/R)/R * _mu * m_p)  * 2.e-15);

    //   1b: Compute the turbulent velocity
    RealType v_turb = _alpha_t(i,j) * cs*cs * Vrel_sqd_OC07(a1, a2, 1/sqrtRe) ;

    // Protect against NaN at low gas density
    if (rho == 0) v_turb = 0 ;

    //   1c: Compute brownian motion

    tmp = 4.2592967532662155e-24 * (_mu * (_grain_masses[k1] + _grain_masses[k2]) / (_grain_masses[k1]*_grain_masses[k2])) * cs*cs; //4.261679179e-24f

    v_turb += tmp;
    
    // Step 2: Add the laminar components in quadrature
    tmp = _wd(i,j,k1).v_R - _wd(i,j,k2).v_R;
    v_turb += tmp*tmp ;

    tmp = _wd(i,j,k1).v_Z - _wd(i,j,k2).v_Z ;
    v_turb += tmp*tmp ;

    tmp = _wd(i,j,k1).v_phi - _wd(i,j,k2).v_phi ;
    v_turb += tmp*tmp ;

    v_turb = sqrt(v_turb) ;

    // Step 3: Compute the kernel
    KernelResult result ;
    
    result.K = xsec * v_turb ;

    result.p_frag = (1.5*(_v_frag/v_turb)*(_v_frag/v_turb) + 1.) * exp(-1.5*(_v_frag/v_turb)*(_v_frag/v_turb)); // From https://iopscience.iop.org/article/10.3847/1538-4357/ac7d58/pdf
    result.p_coag = 1. - result.p_frag;

    // result.p_coag = max(0.0, min(1.0, 10*(1-v_turb/_v_frag))) ;
    // result.p_frag = 1 - result.p_coag ;

    return result ;
}

__device__ __host__
KernelResult BirnstielKernelVertInt::operator()(int i, int j, int k1, int k2) const {

    // Step 0: Compute the geometric cross-section

    RealType a1 = _grain_sizes[k1] ;
    RealType a2 = _grain_sizes[k2] ;

    RealType xsec = M_PI * (a1 + a2)*(a1 + a2) ;

    // Step 1: Compute the turbulent velocity:
    //   1a. Get the Stokes number (a*tmp)
    RealType Sig_g = _wg(i,j).Sig, cs = _cs(i,j), R = _g.Rc(i) ;

    RealType tmp = M_PI/2 * _rho_grain / Sig_g;
    a1 *= tmp ;
    a2 *= tmp ;

    RealType sqrtRe = sqrt(_alpha_t(i,j) * Sig_g / (2.*_mu * m_p)  * 2.e-15);

    //   1b: Compute the turbulent velocity
    RealType v_turb = _alpha_t(i,j) * cs*cs * Vrel_sqd_OC07(a1, a2, 1/sqrtRe) ;//_vrels(k1,k2)*_vrels(k1,k2); // 

    // Protect against NaN at low gas density
    if (Sig_g == 0) v_turb = 0 ;

    //   1c: Compute brownian motion

    tmp = 4.2592967532662155e-24 * (_mu * (_grain_masses[k1] + _grain_masses[k2]) / (_grain_masses[k1]*_grain_masses[k2])) * cs*cs; //4.261679179e-24f


    v_turb += tmp;
    
    // Step 2: Add the laminar components in quadrature
    tmp = _wd(i,j,k1).v_R - _wd(i,j,k2).v_R;
    v_turb += tmp*tmp ;

    // Step 3: Compute the kernel
    KernelResult result ;
    
    double Hp2 = cs*cs*R/_GMstar *R*R;
    double h12 = Hp2 /(1 + a1/_alpha_t(i,j)); 
    double h22 = Hp2 /(1 + a2/_alpha_t(i,j));

    tmp = pow(sqrt(h12)*min(a1,0.5) - sqrt(h22)*min(a2, 0.5), 2.)/(R*R) * _GMstar/(R);
    v_turb += tmp;
    v_turb = sqrt(v_turb) ;

    result.K = xsec * v_turb * 1./sqrt(2.*M_PI*(h12+h22));

    result.p_frag = (1.5*(_v_frag/v_turb)*(_v_frag/v_turb) + 1.) * exp(-1.5*(_v_frag/v_turb)*(_v_frag/v_turb)); // From https://iopscience.iop.org/article/10.3847/1538-4357/ac7d58/pdf
    result.p_coag = 1. - result.p_frag;

    return result ;
}


class CoagulationCacheRef {
public:
  CoagulationCacheRef(const CoagulationCache& cc) 
    : _id(cc._id.get()), _Cijk(cc._Cijk.get()), _Cijk_frag(cc._Cijk_frag.get()),
     stride(cc.stride) 
  { } ;

  __host__ __device__ 
  CoagulationCache::id index(int i, int j) const {
      return _id[i*stride + j] ;
  }

  __host__ __device__ 
  CoagulationCache::coeff Cijk(int i, int j) const {
      return _Cijk[i*stride + j] ;
  }

  __host__ __device__ 
  double Cijk_frag(int i, int j) const {
      return _Cijk_frag[i*stride + j] ;
  }

private:
  const CoagulationCache::id* _id ;
  const CoagulationCache::coeff* _Cijk ;
  const double* _Cijk_frag ;

  int stride ;
} ;


template<class Kernel, class Fragments>
struct _CoagulationRateHelper {
    
    _CoagulationRateHelper(
            const Kernel& _kernel, const SizeGrid& sizes, 
            const CoagulationCache& _cache)
      : kernel(_kernel), grain_masses(sizes.grain_masses()), cache(_cache),
        size(sizes.size())
    { } ;

    Kernel kernel ;
    const RealType *grain_masses ;

    CoagulationCacheRef cache ;

    int size ;
} ;



template<class Kernel, class Fragments>
__global__ void _compute_coagulation_rate(_CoagulationRateHelper<Kernel,Fragments> coag, 
                                          Field3DConstRef<double> dust_density, int num_tracers,
                                          Field3DRef<double> rate) {

    int s0 = threadIdx.x + blockIdx.x*blockDim.x ;
    int iZ = threadIdx.y + blockIdx.y*blockDim.y ;
    int iR = threadIdx.z + blockIdx.z*blockDim.z ;

    int threads_per_cell = blockDim.x*gridDim.x ;

    // Initialize the shared space
    extern __shared__ double shared_mem[] ;
    double *tmp ;


    if (iR < coag.kernel.NR() && iZ < coag.kernel.Nphi()) {
        // Offset the temporary space
        int size = coag.size * (1 + num_tracers) ;

        tmp = shared_mem + (threadIdx.y + threadIdx.z * blockDim.y) * size ;

        // Initialize the storage
        for (int i=s0; i < size; i += threads_per_cell) {
            tmp[i] = 0 ;
            rate(iR,iZ,i) = 0 ;
        }
    } 
    __syncthreads() ;

    // Main coagulation loop
    if (iR < coag.kernel.NR() && iZ < coag.kernel.Nphi()) {
        for (int i=s0; i < coag.size; i += threads_per_cell) {
            double mi = coag.grain_masses[i] ;      
            double ni = dust_density(iR, iZ, i) / mi ;
        
            for (int j = 0; j < coag.size; ++j) {

                auto Kij = coag.kernel(iR, iZ, i, j) ;

                // Kij.p_coag = 1.;

                double mj = coag.grain_masses[j] ;
                double nj = dust_density(iR, iZ, j) / mj ;
        
                double tot_rate = Kij.K * nj * ni ;
                
                if (tot_rate == 0) 
                    continue ;

                atomicAdd_block(&rate(iR, iZ, i), 
                                -tot_rate * mi * (Kij.p_coag + Kij.p_frag)) ;

                for (int t=1; t < num_tracers+1; t++) {
                    double tracer_rate = Kij.K * nj * 
                        dust_density(iR, iZ, i + t*coag.size) * (Kij.p_coag + Kij.p_frag) ;
                    atomicAdd_block(&rate(iR, iZ, i + t*coag.size), -tracer_rate) ;
                }

                // Kij.p_coag = 1.;
                // Coagulation using Brauer's method.
                if (Kij.p_coag > 0) {
                    double coag_rate = tot_rate * mi * Kij.p_coag ;
                
                    int k = coag.cache.index(i,j).coag ;
                    double f = coag.cache.Cijk(i,j).coag ;
                
                    if (k < coag.size)
                        atomicAdd_block(&rate(iR,iZ,k), f * coag_rate) ;
                    if (k + 1 < coag.size) 
                        atomicAdd_block(&rate(iR,iZ,k+1), (1 - f) * coag_rate) ;

                    for (int t=1; t < num_tracers+1; t++) {
                        double tracer_rate = Kij.K * nj *
                            dust_density(iR, iZ, i + t*coag.size) * Kij.p_coag ;

                        if (k < coag.size)
                            atomicAdd_block(&rate(iR, iZ, k + t*coag.size), f * tracer_rate) ;
                        if (k + 1 < coag.size) 
                            atomicAdd_block(&rate(iR, iZ, k + 1 + t*coag.size), (1-f) * tracer_rate) ;
                    }
                }

                // Fragmentation using the self-similar bins method
                if (Kij.p_frag > 0 && j <= i) {
                    double frag_rate = tot_rate * Kij.p_frag ;
                    if (i == j) frag_rate /= 2 ;

                    int k     = coag.cache.index(i,j).frag;
                    int k_rem = coag.cache.index(i,j).remnant;
                    double m_rem = coag.cache.Cijk(i,j).remnant ;
                    double eps = coag.cache.Cijk(i,j).eps;

                    atomicAdd_block(&tmp[k],              frag_rate * ((mi - m_rem) + mj )) ;
                    atomicAdd_block(&rate(iR,iZ,k_rem),   frag_rate * (          m_rem) * eps) ;
                    atomicAdd_block(&rate(iR,iZ,k_rem+1), frag_rate * (          m_rem) * (1-eps)) ;

                    double rho_tot = dust_density(iR, iZ, i) + dust_density(iR, iZ, j) ;
                    for (int t=1; t < num_tracers+1; t++) {
                        double tracer_rate = frag_rate * 
                            (dust_density(iR, iZ, i + t*coag.size) + dust_density(iR, iZ, j + t*coag.size)) ;
                        tracer_rate /= rho_tot ;
                        
                        atomicAdd_block(&rate(iR,iZ,k_rem + t*coag.size),     tracer_rate * (          m_rem) * eps) ;
                        atomicAdd_block(&rate(iR,iZ,k_rem + 1 + t*coag.size), tracer_rate * (          m_rem) * (1-eps)) ;
                        atomicAdd_block(&tmp[k + t*coag.size],                tracer_rate * ((mi - m_rem)+ mj)) ;
                    }
                }
            }
        }
    }

    __syncthreads() ;

    // Distribute the fragmentation products
    if (iR < coag.kernel.NR() && iZ < coag.kernel.Nphi()) {
        for (int j=s0; j < coag.size; j += threads_per_cell) {
            for (int t=0; t < num_tracers+1; t++)
                for (int i = 0; i < coag.size; ++i) 
                    rate(iR,iZ,j+t*coag.size) += tmp[i + t*coag.size] * coag.cache.Cijk_frag(i,j) ;
            }
    }
    // if (iR==52 && iZ==2 && s0 == 20) {
    //     double ratesum=0;
    //     for (int k=0; k<coag.size; k++) {
    //         ratesum += rate(52,2,k); 
    //     }
    //     printf("%g\n", ratesum);
    // }
}

template<class Kernel, class Fragments>
void CoagulationRate<Kernel,Fragments>::operator()(const Field3D<double>& dust_density,
                                                   Field3D<double>& rate) const {

    if ((dust_density.Nd % _grain_sizes.size()) != 0)
        throw std::invalid_argument("Number of densities must be an integer multiple of the "
                                     "number of grain sizes.") ;

    int num_tracers =  dust_density.Nd / _grain_sizes.size() - 1 ; 

    if (dust_density.Nd > 49152/sizeof(double))
        throw std::invalid_argument("CoagulationRate only supports < 6144 densities.");

    _CoagulationRateHelper<Kernel,Fragments> helper(
        _kernel,_grain_sizes, _cache 
    ) ;

    // Setup Blocks / threads
    //   Make sure we fit at least 1 block into the shared memory - max. shared mem per thread block is 48kB
    dim3 threads(32, 16, 1) ;
    while (threads.y*dust_density.Nd > 49152/sizeof(double)) {
        threads.x *= 2 ;
        threads.y /= 2 ;
    }
    // If we can fit 2 blocks, make it so. - assuming 64kB per multiprocessor
    if (threads.y > 1 and threads.y*dust_density.Nd> 32768/sizeof(double)) {
        threads.x *= 2 ;
        threads.y /= 2 ;
    }
    // Reducing threads.y helps further avoid shared memory size problems.
    // If threads.x > 64 then then the occupancy will be the same whether
    // you have more blocks per thread or threads per block, so small blocks
    // shouldn't limit performance either.
    if (threads.x >= 64)
        threads.y = 1;
    int mem_size = sizeof(double)*threads.y*dust_density.Nd ;

    // Setup the blocks:
    dim3 blocks(
        1, // Must not have more than 1 block in x
        (_kernel.Nphi() + threads.y-1)/threads.y,
        _kernel.NR()
    ) ;


    _compute_coagulation_rate<<<blocks, threads, mem_size>>>(helper, dust_density, num_tracers, rate) ;
    check_CUDA_errors("_compute_coagulation_rate") ;

}

template class CoagulationRate<BirnstielKernel,SimpleErosion> ;
template class CoagulationRate<ConstantKernel,SimpleErosion> ;
template class CoagulationRate<BirnstielKernelVertInt,SimpleErosion> ;