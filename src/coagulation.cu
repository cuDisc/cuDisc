

#include <iostream>
#include <stdexcept>

#include "coagulation/coagulation.h"
#include "coagulation/fragments.h"
#include "dustdynamics.h"
#include "drag_const.h"

// Handle atomics across CUDA & HIP
#ifdef __HIP_PLATFORM_AMD__
    #define ATOMIC_ADD_BLOCK atomicAdd
    #define MIN fmin
#else
    #define ATOMIC_ADD_BLOCK atomicAdd_block
    #define MIN min
#endif

// An quadratic approximation to OC07's St*. Good to 2%
__host__ __device__
RealType St_star_OC07(RealType St) {
    if (St > 1)
        return 1 ;

    RealType a = 0.51505382481121010 ;
    RealType b = 0.12876345620280252 ;

    return St*(1 - b + sqrt(b*b + a*(1-St)/(1+St))) ;
}

// Ormel & Cuzzi Turbulent squared relative velocity.
//   Scaled to the R.M.S. turbulent speed (\sqrt{alpha} c_s)
//   Here St1 and St2 are the particle Stokes number, St_star_max is
//   St_star_OC07(max(St1, St2)) and sqrtRe_1 is the inverse of the square
//   root of the Reynolds number
__host__ __device__
RealType Vrel_sqd_OC07(RealType St1, RealType St2, RealType St_star_max, RealType sqrtRe_1) {

    RealType Sts = max(St_star_max, sqrtRe_1) ;

    auto V1 = [Sts](RealType St) -> RealType {
        return St * (St / (Sts + St) - St / (1 + St)) ;
    } ;
    auto V2 = [Sts, sqrtRe_1](RealType St) -> RealType {
        // return Sts*Sts / (St + Sts) - sqrtRe_1*sqrtRe_1/(St + sqrtRe_1) ;
        return (Sts-sqrtRe_1) +  (St*St) / (St + Sts) - (St*St) /(St + sqrtRe_1) ;
    } ;

    return max(((St1 - St2) / (St1 + St2)) * (V1(St1) - V1(St2)) + V2(St1) + V2(St2),0.) ;
}

__host__ __device__
RealType Vrel_sqd_OC07(RealType St1, RealType St2, RealType sqrtRe_1) {
    return Vrel_sqd_OC07(St1, St2, St_star_OC07(max(St1, St2)), sqrtRe_1) ;
}

template<bool use_full_stokes>
__device__ __host__
typename BirnstielKernel<use_full_stokes>::CellProps
BirnstielKernel<use_full_stokes>::cell_props(int i, int j) const {

    CellProps cell ;
    cell.rho = _wg(i,j).rho ;
    cell.cs = _cs(i,j) ;
    cell.R = _g.Rc(i) ;
    cell.alpha = _alpha_t(i,j) ;
    cell.mu = _mu(i,j) ;

    cell.Omega = sqrt(_GMstar/cell.R)/cell.R;
    RealType mfp = cell.mu * m_p / (cell.rho * 2.e-15);

    RealType sqrtRe = sqrt(cell.alpha * cell.cs / cell.Omega / mfp);
    cell.sqrtRe_1 = 1/sqrtRe ;

    return cell ;
}

template<bool use_full_stokes>
__device__ __host__
typename BirnstielKernel<use_full_stokes>::SizeProps
BirnstielKernel<use_full_stokes>::size_props(int i, int j, int k, const CellProps& cell) const {

    Grain grain = _sizes.grain_props(i,j,k);
    Prims wd = _wd(i,j,k) ;

    SizeProps s ;
    s.a = grain.a ;

    // Stokes number
    s.St = calc_t_s<use_full_stokes>(wd, _wg(i,j), s.a, grain.rho, cell.cs, cell.mu) * cell.Omega;
    s.St_star = St_star_OC07(s.St) ;

    s.m = 4.188790205f *  pow(grain.a, 3.) * grain.rho;
    s.i_to_t_rat = 1. - _sizes.base_mass(k)/s.m;

    s.v_R = wd.v_R ;
    s.v_Z = wd.v_Z ;
    s.v_phi = wd.v_phi ;

    return s ;
}

template<bool use_full_stokes>
__device__ __host__
KernelResult BirnstielKernel<use_full_stokes>::operator()(const CellProps& cell, const SizeProps& s1, const SizeProps& s2) const {

    // Step 0: Compute the geometric cross-section
    RealType xsec = M_PI * (s1.a + s2.a)*(s1.a + s2.a) ;

    // Step 1: Compute the turbulent velocity:
    RealType cs = cell.cs ;
    RealType tmp;

    //   1b: Compute the turbulent velocity
    RealType St_star_max = s1.St >= s2.St ? s1.St_star : s2.St_star ;
    RealType v_turb = cell.alpha * cs*cs * Vrel_sqd_OC07(s1.St, s2.St, St_star_max, cell.sqrtRe_1) ;

    // Protect against NaN at low gas density
    if (cell.rho == 0) v_turb = 0 ;

    //   1c: Compute brownian motion
    RealType m1 = s1.m, m2 = s2.m ;

    tmp = 4.2592967532662155e-24 * (cell.mu * (m1 + m2) / (m1*m2)) * cs*cs; //4.261679179e-24f

    v_turb += tmp;
    
    // Step 2: Add the laminar components in quadrature
    tmp = s1.v_R - s2.v_R;
    v_turb += tmp*tmp ;

    tmp = s1.v_Z - s2.v_Z ;
    v_turb += tmp*tmp ;

    tmp = s1.v_phi - s2.v_phi ;
    v_turb += tmp*tmp ;

    v_turb = sqrt(v_turb) ;

    // Step 3: Compute the kernel
    KernelResult result ;
    
    result.K = xsec * v_turb ;

    RealType _v_frag = _v_frag_b + (_v_frag_i-_v_frag_b) * min(1., 5.*s1.i_to_t_rat + 5.*s2.i_to_t_rat);

    result.p_frag = (1.5*(_v_frag/v_turb)*(_v_frag/v_turb) + 1.) * exp(-1.5*(_v_frag/v_turb)*(_v_frag/v_turb)); // From https://iopscience.iop.org/article/10.3847/1538-4357/ac7d58/pdf
    result.p_coag = 1. - result.p_frag;

    return result ;
}

template<bool use_full_stokes>
__device__ __host__
typename BirnstielKernelVertInt<use_full_stokes>::CellProps
BirnstielKernelVertInt<use_full_stokes>::cell_props(int i, int j) const {

    CellProps cell ;
    cell.Sig_g = _wg(i,j).Sig ;
    cell.cs = _cs(i,j) ;
    cell.R = _g.Rc(i) ;
    cell.alpha = _alpha_t(i,j) ;

    RealType cs = cell.cs, R = cell.R ;

    cell.Omega = sqrt(_GMstar/R)/R;
    RealType mfp = 2.5066f * (cs/cell.Omega) * _mu * m_p / (cell.Sig_g * 2.e-15);

    RealType sqrtRe = sqrt(cell.alpha * cs / cell.Omega / mfp);
    cell.sqrtRe_1 = 1/sqrtRe ;

    cell.Hp2 = cs*cs*R/_GMstar *R*R;

    return cell ;
}

template<bool use_full_stokes>
__device__ __host__
typename BirnstielKernelVertInt<use_full_stokes>::SizeProps
BirnstielKernelVertInt<use_full_stokes>::size_props(int i, int j, int k, const CellProps& cell) const {

    Grain grain = _sizes.grain_props(i,j,k);
    Prims1D wd = _wd(i,j,k) ;

    SizeProps s ;
    s.a = grain.a ;

    // Stokes number
    s.St = calc_t_s<use_full_stokes>(wd, _wg(i,j), s.a, grain.rho, cell.cs, _mu, cell.Omega) * cell.Omega;
    s.St_star = St_star_OC07(s.St) ;

    s.m = 4.188790205f *  pow(grain.a, 3.) * grain.rho;
    s.i_to_t_rat = 1. - _sizes.base_mass(k)/s.m;

    s.v_R = wd.v_R ;
    s.v_phi = wd.v_phi ;

    // Dust scale height squared, and the settling speed scaled by Omega/R
    s.h2 = cell.Hp2 /(1 + s.St/cell.alpha);
    s.v_settle = sqrt(s.h2)*min(s.St, 0.5) ;

    return s ;
}

template<bool use_full_stokes>
__device__ __host__
KernelResult BirnstielKernelVertInt<use_full_stokes>::operator()(const CellProps& cell, const SizeProps& s1, const SizeProps& s2) const {

    // Step 0: Compute the geometric cross-section
    RealType xsec = M_PI * (s1.a + s2.a)*(s1.a + s2.a) ;

    // Step 1: Compute the turbulent velocity:
    RealType cs = cell.cs, R = cell.R ;
    RealType tmp;

    //   1b: Compute the turbulent velocity
    RealType St_star_max = s1.St >= s2.St ? s1.St_star : s2.St_star ;
    RealType v_turb = cell.alpha * cs*cs * Vrel_sqd_OC07(s1.St, s2.St, St_star_max, cell.sqrtRe_1) ;

    // Protect against NaN at low gas density
    if (cell.Sig_g == 0) v_turb = 0 ;

    //   1c: Compute brownian motion
    RealType m1 = s1.m, m2 = s2.m ;

    tmp = 4.2592967532662155e-24 * (_mu * (m1 + m2) / (m1*m2)) * cs*cs; //4.261679179e-24f

    v_turb += tmp;
    
    // Step 2: Add the laminar components in quadrature
    tmp = s1.v_R - s2.v_R;
    v_turb += tmp*tmp ;

    tmp = s1.v_phi - s2.v_phi ;
    v_turb += tmp*tmp ;

    // Step 3: Compute the kernel
    KernelResult result ;

    tmp = pow(s1.v_settle - s2.v_settle, 2.)/(R*R) * _GMstar/(R);
    v_turb += tmp;
    v_turb = sqrt(v_turb) ;

    result.K = xsec * v_turb * 1./sqrt(2.*M_PI*(s1.h2+s2.h2));

    RealType _v_frag = _v_frag_b + (_v_frag_i-_v_frag_b) * min(1., 5.*s1.i_to_t_rat + 5.*s2.i_to_t_rat);

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
      return _id[j*stride + i] ;
  }

  __host__ __device__ 
  CoagulationCache::coeff Cijk(int i, int j) const {
      return _Cijk[j*stride + i] ;
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
    
    using SizeProps = typename Kernel::SizeProps ;

    _CoagulationRateHelper(
            const Kernel& _kernel, const SizeGrid& sizes, 
            const CoagulationCache& _cache, SizeProps* _props)
      : kernel(_kernel), grain_masses(sizes.grain_masses()), cache(_cache),
        props(_props), size(sizes.size())
    { } ;

    // Properties of size k in cell (i,j)
    __device__ __host__
    SizeProps& size_props(int i, int j, int k) const {
        return props[(i*kernel.Nphi() + j)*size + k] ;
    }

    Kernel kernel ;
    const RealType *grain_masses ;

    CoagulationCacheRef cache ;

    SizeProps* props ;

    int size ;
} ;

// Compute the properties of each grain size in each cell that the kernel needs
template<class Kernel, class Fragments>
__global__ void _compute_size_props(_CoagulationRateHelper<Kernel,Fragments> coag, FieldRef<bool> active) {

    int k0 = threadIdx.x + blockIdx.x*blockDim.x ;
    int iZ0 = threadIdx.y + blockIdx.y*blockDim.y ;
    int iR0 = threadIdx.z + blockIdx.z*blockDim.z ;

    for (int iR=iR0; iR < coag.kernel.NR(); iR += blockDim.z*gridDim.z)
        for (int iZ=iZ0; iZ < coag.kernel.Nphi(); iZ += blockDim.y*gridDim.y) {
            if (!active(iR,iZ)) continue ;

            auto cell = coag.kernel.cell_props(iR, iZ) ;
            for (int k=k0; k < coag.size; k += blockDim.x*gridDim.x)
                coag.size_props(iR, iZ, k) = coag.kernel.size_props(iR, iZ, k, cell) ;
        }
}



template<class Kernel, class Fragments>
__global__ void _compute_coagulation_rate(_CoagulationRateHelper<Kernel,Fragments> coag, 
                                          Field3DConstRef<double> dust_density, int num_tracers,
                                          Field3DRef<double> rate, FieldRef<bool> active) {

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

    // Check the bounds first: active has no entries beyond the grid
    bool cell_active = iR < coag.kernel.NR() && iZ < coag.kernel.Nphi() && active(iR,iZ) ;

    // Main coagulation loop
    if (cell_active) {
        auto cell = coag.kernel.cell_props(iR, iZ) ;

        for (int i=s0; i < coag.size; i += threads_per_cell) {
            double mi = coag.grain_masses[i] ;      
            double ni = dust_density(iR, iZ, i) / mi ;

            auto props_i = coag.size_props(iR, iZ, i) ;
        
            for (int j = 0; j < coag.size; ++j) {

                auto Kij = coag.kernel(cell, props_i, coag.size_props(iR, iZ, j)) ;

                // Kij.p_coag = 1.;
                // Kij.p_frag = 0.;

                double mj = coag.grain_masses[j] ;
                double nj = dust_density(iR, iZ, j) / mj ;
        
                double tot_rate = Kij.K * nj * ni ;
                
                if (tot_rate == 0) 
                    continue ;

                ATOMIC_ADD_BLOCK(&rate(iR, iZ, i), 
                                -tot_rate * mi * (Kij.p_coag + Kij.p_frag)) ;

                for (int t=1; t < num_tracers+1; t++) {
                    double tracer_rate = Kij.K * nj * 
                        dust_density(iR, iZ, i + t*coag.size) * (Kij.p_coag + Kij.p_frag) ;
                    ATOMIC_ADD_BLOCK(&rate(iR, iZ, i + t*coag.size), -tracer_rate) ;
                }

                // Kij.p_coag = 1.;
                // Coagulation using Brauer's method.
                if (Kij.p_coag > 0) {
                    double coag_rate = tot_rate * mi * Kij.p_coag ;
                
                    int k = coag.cache.index(i,j).coag ;
                    double f = coag.cache.Cijk(i,j).coag ;
                
                    if (k < coag.size)
                        ATOMIC_ADD_BLOCK(&rate(iR,iZ,k), f * coag_rate) ;
                    if (k + 1 < coag.size) 
                        ATOMIC_ADD_BLOCK(&rate(iR,iZ,k+1), (1 - f) * coag_rate) ;

                    for (int t=1; t < num_tracers+1; t++) {
                        double tracer_rate = Kij.K * nj *
                            dust_density(iR, iZ, i + t*coag.size) * Kij.p_coag ;

                        if (k < coag.size)
                            ATOMIC_ADD_BLOCK(&rate(iR, iZ, k + t*coag.size), f * tracer_rate) ;
                        if (k + 1 < coag.size) 
                            ATOMIC_ADD_BLOCK(&rate(iR, iZ, k + 1 + t*coag.size), (1-f) * tracer_rate) ;
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

                    // A remnant in the largest bin stays there, rather than leaving the grid
                    if (k_rem + 1 >= coag.size) eps = 1 ;

                    ATOMIC_ADD_BLOCK(&tmp[k],              frag_rate * ((mi - m_rem) + mj )) ;
                    ATOMIC_ADD_BLOCK(&rate(iR,iZ,k_rem),   frag_rate * (          m_rem) * eps) ;
                    if (k_rem + 1 < coag.size)
                        ATOMIC_ADD_BLOCK(&rate(iR,iZ,k_rem+1), frag_rate * (          m_rem) * (1-eps)) ;

                    // double rho_tot = dust_density(iR, iZ, i) + dust_density(iR, iZ, j) ;
                    for (int t=1; t < num_tracers+1; t++) {
                        double tracer_rate = frag_rate * (dust_density(iR, iZ, i + t*coag.size)/ni + dust_density(iR, iZ, j + t*coag.size)/nj)/(mi+mj) ;
                        
                        ATOMIC_ADD_BLOCK(&rate(iR,iZ,k_rem + t*coag.size),     tracer_rate * (          m_rem) * eps) ;
                        if (k_rem + 1 < coag.size)
                            ATOMIC_ADD_BLOCK(&rate(iR,iZ,k_rem + 1 + t*coag.size), tracer_rate * (          m_rem) * (1-eps)) ;
                        ATOMIC_ADD_BLOCK(&tmp[k + t*coag.size],                tracer_rate * ((mi - m_rem)+ mj)) ;
                    }
                }
            }
        }
    }

    // Every thread in the block must reach the barrier, so it is outside the branches
    __syncthreads() ;

    // Distribute the fragmentation products
    if (cell_active) {
        for (int j=s0; j < coag.size; j += threads_per_cell) {
            for (int t=0; t < num_tracers+1; t++)
                for (int i = 0; i < coag.size; ++i)
                    rate(iR,iZ,j+t*coag.size) += tmp[i + t*coag.size] * coag.cache.Cijk_frag(i,j) ;
        }
    }

}

template<class Kernel, class Fragments>
void CoagulationRate<Kernel,Fragments>::operator()(const Field3D<double>& dust_density,
                                                   Field3D<double>& rate, Field<bool>& active) const {


    if ((dust_density.Nd % _grain_sizes.size()) != 0)
        throw std::invalid_argument("Number of densities must be an integer multiple of the "
                                     "number of grain sizes.") ;

    int num_tracers =  dust_density.Nd / _grain_sizes.size() - 1 ; 

    if (dust_density.Nd > 49152/int(sizeof(double)))
        throw std::invalid_argument("CoagulationRate only supports < 6144 densities.");

    // Properties of each grain size in each cell
    using SizeProps = typename Kernel::SizeProps ;
    CudaArray<SizeProps> props =
        make_CudaArray<SizeProps>(_kernel.NR() * _kernel.Nphi() * _grain_sizes.size()) ;

    _CoagulationRateHelper<Kernel,Fragments> helper(
        _kernel,_grain_sizes, _cache, props.get()
    ) ;

    dim3 props_threads(32, 4, 2) ;
    dim3 props_blocks((_grain_sizes.size() + 31)/32,
                      (_kernel.Nphi() + 3)/4,
                      (_kernel.NR() + 1)/2) ;
    _compute_size_props<<<props_blocks, props_threads>>>(helper, active) ;
    check_CUDA_errors("_compute_size_props") ;

    // Setup Blocks / threads
    //   Make sure we fit at least 1 block into the shared memory - max. shared mem per thread block is 48kB
    dim3 threads(32, 16, 1) ;
    while (threads.y*dust_density.Nd > 49152/int(sizeof(double))) {
        threads.x *= 2 ;
        threads.y /= 2 ;
    }
    // If we can fit 2 blocks, make it so. - assuming 64kB per multiprocessor
    if (threads.y > 1 and threads.y*dust_density.Nd> 32768/int(sizeof(double))) {
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


    _compute_coagulation_rate<<<blocks, threads, mem_size>>>(helper, dust_density, num_tracers, rate, active) ;
    check_CUDA_errors("_compute_coagulation_rate") ;

}

template class CoagulationRate<BirnstielKernel<false>,SimpleErosion> ;
template class CoagulationRate<BirnstielKernel<true>,SimpleErosion> ;

template class CoagulationRate<BirnstielKernelVertInt<false>,SimpleErosion> ;
template class CoagulationRate<BirnstielKernelVertInt<true>,SimpleErosion> ;

template class CoagulationRate<ConstantKernel,SimpleErosion> ;
