

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
template<class Helper>
__global__ void _compute_size_props(Helper coag, FieldRef<bool> active) {

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

/* Threads per block for the coagulation rate kernels, which use 
 * shared_per_cell doubles of shared memory for each cell.
 */
static dim3 _coag_rate_threads(int shared_per_cell) {
    //   Make sure we fit at least 1 block into the shared memory - max. shared mem per thread block is 48kB
    dim3 threads(32, 16, 1) ;
    while (threads.y*shared_per_cell > 49152/int(sizeof(double))) {
        threads.x *= 2 ;
        threads.y /= 2 ;
    }
    // If we can fit 2 blocks, make it so. - assuming 64kB per multiprocessor
    if (threads.y > 1 and threads.y*shared_per_cell> 32768/int(sizeof(double))) {
        threads.x *= 2 ;
        threads.y /= 2 ;
    }
    // Reducing threads.y helps further avoid shared memory size problems.
    // If threads.x > 64 then then the occupancy will be the same whether
    // you have more blocks per thread or threads per block, so small blocks
    // shouldn't limit performance either.
    if (threads.x >= 64)
        threads.y = 1;

    return threads ;
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
    dim3 threads = _coag_rate_threads(dust_density.Nd) ;
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


/* CoagulationRateMHL
 *
 * Port of coag_toolkit's CoagMHL and FastFragmentationMHL. Each thread handles
 * one size, i, and loops over the larger sizes, j >= i. For coagulation the
 * kernel is interpolated across bin j (the larger particle) and for 
 * fragmentation across the target, which is also bin j. Thus both need the
 * slope of K(i, j) along j, which is computed from a sliding window of kernel
 * values.
 */

// Device view of CoagulationCacheMHL
class CoagulationCacheMHLRef {
  public:
    CoagulationCacheMHLRef(const CoagulationCacheMHL& c)
      : m(c.m.get()), me(c.me.get()), dm(c.dm.get()), dlgm(c.dlgm.get()),
        lgx(c.lgx.get()), xm1(c.xm1.get()),
        dlg_lo(c.dlg_lo.get()), dlg_up(c.dlg_up.get()),
        d2(c.d2.get()), d3(c.d3.get()), E2(c.E2.get()), E3(c.E3.get()),
        Cfrag(c.Cfrag.get()), pairs(c._pairs.get()),
        nodes(c.d_nodes.get()), runs(c.d_runs.get()), pieces(c.d_pieces.get()),
        segments(c.d_segments.get()), size(c.size())
    { } ;

    __host__ __device__
    const CoagulationCacheMHL::Pair& pair(int i, int j) const {
        return pairs[j*size + i] ;
    }

    const double *m, *me, *dm, *dlgm, *lgx, *xm1, *dlg_lo, *dlg_up ;
    const double *d2, *d3, *E2, *E3 ;
    const double *Cfrag ;
    const CoagulationCacheMHL::Pair* pairs ;
    const CoagulationCacheMHL::Node* nodes ;
    const CoagulationCacheMHL::RemnantRun* runs ;
    const CoagulationCacheMHL::RemnantPiece* pieces ;
    const CoagulationCacheMHL::FragSegment* segments ;
    int size ;
} ;

/* Power-law reconstruction of the density in a bin:
 *   Coagulation   : C (m/me)^(q-1) per unit mass, normalized to unit mass
 *   Fragmentation : Cf (m/me)^(qf-1), with q limited by q_min. 
 *                   Nf is the number of particles per unit mass.
 */
struct _MHLSlopes {
    double C, q, Cf, qf, Nf ;
} ;

template<class Kernel>
struct _CoagulationRateMHLHelper {

    using SizeProps = typename Kernel::SizeProps ;

    _CoagulationRateMHLHelper(
            const Kernel& _kernel, const CoagulationCacheMHL& _cache, 
            SizeProps* _props, _MHLSlopes* _slopes, int* _max_bin,
            double _q_min, double _tail_eps)
      : kernel(_kernel), cache(_cache), props(_props), slopes(_slopes), 
        max_bin(_max_bin), q_min(_q_min), tail_eps(_tail_eps), size(_cache.size())
    { } ;

    // Properties of size k in cell (i,j)
    __device__ __host__
    SizeProps& size_props(int i, int j, int k) const {
        return props[(i*kernel.Nphi() + j)*size + k] ;
    }
    // Slopes of size k in cell (i,j)
    __device__ __host__
    _MHLSlopes* cell_slopes(int i, int j) const {
        return slopes + (i*kernel.Nphi() + j)*size ;
    }
    // One past the last non-empty bin in cell (i,j)
    __device__ __host__
    int& last_bin(int i, int j) const {
        return max_bin[i*kernel.Nphi() + j] ;
    }

    Kernel kernel ;
    CoagulationCacheMHLRef cache ;

    SizeProps* props ;
    _MHLSlopes* slopes ;
    int* max_bin ;

    double q_min, tail_eps ;
    int size ;
} ;

// The argument of smaller magnitude if a and b have the same sign, else zero
__device__ inline
double _minmod(double a, double b) {
    if (a > 0 && b > 0) return min(a, b) ;
    if (a < 0 && b < 0) return max(a, b) ;
    return 0 ;
}

/* MC-limited logarithmic slope of the density across bin k, from the mass
 * densities per unit mass of bins k-1, k and k+1 (fm, f0, fp).
 *
 * As for the kernel (_MHL_kernel_slope), the central difference is limited so
 * that the reconstruction at the edges of the bin stays within the range of
 * the neighbouring values; power laws are unaffected. An empty neighbour
 * (ln 0 = -inf) constrains only the sign, which makes the slope continuous as
 * a neighbouring bin empties or fills. (coag_toolkit's unlimited slope instead
 * reuses the previous slope when a neighbour is exactly empty, so the rates
 * jump when it receives any mass, which forces many small coagulation steps.)
 * It also stops the slope depending on a neighbour that holds only
 * round-off-level mass.
 *
 * The first and last bins have only one neighbour:
 *   - First bin: the one-sided slope to bin 1, limited (minmod) by the slope
 *     between bins 1 and 2 (fpp), so that it is used only where it is
 *     consistent with the trend of the distribution. Power laws are again
 *     unaffected, while a nearly empty bin 1 gives q = 0.
 *   - Last bin: zero-gradient ghost cell (q = 0). A one-sided difference to a
 *     nearly empty neighbour made the rates of an isolated top bin vary
 *     erratically between integration stages.
 * |q| is also capped so that it stays finite.
 */
__device__ inline
double _MHL_limited_density_slope(double fm, double f0, double fp, double fpp, int k, int size,
                                  const CoagulationCacheMHLRef& c) {
    double q ;
    if (f0 <= 0 || k + 1 == size)
        q = 0 ;  // Empty bin (the slope is not used) or last bin
    else if (k == 0) {
        q = log(fp/f0) / c.dlgm[0] ;
        if (size > 2)
            q = _minmod(q, log(fpp/fp) / log(c.m[2]/c.m[1])) ;
        else
            q = 0 ;
    }
    else {
        q = log(fp/fm) / c.dlgm[k] ;   // NaN if both are empty: minmod gives 0
        q = _minmod(q, log(fp/f0) / c.dlg_up[k]) ;
        q = _minmod(q, log(f0/fm) / c.dlg_lo[k]) ;
    }
    double q_max = 600 / c.lgx[k] ;
    return max(-q_max, min(q, q_max)) ;
}

// Compute the power-law reconstruction of the density. One thread per cell.
template<class Helper>
__global__ void _compute_density_slopes_MHL(Helper coag, Field3DConstRef<double> rho, 
                                            FieldRef<bool> active) {

    int iZ = threadIdx.x + blockIdx.x*blockDim.x ;
    int iR = threadIdx.y + blockIdx.y*blockDim.y ;

    if (iR >= coag.kernel.NR() || iZ >= coag.kernel.Nphi() || !active(iR,iZ))
        return ;

    const CoagulationCacheMHLRef& c = coag.cache ;
    int size = coag.size ;
    _MHLSlopes* slopes = coag.cell_slopes(iR, iZ) ;

    // Mass density per unit mass
    auto minv = [&](int k) { return rho(iR,iZ,k) / c.dm[k] ; } ;
    // Integrate[ y^(q-1), {y, 1, me[k+1]/me[k]}]
    auto integ = [&](int k, double q) { 
        return (q == 0) ? c.lgx[k] : expm1(q * c.lgx[k]) / q ;
    } ;

    // Negative densities (e.g. in intermediate integration stages) count as empty
    auto f = [&](int b) { return max(minv(b), 0.) ; } ;

    int max_bin = 0 ;
    for (int k = 0; k < size; ++k) {
        double q = _MHL_limited_density_slope(f(max(k-1, 0)), f(k), f(min(k+1, size-1)),
                                              f(min(k+2, size-1)), k, size, c) ;
        // Fragmentation also limits the slope from below
        double qf = max(q, coag.q_min) ;

        _MHLSlopes s ;
        s.q = q ;
        s.C = 1 / (c.me[k] * integ(k, q + 1)) ;
        s.qf = qf ;
        s.Cf = 1 / (c.me[k] * integ(k, qf + 1)) ;
        s.Nf = s.Cf * integ(k, qf) ;
        slopes[k] = s ;

        if (rho(iR,iZ,k) != 0) max_bin = k ;
    }
    coag.last_bin(iR, iZ) = max_bin + 1 ;
}

/* Logarithmic slope of the kernel across bin j, given its values at the
 * centres of bins j-1, j and j+1 (km, k0, kp).
 *
 * The central difference (reusing the last slope if a neighbour vanishes, as
 * in coag_toolkit) is limited, as in the MC limiter, so that the power-law
 * interpolation does not exceed the range of the neighbouring values at the
 * edges of bin j:
 *    Q = minmod(Q_central, ln(kp/k0)/ln(me_{j+1}/m_j), ln(k0/km)/ln(m_j/me_j)).
 * Using the distances from the centre to the edges, rather than half the
 * spacing of the centres, keeps this bound exact for the (arithmetic) bin
 * centres. Without it, the kernel can be extrapolated by many e-folds across
 * a bin where it varies rapidly, e.g. the onset of fragmentation for similar
 * sizes. Power-law kernels are unaffected, except in the first and last bins:
 * these have no outer neighbour to bound the extrapolation, so they are
 * treated as having zero-gradient ghost cells (Q = 0).
 *
 * The interpolated kernel is evaluated as a small coefficient times a large
 * integral, so |Q| is also capped at |Q| ln(me_{j+1}/me_j) <= 600 to keep
 * both finite. Reducing |Q| keeps the edge values within the bounds.
 */
__device__ inline
double _MHL_kernel_slope(double km, double k0, double kp, bool interior,
                         double dlgm, double dlg_lo, double dlg_up, double& Q_save) {
    double Q = Q_save ;
    if (km > 0 && kp > 0)
        Q = Q_save = log(kp/km) / dlgm ;

    if (not interior) return 0 ;

    // A vanishing neighbour gives an infinite bound, constraining only the sign
    if (k0 > 0) {
        Q = _minmod(Q, log(kp/k0) / dlg_up) ;
        Q = _minmod(Q, log(k0/km) / dlg_lo) ;
    }

    double Q_max = 600 / (dlg_lo + dlg_up) ;
    return max(-Q_max, min(Q, Q_max)) ;
}

/* _MHLCell
 *
 * Computes the coagulation/fragmentation rates for pairs of bins in a cell.
 *
 * Tracers follow the mass: mass leaving bin a carries the tracer fractions
 * of bin a, while the fragmentation products carry the mass-weighted mean
 * fractions of the colliding particles.
 */
struct _MHLCell {
    using Node = CoagulationCacheMHL::Node ;

    CoagulationCacheMHLRef c ;
    Field3DConstRef<double> rho ;
    Field3DRef<double> rate ;
    const _MHLSlopes* slopes ;
    double *frag, *D0, *D1 ;
    int iR, iZ, size, nt ;
    double tail_eps ;

    __device__ double density(int k) const {
        return rho(iR, iZ, k) ;
    }

    __device__ void add(int k, double dm) {
        ATOMIC_ADD_BLOCK(&rate(iR, iZ, k), dm) ;
    }

    // Mass fraction of tracer t in bin a
    __device__ double tracer_fraction(int a, int t) const {
        return density(a + t*size) / density(a) ;
    }

    // Mass dm arrives in bin k (k >= size is off the grid) from bin a
    __device__ void gain(int k, double dm, int a) {
        if (k >= size) return ;
        add(k, dm) ;
        for (int t=1; t < nt+1; t++)
            add(k + t*size, dm * tracer_fraction(a, t)) ;
    }

    // Mass dm is removed from bin a
    __device__ void lose(int a, double dm) {
        add(a, -dm) ;
        for (int t=1; t < nt+1; t++)
            add(a + t*size, -dm * tracer_fraction(a, t)) ;
    }

    /* Fragmentation products carry the tracer mass fractions
     *    ft * x_t + fp * x_p,
     * where x is the tracer fraction in the target / projectile bins and
     * ft, fp are their shares of the mass processed.
     */
    struct Mix {
        int t, p ;
        double ft, fp ;
    } ;

    // Add the mass dm with composition mix to bin k of the array arr
    __device__ void deposit(double* arr, int k, double dm, const Mix& mix) {
        ATOMIC_ADD_BLOCK(&arr[k], dm) ;
        for (int t=1; t < nt+1; t++)
            ATOMIC_ADD_BLOCK(&arr[k + t*size], 
                dm * (mix.ft * tracer_fraction(mix.t, t) + mix.fp * tracer_fraction(mix.p, t))) ;
    }

    // Integrals of the reconstructed density from the lower edge of the bin to y
    __device__ void integrals(const Node& y, double ci, double qe, double& P0, double& P1) const {
        if (y.lgy == 0) 
            P0 = P1 = 0 ;
        else
            power_law_integ_pair(ci, y.lgy, y.ym1, qe, P0, P1) ;
    }

    /* coag_pair
     *
     * Coagulation of bins i <= j, with the kernel 
     *    K(m_i, m) = ratec (m/me_j)^Q
     * across bin j. Returns the mass lost by bin i, which is accumulated by
     * the caller.
     */
    __device__ double coag_pair(int i, int j, double ratec, double Q) {
        const CoagulationCacheMHL::Pair& p = c.pair(i, j) ;

        double rr = density(i) * density(j) ;

        double cj = slopes[j].C ;
        double qe = slopes[j].q + Q ;
        double qep1 = qe + 1 ;

        // Avoid double counting
        if (i == j) ratec /= 2 ;

        /* The first range starts at the lower edge of bin j, where the
         * integrals are zero.
         */
        const Node* node = c.nodes + p.node_start ;

        double P0_l = 0, P1_l = 0 ;
        double loss_i = 0 ;
        for (int k = p.kmin; k < p.kmax; k++) {
            // Integrals at the upper boundary of the range
            const Node& nu = node[k - p.kmin + 1] ;

            double P0_u, P1_u ;
            double em1_u = power_law_integ_pair(cj, nu.lgy, nu.ym1, qe, P0_u, P1_u) ;

            double from_i, from_j = 0 ;
            if (j == p.kmin and k == j+1 and c.m[i] < 1e-7*c.me[j+1]) {
                /* Use a series expansion because there is only a tiny amount of
                   mass in bin j+1 */
                double ratio = c.m[i] / c.me[j+1] ;

                // Mass moving from bin i to bin k
                double coeff = ratec * cj * y_pow_q(em1_u, nu.lgy, qe) ;
                from_i = coeff * ratio*(1 - 0.5*(qe-1)*ratio*(1 - (qe-2)*ratio/3.)) * rr ;

                // Mass moving from bin j to bin k
                coeff *= (c.me[j]/c.m[i]) * (c.me[j+1] / c.me[j]) ;
                from_j = coeff * ratio*(1 - 0.5*(qep1-1)*ratio*(1 - (qep1-2)*ratio/3.)) * rr ;
            }
            else {
                // Use the full expression
                from_i = ratec * (P0_u - P0_l) * rr ;
                if (j != k)
                    from_j = (ratec/c.m[i]) * c.me[j] * (P1_u - P1_l) * rr ;
            }

            loss_i += from_i ;
            gain(k, from_i, i) ;

            if (from_j != 0) {
                lose(j, from_j) ;
                gain(k, from_j, j) ;
            }

            // The upper boundary of this range is the lower one of the next
            P0_l = P0_u ;
            P1_l = P1_u ;
        }
        return loss_i ;
    }

    /* remnant_run
     *
     * Remnant mass of a run, deposited into the destination bins. The pieces
     * are evaluated exactly down to a cut chosen so that the integrand's 
     * weight, (m/m0)^a, is flat to within tail_eps below it. The mass below
     * the cut is computed exactly, but is distributed with the analytic shape
     * dM ~ r (1 + beta r) dr in a second stage.
     *
     * The integrals at each boundary are computed once and shared by the
     * neighbouring pieces, so the pieces sum to the integral over the run.
     */
    __device__ double remnant_run(const CoagulationCacheMHL::RemnantRun& run,
                                  double ci, double qe, double rate_f, const Mix& mix) {
        double a = qe - 1 ;
        bool use_tail = run.tail && tail_eps > 0 ;
        // Cut where me[k] |a / (R1 m0)| <= tail_eps
        double cut = use_tail ? tail_eps / fabs(a * run.inv_R1m0) : 0 ;

        double P0_b, P1_b ;
        integrals(run.start, ci, qe, P0_b, P1_b) ;

        double total = 0 ;
        for (int n = run.piece_start; n < run.piece_end; n++) {
            const CoagulationCacheMHL::RemnantPiece& piece = c.pieces[n] ;

            double P0_e, P1_e ;
            integrals(piece.end, ci, qe, P0_e, P1_e) ;

            double dm = run.dir * rate_f * 
                (run.R0 * (P0_e - P0_b) + run.R1me * (P1_e - P1_b)) ;
            deposit(&rate(iR, iZ, 0), piece.k, dm, mix) ;
            total += dm ;

            P0_b = P0_e ;
            P1_b = P1_e ;

            int k = piece.k ;
            if (use_tail && k > 0 && n + 1 < run.piece_end && c.me[k] <= cut) {
                // The tail: remnants in 0 < r < me[k], from here to the end of the run
                double P0_z, P1_z ;
                integrals(run.end, ci, qe, P0_z, P1_z) ;

                double M = run.dir * rate_f * 
                    (run.R0 * (P0_z - P0_b) + run.R1me * (P1_z - P1_b)) ;

                if (M > 0) {
                    // Split into the flat and linear (in r) parts
                    double s = (2/3.) * a * c.me[k] * run.inv_R1m0 ;
                    double M1 = M * s / (1 + s) ;
                    double M0 = M - M1 ;

                    deposit(D0, k, M0 / c.E2[k], mix) ;
                    deposit(D1, k, M1 / c.E3[k], mix) ;
                }
                else // Round-off: keep the mass, but avoid negative shapes
                    deposit(&rate(iR, iZ, 0), k, M, mix) ;

                total += M ;
                break ;
            }
        }
        return total ;
    }

    /* frag_pair
     *
     * Fragmentation of the target bin t by the (point mass) projectile bin
     * p <= t, with the kernel K(m, m_p) = KF (m/me_t)^Q across bin t. KF
     * includes the factor 1/2 for t == p. Returns the mass lost by bin p, 
     * which is accumulated by the caller.
     */
    __device__ double frag_pair(int p, int t, double KF, double Q) {
        // Collision rate, per unit mass of the target in bin t
        double rate_f = KF * density(t) * density(p) / c.m[p] ;

        const _MHLSlopes& s = slopes[t] ;
        double ci = s.Cf ;
        double qe = s.qf + Q ;
        double me_t = c.me[t] ;

        /* Losses: integrals over the whole of bin t. For a constant kernel
         * these follow from the normalization of the density.
         */
        double loss_t, loss_p ;
        if (Q == 0) {
            loss_t = rate_f ;
            loss_p = rate_f * c.m[p] * s.Nf ;
        }
        else {
            double P0, P1 ;
            power_law_integ_pair(ci, c.lgx[t], c.xm1[t], qe, P0, P1) ;
            loss_t = rate_f * me_t * P1 ;
            loss_p = rate_f * c.m[p] * P0 ;
        }
        lose(t, loss_t) ;

        double loss_tot = loss_t + loss_p ;
        Mix mix = {t, p, 0, 0} ;
        if (nt > 0 && loss_tot != 0) {
            mix.ft = loss_t / loss_tot ;
            mix.fp = loss_p / loss_tot ;
        }

        /* Remnants and fragments, segment by segment. Each segment's
         * fragments are the mass processed by its collisions less its remnant;
         * those of the last segment take the remaining mass budget.
         */
        const CoagulationCacheMHL::Pair& pr = c.pair(p, t) ;

        double P0_seg = 0, P1_seg = 0 ;   // at the start of the segment
        double m_done = 0 ;               // mass processed by earlier segments
        int r = pr.run_start ;
        for (int g = pr.seg_start; g < pr.seg_end; ++g) {
            const CoagulationCacheMHL::FragSegment& seg = c.segments[g] ;

            double rem_seg = 0 ;
            for (; r < seg.run_end; ++r)
                rem_seg += remnant_run(c.runs[r], ci, qe, rate_f, mix) ;

            if (g + 1 == pr.seg_end) {
                // Fragments of the last segment take the remaining mass
                deposit(frag, seg.s, loss_tot - m_done - rem_seg, mix) ;
                break ;
            }

            // Mass processed by the collisions of this segment
            double P0_hi, P1_hi ;
            integrals(seg.end, ci, qe, P0_hi, P1_hi) ;

            double m_seg = rate_f * (me_t * (P1_hi - P1_seg) + c.m[p] * (P0_hi - P0_seg)) ;
            P0_seg = P0_hi ;
            P1_seg = P1_hi ;

            deposit(frag, seg.s, m_seg - rem_seg, mix) ;
            m_done += m_seg ;
        }

        return loss_p ;
    }
} ;

/* Neumaier's compensated sum of arr[k] for k = last, last-1, ..., first. 
 * Summing downwards adds the (typically) smaller terms first.
 */
__device__ inline
double _suffix_sum(const double* arr, int first, int last) {
    double sum = 0, comp = 0 ;
    for (int k = last; k >= first; --k) {
        double v = arr[k] ;
        double tmp = sum + v ;
        if (fabs(sum) >= fabs(v))
            comp += (sum - tmp) + v ;
        else
            comp += (v - tmp) + sum ;
        sum = tmp ;
    }
    return sum + comp ;
}

template<class Kernel>
__global__ void _compute_coagulation_rate_MHL(_CoagulationRateMHLHelper<Kernel> coag, 
                                              Field3DConstRef<double> dust_density, int num_tracers,
                                              Field3DRef<double> rate, FieldRef<bool> active) {

    int s0 = threadIdx.x + blockIdx.x*blockDim.x ;
    int iZ = threadIdx.y + blockIdx.y*blockDim.y ;
    int iR = threadIdx.z + blockIdx.z*blockDim.z ;

    int threads_per_cell = blockDim.x*gridDim.x ;

    int size = coag.size ;
    int Nd = size * (1 + num_tracers) ;

    /* Shared space for the second stage: 
     *    frag : the fragments, by scale bin
     *    D0/1 : the remnant tails, by the bin, K, below which they lie
     */
    extern __shared__ double shared_mem[] ;
    double *frag = nullptr, *D0 = nullptr, *D1 = nullptr ;

    if (iR < coag.kernel.NR() && iZ < coag.kernel.Nphi()) {
        frag = shared_mem + 3 * (threadIdx.y + threadIdx.z * blockDim.y) * Nd ;
        D0 = frag + Nd ;
        D1 = D0 + Nd ;

        // Initialize the storage
        for (int k=s0; k < Nd; k += threads_per_cell) {
            frag[k] = D0[k] = D1[k] = 0 ;
            rate(iR,iZ,k) = 0 ;
        }
    } 
    __syncthreads() ;

    // Check the bounds first: active has no entries beyond the grid
    bool cell_active = iR < coag.kernel.NR() && iZ < coag.kernel.Nphi() && active(iR,iZ) ;

    // Stage 1: coagulation, losses, remnants, and fragment / tail totals
    if (cell_active) {
        auto cell = coag.kernel.cell_props(iR, iZ) ;

        _MHLCell C = {
            coag.cache, dust_density, rate, coag.cell_slopes(iR, iZ), 
            frag, D0, D1, iR, iZ, size, num_tracers, coag.tail_eps
        } ;
        const CoagulationCacheMHLRef& c = coag.cache ;

        int max_bin = coag.last_bin(iR, iZ) ;
        for (int i=s0; i < max_bin; i += threads_per_cell) {
            auto props_i = coag.size_props(iR, iZ, i) ;
            double rho_i = dust_density(iR, iZ, i) ;

            // Sliding window of K(i, j-1), K(i, j), K(i, j+1)
            KernelResult K0 = coag.kernel(cell, props_i, props_i) ;
            KernelResult Km = (i > 0) ? coag.kernel(cell, props_i, coag.size_props(iR, iZ, i-1)) : K0 ;
            KernelResult Kp ;

            double Qc_save = 0, Qf_save = 0 ;
            double loss_i = 0 ;
            for (int j = i; j < max_bin; ++j) {
                if (j + 1 < size)
                    Kp = coag.kernel(cell, props_i, coag.size_props(iR, iZ, j+1)) ;

                // Neighbours used for the slope across bin j
                const KernelResult& Kl = (j == 0) ? K0 : Km ;
                const KernelResult& Ku = (j + 1 == size) ? K0 : Kp ;

                bool interior = j > 0 && j + 1 < size ;
                double Qc = _MHL_kernel_slope(Kl.K * Kl.p_coag, K0.K * K0.p_coag, Ku.K * Ku.p_coag,
                                              interior, c.dlgm[j], c.dlg_lo[j], c.dlg_up[j], Qc_save) ;
                double Qf = _MHL_kernel_slope(Kl.K * Kl.p_frag, K0.K * K0.p_frag, Ku.K * Ku.p_frag,
                                              interior, c.dlgm[j], c.dlg_lo[j], c.dlg_up[j], Qf_save) ;

                if (rho_i != 0 && dust_density(iR, iZ, j) != 0) {
                    // K(m_i, m) = K_ij (m/m_j)^Q = K_ij (me_j/m_j)^Q (m/me_j)^Q across bin j
                    double Kc = K0.K * K0.p_coag ;
                    if (Kc > 0) {
                        if (Qc != 0) Kc *= pow(c.me[j] / c.m[j], Qc) ;
                        loss_i += C.coag_pair(i, j, Kc, Qc) ;
                    }

                    double Kf = K0.K * K0.p_frag ;
                    if (Kf > 0) {
                        if (Qf != 0) Kf *= pow(c.me[j] / c.m[j], Qf) ;
                        if (i == j) Kf /= 2 ;
                        loss_i += C.frag_pair(i, j, Kf, Qf) ;
                    }
                }

                Km = K0 ;
                K0 = Kp ;
            }

            if (loss_i != 0) 
                C.lose(i, loss_i) ;
        }
    }

    // Every thread in the block must reach the barrier, so it is outside the branches
    __syncthreads() ;

    // Stage 2: distribute the fragments and the remnant tails
    if (cell_active) {
        const CoagulationCacheMHLRef& c = coag.cache ;
        for (int k=s0; k < size; k += threads_per_cell) {
            for (int t=0; t < num_tracers+1; t++) {
                double* fr = frag + t*size ;
                double gain = 0 ;
                for (int s = 0; s < size; ++s)
                    gain += fr[s] * c.Cfrag[s*size + k] ;

                // Tails below me[K] for all K > k
                if (k + 1 < size)
                    gain += c.d2[k] * _suffix_sum(D0 + t*size, k+1, size-1) +
                            c.d3[k] * _suffix_sum(D1 + t*size, k+1, size-1) ;

                rate(iR,iZ,k+t*size) += gain ;
            }
        }
    }
}

template<class Kernel, class Fragments>
void CoagulationRateMHL<Kernel,Fragments>::operator()(const Field3D<double>& dust_density,
                                                      Field3D<double>& rate, Field<bool>& active) const {

    int size = _grain_sizes.size() ;

    if ((dust_density.Nd % size) != 0)
        throw std::invalid_argument("Number of densities must be an integer multiple of the "
                                     "number of grain sizes.") ;
    if (size < 2)
        throw std::invalid_argument("CoagulationRateMHL requires at least 2 grain sizes.") ;

    int num_tracers =  dust_density.Nd / size - 1 ; 

    // Shared memory for the fragments and both tail arrays
    int shared_per_cell = 3 * dust_density.Nd ;
    if (shared_per_cell > 49152/int(sizeof(double)))
        throw std::invalid_argument("CoagulationRateMHL only supports < 2048 densities.");

    int NR = _kernel.NR(), Nphi = _kernel.Nphi() ;

    // Properties of each grain size in each cell, and the density reconstruction
    using SizeProps = typename Kernel::SizeProps ;
    CudaArray<SizeProps> props = make_CudaArray<SizeProps>(NR * Nphi * size) ;
    CudaArray<_MHLSlopes> slopes = make_CudaArray<_MHLSlopes>(NR * Nphi * size) ;
    CudaArray<int> max_bin = make_CudaArray<int>(NR * Nphi) ;

    _CoagulationRateMHLHelper<Kernel> helper(
        _kernel, _cache, props.get(), slopes.get(), max_bin.get(), _q_min, _tail_eps
    ) ;

    dim3 props_threads(32, 4, 2) ;
    dim3 props_blocks((size + 31)/32, (Nphi + 3)/4, (NR + 1)/2) ;
    _compute_size_props<<<props_blocks, props_threads>>>(helper, active) ;
    check_CUDA_errors("_compute_size_props") ;

    dim3 slope_threads(32, 4) ;
    dim3 slope_blocks((Nphi + 31)/32, (NR + 3)/4) ;
    _compute_density_slopes_MHL<<<slope_blocks, slope_threads>>>(helper, dust_density, active) ;
    check_CUDA_errors("_compute_density_slopes_MHL") ;

    // Setup Blocks / threads, within the limit set by the kernel's register use
    dim3 threads = _coag_rate_threads(shared_per_cell) ;

    cudaFuncAttributes attr ;
    cudaFuncGetAttributes(&attr, _compute_coagulation_rate_MHL<Kernel>) ;
    check_CUDA_errors("cudaFuncGetAttributes") ;
    while (int(threads.x*threads.y) > attr.maxThreadsPerBlock && threads.y > 1)
        threads.y /= 2 ;
    while (int(threads.x*threads.y) > attr.maxThreadsPerBlock)
        threads.x /= 2 ;

    int mem_size = sizeof(double)*threads.y*shared_per_cell ;

    dim3 blocks(
        1, // Must not have more than 1 block in x
        (Nphi + threads.y-1)/threads.y,
        NR
    ) ;

    _compute_coagulation_rate_MHL<<<blocks, threads, mem_size>>>(
        helper, dust_density, num_tracers, rate, active) ;
    check_CUDA_errors("_compute_coagulation_rate_MHL") ;
}

template class CoagulationRateMHL<BirnstielKernel<false>,SimpleErosion> ;
template class CoagulationRateMHL<BirnstielKernel<true>,SimpleErosion> ;

template class CoagulationRateMHL<BirnstielKernelVertInt<false>,SimpleErosion> ;
template class CoagulationRateMHL<BirnstielKernelVertInt<true>,SimpleErosion> ;

template class CoagulationRateMHL<ConstantKernel,SimpleErosion> ;
