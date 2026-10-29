
#ifndef _CUDISC_HEADERS_COAGULATION_COAGULATION_H_
#define _CUDISC_HEADERS_COAGULATION_COAGULATION_H_

#include <cmath>
#include <vector>

#include "cuda_array.h"
#include "field.h"
#include "grid.h"
#include "star.h"

#include "coagulation/size_grid.h"
#include "coagulation/kernels.h"
#include "coagulation/fragments.h"

class CoagulationCacheRef ;

class CoagulationCache {
  public:

    CoagulationCache(int size, int block_size=-1) {   
        if (block_size < 0) { // Use a power of 2, up to 16 bytes
            block_size = 1 ; 
            int max_size = 16 ;
            while (block_size < size && block_size < max_size)
                block_size *= 2 ;
        }
        // Setup storage, padding the storage in j to be a multiple of block_size:
        stride = block_size * ((size + block_size-1) / block_size) ;

        _id = make_CudaArray<id>(size * stride) ;
        _Cijk = make_CudaArray<coeff>(size * stride) ;
        _Cijk_frag = make_CudaArray<double>(size * stride) ;
    }

    /* index and Cijk are stored with i varying fastest, as the threads that
     * use them handle consecutive sizes i for the same j. Cijk_frag is read
     * with consecutive j, so it is stored with j varying fastest.
     */
    struct id {
        int coag, frag, remnant ;
    } ;
    struct coeff {
        double coag, remnant, eps ;
    } ;

    id& index(int i, int j) {
        return _id[j*stride + i] ;
    }
    id index(int i, int j) const {
        return _id[j*stride + i] ;
    }

    coeff& Cijk(int i, int j) {
        return _Cijk[j*stride + i] ;
    }
    coeff Cijk(int i, int j) const {
        return _Cijk[j*stride + i] ;
    }

    double& Cijk_frag(int i, int j) {
        return _Cijk_frag[i*stride + j] ;
    } 
    double Cijk_frag(int i, int j) const {
        return _Cijk_frag[i*stride + j] ;
    }


  private:
    CudaArray<id> _id ;
    CudaArray<coeff> _Cijk ;
    CudaArray<double> _Cijk_frag ;

    int stride ;

    friend class CoagulationCacheRef ;
} ;

template<class Kernel, class Fragments>
class CoagulationRate {
  public:

    CoagulationRate(SizeGrid& grains, Kernel kernel, Fragments fragments)
     : _kernel(kernel), _grain_sizes(grains), _cache(grains.size())
    {
        _set_coagulation_properties() ;
        _set_fragment_properties(fragments) ;
    }

    void set_kernel(Kernel kernel) {
        _kernel = kernel ;
    }

    void set_fragments(Fragments fragments) {
        _set_fragment_properties(fragments) ;
    }

    void operator()(const Field3D<double>& dust_density, Field3D<double>& rate, Field<bool>& active) const ;

    struct id {
        int coag, frag, remnant, eps ;
    } ;
    struct coeff {
        double coag, remnant ;
    } ;

  private:
    Kernel _kernel ;
    SizeGrid& _grain_sizes ;

    // Helpers for coagulation / fragmentation
    CoagulationCache _cache ;

    void _set_coagulation_properties() ;
    void _set_fragment_properties(Fragments) ;
} ;

template<class Kernel, class Fragments>
CoagulationRate<Kernel, Fragments> create_coagulation_rate(SizeGrid& sizes, Kernel kernel, Fragments fragments) {
    return CoagulationRate<Kernel, Fragments>(sizes, std::move(kernel), std::move(fragments)) ;
}


/* Power-law reconstruction helpers for CoagulationRateMHL (ported from
 * coag_toolkit, MHL_coag.h/cpp).
 */

/* power_law_integ
 *
 * Safe evaluation of the integral:
 *    ci * Integrate[ y^(q-1), {y, 1, x}]
 * about q ~ 0.
 */
__host__ __device__ inline
double power_law_integ(double ci, double x, double q) {
    const double log_x = log(x) ;
    if (q == 0.0)
        return ci * log_x ;

    return ci * expm1(q * log_x) / q ;
}

/* power_law_integ_pair
 *
 * Evaluates both power_law_integ(ci, y, q) and power_law_integ(ci, y, q+1),
 * given log(y) and y-1, so that the (fixed) integration limits can be cached
 * and only one call to expm1 is needed in most cases.
 *
 * Returns expm1(q*log(y)).
 */
__host__ __device__ inline
double power_law_integ_pair(double ci, double lgy, double ym1,
                            double q, double& P0, double& P1) {
    double em1 = expm1(q * lgy) ;

    if (q == 0.0)
        P0 = ci * lgy ;
    else
        P0 = ci * em1 / q ;

    // expm1((q+1)*lgy) = expm1(q*lgy)*y + (y-1), unless this would cancel
    double qp1 = q + 1 ;
    if (qp1 == 0.0)
        P1 = ci * lgy ;
    else if (fabs(qp1 * lgy) < 0.1)
        P1 = ci * expm1(qp1 * lgy) / qp1 ;
    else
        P1 = ci * (em1 * (1 + ym1) + ym1) / qp1 ;

    return em1 ;
}

/* y_pow_q
 *
 * Computes y^q, given expm1(q*log(y)). Uses 1 + expm1 unless y^q << 1, where
 * it would lose all precision (e.g. in steep tails of the distribution).
 */
__host__ __device__ inline
double y_pow_q(double em1, double lgy, double q) {
    if (em1 > -0.5)
        return 1 + em1 ;
    return exp(q * lgy) ;
}

/* CoagulationCacheMHL
 *
 * Grid (and fragment distribution) dependent coefficients for
 * CoagulationRateMHL. Pairs of bins are indexed by (i, j) with i <= j.
 */
class CoagulationCacheMHL {
  public:
    CoagulationCacheMHL(int size) ;

    /* Pair
     *
     * Coagulation of bins i and j (i <= j) produces masses in bins
     * kmin <= k < kmax (k == size is off the grid). The boundaries of the
     * ranges of masses in bin j that produce each k are
     * nodes[node_start + k - kmin] and the one following.
     *
     * Fragmentation of the target, j, by the projectile, i, is described by
     * the segments [seg_start, seg_end), whose remnant runs begin at
     * run_start.
     */
    struct Pair {
        int kmin, kmax, node_start ;
        int seg_start, seg_end, run_start ;
    } ;

    /* Node
     *
     * A mass, m, in a bin with lower edge me, stored as log(m/me), m/me - 1.
     */
    struct Node {
        double lgy, ym1 ;
    } ;

    /* RemnantRun
     *
     * The remnant pieces of a sub-interval of the target bin over which the
     * remnant mass is affine, m_rem = R0 + R1*m (R1me = R1*me). The pieces
     * [piece_start, piece_end) are ordered from the largest remnant down
     * (walk order). The walk starts at the target mass 'start' and ends at
     * 'end'; dir = +1 (-1) if the walk is in the direction of increasing
     * (decreasing) target mass.
     *
     * If tail is true, the remnant falls to zero at the end of the walk,
     * m0, and inv_R1m0 = 1 / (R1 * m0).
     */
    struct RemnantRun {
        int piece_start, piece_end ;
        double R0, R1me ;
        Node start, end ;
        double dir, inv_R1m0 ;
        bool tail ;
    } ;

    /* RemnantPiece
     *
     * The remnant of the target masses between the end of the previous piece
     * (or the start of the run) and 'end' falls in bin k.
     */
    struct RemnantPiece {
        int k ;
        Node end ;
    } ;

    /* FragSegment
     *
     * A sub-interval of the target bin, ending at 'end', whose fragments have
     * the scale bin s. Its remnant runs end at run_end.
     */
    struct FragSegment {
        int s ;
        Node end ;
        int run_end ;
    } ;

    Pair& pair(int i, int j) {
        return _pairs[j*_size + i] ;
    }

    int size() const {
        return _size ;
    }

    /* Per-bin properties. dlg_lo / dlg_up are the logarithmic distances from 
     * the centre of the bin to its lower / upper edge. */
    CudaArray<double> m, me, dm, dlgm, lgx, xm1, dlg_lo, dlg_up ;

    /* Shape of the remnant tails: the tail below me[K] has mass fractions
     *   d2[k]/E2[K] and d3[k]/E3[K]
     * in bins k < K for its flat and linear parts.
     */
    CudaArray<double> d2, d3, E2, E3 ;

    // Fragment distribution, Cfrag[s*size + k], for scale bin s
    CudaArray<double> Cfrag ;

    CudaArray<Pair> _pairs ;

    std::vector<Node> nodes ;
    std::vector<RemnantRun> runs ;
    std::vector<RemnantPiece> pieces ;
    std::vector<FragSegment> segments ;

    // Device copies of the variable length tables
    CudaArray<Node> d_nodes ;
    CudaArray<RemnantRun> d_runs ;
    CudaArray<RemnantPiece> d_pieces ;
    CudaArray<FragSegment> d_segments ;

    // Copy the variable length tables to the device
    void upload() ;

  private:
    int _size ;
} ;

/* CoagulationRateMHL
 *
 * Coagulation/fragmentation rates using the method of Lee (2000), following
 * Booth et al. (2018) and coag_toolkit (CoagMHL and FastFragmentationMHL).
 *
 * The density is reconstructed as a power law within each bin, and the kernel
 * is interpolated as a power law across the larger particle's bin. The
 * integrals over the ranges of masses that produce each coagulation product or
 * fragmentation remnant are then exact. This greatly reduces the numerical
 * diffusion compared with CoagulationRate.
 *
 * Unlike coag_toolkit, the slopes of both power laws are limited (MC type) so
 * that the reconstruction does not exceed the range of the values in the
 * neighbouring bins. In the first and last bins the kernel is flat, as is the
 * density in the last bin; the density in the first bin uses the one-sided
 * slope where it is consistent with the slope between bins 1 and 2:
 *   - Kernel: where the kernel varies by many e-folds between bins (e.g. the
 *     onset of fragmentation of similar sized particles), the unlimited
 *     extrapolation to the bin edges can be wrong by many orders of magnitude.
 *   - Density: the limited slope is continuous as a neighbouring bin empties
 *     or fills, and is insensitive to bins holding round-off-level mass. The
 *     unlimited slope is neither, which forces many small coagulation steps.
 *     Power-law parts of the distribution are unaffected.
 *
 * Remnants that fall towards zero mass (e.g. erosion by similar mass
 * particles) would otherwise be spread over every bin below them. Below a
 * cut, chosen in each cell from the slope of the reconstruction so that the
 * weight of the integrand is nearly flat, the mass of the tail is still
 * computed exactly, but it is distributed with a fixed analytic shape (to
 * first order in the slope). remnant_tail_eps controls the cut; zero
 * computes all remnant pieces exactly, as in coag_toolkit.
 *
 * Params:
 *   q_min            : Lower limit on the power-law index of the density
 *                      reconstruction used for fragmentation.
 *   remnant_tail_eps : Accuracy parameter for the remnant tails.
 */
template<class Kernel, class Fragments>
class CoagulationRateMHL {
  public:

    CoagulationRateMHL(SizeGrid& grains, Kernel kernel, Fragments fragments,
                       double q_min=-10, double remnant_tail_eps=1e-2)
     : _kernel(kernel), _grain_sizes(grains), _cache(grains.size()),
       _q_min(q_min), _tail_eps(remnant_tail_eps)
    {
        _set_coagulation_properties() ;
        _set_fragment_properties(fragments) ;
    }

    void set_kernel(Kernel kernel) {
        _kernel = kernel ;
    }

    void set_fragments(Fragments fragments) {
        _set_fragment_properties(fragments) ;
    }

    void set_min_slope(double q_min) {
        _q_min = q_min ;
    }

    void set_remnant_tail_eps(double eps) {
        _tail_eps = eps ;
    }

    void operator()(const Field3D<double>& dust_density, Field3D<double>& rate, Field<bool>& active) const ;

  private:
    Kernel _kernel ;
    SizeGrid& _grain_sizes ;

    CoagulationCacheMHL _cache ;

    double _q_min, _tail_eps ;

    void _set_coagulation_properties() ;
    void _set_fragment_properties(Fragments) ;
} ;

template<class Kernel, class Fragments>
CoagulationRateMHL<Kernel, Fragments> create_coagulation_rate_MHL(
        SizeGrid& sizes, Kernel kernel, Fragments fragments,
        double q_min=-10, double remnant_tail_eps=1e-2) {
    return CoagulationRateMHL<Kernel, Fragments>(
        sizes, std::move(kernel), std::move(fragments), q_min, remnant_tail_eps) ;
}


#endif//_CUDISC_HEADERS_COAGULATION_COAGULATION_H_