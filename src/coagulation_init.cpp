
#include <assert.h>
#include <algorithm>
#include <limits>
#include <vector>

#include "coagulation/coagulation.h"
#include "coagulation/fragments.h"

#include "timing.h"

template<class Kernel, class Fragments>
void CoagulationRate<Kernel,Fragments>::_set_coagulation_properties() {
    CodeTiming::BlockTimer block =
        timer->StartNewTimer("CoagulationRate::_set_coagulation_properties");

    int size = _grain_sizes.size();
    const RealType* _m = _grain_sizes.grain_masses();

    /* Set up the indices of the combined products */
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j) {
            double mass_tot = _m[i] + _m[j];

            // Brauer method needs largest k such that m_c(k) <= mass_tot.
            // First find m_e[k-1] < mass_tot < m_e[k]
            int k = _grain_sizes.grid_index(mass_tot);

            // Growth takes us off the grid
            if (k >= size) {
                _cache.index(i, j).coag = size;
                _cache.Cijk(i, j).coag = 0;
                continue;
            }

            // Now correct k to m_c[k-1] < mass_tot < m_c[k]
            if (_m[k - 1] > mass_tot) k = k - 1;

            // Store the lower index
            assert(k > 0 && k <= size);
            _cache.index(i, j).coag = k - 1;

            // Fraction of rate into lower bin
            double m_l, m_u;

            // if (k == size)
            //     m_u = 2 * _m[k - 1] - _m[k - 2];
            // else
            m_u = _m[k];

            if (k == 0)
                m_l = 2 * _m[0] - _m[1];
            else
                m_l = _m[k - 1];

            _cache.Cijk(i, j).coag = (m_u - mass_tot) / (m_u - m_l);

            assert(m_u >= mass_tot);
            assert(m_l <= mass_tot);
            assert(_cache.Cijk(i, j).coag >= 0 && _cache.Cijk(i, j).coag <= 1);
        }
}

template<class Kernel, class Fragments>
void CoagulationRate<Kernel,Fragments>::_set_fragment_properties(Fragments fragments) {
    CodeTiming::BlockTimer block =
        timer->StartNewTimer("CoagulationRate::_set_fragment_properties");

    /* Cache the coefficients */
    int size = _grain_sizes.size();
    const RealType* _m = _grain_sizes.grain_masses();

    /* Get the bin containing this mass */
    const SizeGrid& g = _grain_sizes ;
    auto index = [&g, size](RealType m) -> int {
        // return std::max(1, std::min(g.grid_index(m), size)) - 1;
        for (int i=1; i<size; i++){
            if (g.centre_mass(i) > m) 
                return i-1; 
        }
        return size-1;
    } ;

    /* Set up the indices of the combined products */
    for (int i = 0; i < size; ++i)
        for (int j = 0; j <  size; ++j) {
            /* Fragments */
            double mass_scale = fragments.mass_scale(_m[i], _m[j]);
            _cache.index(i, j).frag = index(mass_scale) ;

            /* Remnants */
            _cache.Cijk(i, j).remnant = fragments.remnant_mass(_m[i], _m[j]);
            _cache.index(i, j).remnant = index(_cache.Cijk(i, j).remnant) ;
            if (_cache.index(i, j).remnant < size-1) {
                _cache.Cijk(i,j).eps = std::min(1.,(_m[_cache.index(i, j).remnant + 1] - _cache.Cijk(i, j).remnant) / (_m[_cache.index(i, j).remnant + 1] - _m[_cache.index(i, j).remnant]));
            }
            else {_cache.Cijk(i,j).eps = 1.;}
            assert(_cache.index(i, j).remnant <= std::max(i, j));
        }

    /* Store the fragment product distributions */
    for (int i = 0; i < size; ++i) {
        double mm = _grain_sizes.edge_mass(0) ;
        double fm = fragments.cumulative_fragment_distribution(mm,_m[i]) ;
        for (int j = 0; j < size; ++j) {
            double mp = _grain_sizes.edge_mass(j + 1);
            double fp = fragments.cumulative_fragment_distribution(mp,_m[i]) ;

            _cache.Cijk_frag(i, j) = fp - fm ;

            mm = mp ;
            fm = fp ;

            assert(_cache.Cijk_frag(i, j) >= 0);
        }
    }
}

template class CoagulationRate<BirnstielKernel<false>,SimpleErosion> ;
template class CoagulationRate<BirnstielKernel<true>,SimpleErosion> ;

template class CoagulationRate<BirnstielKernelVertInt<false>,SimpleErosion> ;
template class CoagulationRate<BirnstielKernelVertInt<true>,SimpleErosion> ;

template class CoagulationRate<ConstantKernel,SimpleErosion> ;


/* CoagulationRateMHL setup (ported from coag_toolkit, MHL_coag.cpp and
 * MHL_frag.cpp)
 */

CoagulationCacheMHL::CoagulationCacheMHL(int size)
 : m(make_CudaArray<double>(size)), me(make_CudaArray<double>(size+1)),
   dm(make_CudaArray<double>(size)), dlgm(make_CudaArray<double>(size)),
   lgx(make_CudaArray<double>(size)), xm1(make_CudaArray<double>(size)),
   dlg_lo(make_CudaArray<double>(size)), dlg_up(make_CudaArray<double>(size)),
   d2(make_CudaArray<double>(size)), d3(make_CudaArray<double>(size)),
   E2(make_CudaArray<double>(size+1)), E3(make_CudaArray<double>(size+1)),
   Cfrag(make_CudaArray<double>(size*size)),
   _pairs(make_CudaArray<Pair>(size*size)),
   _size(size)
{ }

template<class T>
static void _upload_table(const std::vector<T>& v, CudaArray<T>& d) {
    d = make_CudaArray<T>(std::max<std::size_t>(v.size(), 1)) ;
    std::copy(v.begin(), v.end(), d.get()) ;
}

void CoagulationCacheMHL::upload() {
    _upload_table(nodes, d_nodes) ;
    _upload_table(runs, d_runs) ;
    _upload_table(pieces, d_pieces) ;
    _upload_table(segments, d_segments) ;
}

/* Index i such that me[i-1] < mass <= me[i], c.f. SizeGrid::grid_index, but
 * using the double precision edges.
 */
static int _grid_index(const CoagulationCacheMHL& c, double mass) {
    const double* me = c.me.get() ;
    return std::lower_bound(me, me + c.size() + 1, mass) - me ;
}

template<class Kernel, class Fragments>
void CoagulationRateMHL<Kernel,Fragments>::_set_coagulation_properties() {
    CodeTiming::BlockTimer block =
        timer->StartNewTimer("CoagulationRateMHL::_set_coagulation_properties");

    int size = _grain_sizes.size() ;
    CoagulationCacheMHL& c = _cache ;

    /* Save the mass centres/edges */
    double *_m = c.m.get(), *_me = c.me.get() ;
    _me[0] = _grain_sizes.edge_mass(0) ;
    for (int i = 0; i < size; ++i) {
        _m [i] = _grain_sizes.centre_mass(i) ;
        _me[i+1] = _grain_sizes.edge_mass(i+1) ;
        c.dm[i] = _me[i+1] - _me[i] ;
        c.lgx[i] = std::log(_me[i+1] / _me[i]) ;
        c.xm1[i] = c.dm[i] / _me[i] ;
        c.dlg_lo[i] = std::log(_m[i] / _me[i]) ;
        c.dlg_up[i] = std::log(_me[i+1] / _m[i]) ;
    }
    for (int i = 0; i < size; ++i) {
        if (i == 0)
            c.dlgm[i] = std::log(_m[1] / _m[0]) ;
        else if (i + 1 == size)
            c.dlgm[i] = std::log(_m[size-1]/_m[size-2]) ;
        else
            c.dlgm[i] = std::log(_m[i+1]/_m[i-1]) ;
    }

    /* Shapes of the remnant tails. The tail below me[K] is distributed as
     * dM ~ r(1 + beta r) dr over 0 < r < me[K], i.e. bin k < K receives the
     * fractions
     *    d2[k]/E2[K],  d3[k]/E3[K]
     * of the flat and linear parts, where d2[k] = x_{k+1}^2 - x_k^2, etc.,
     * x_k = me[k]/me[size] and x_0 -> 0.
     *
     * The differences are factored to avoid cancellation, and E2, E3 are the
     * sums of the stored values (accumulated in long double and rounded once)
     * so that each shape sums to one to within a few ulps.
     */
    {
        long double S = _me[size] ;
        long double E2 = 0, E3 = 0 ;
        c.E2[0] = c.E3[0] = 0 ;
        for (int k = 0; k < size; ++k) {
            long double xl = (k == 0) ? 0 : _me[k] / S ;
            long double xu = _me[k+1] / S ;
            long double dx = ((long double)_me[k+1] - ((k == 0) ? 0 : _me[k])) / S ;

            c.d2[k] = dx * (xu + xl) ;
            c.d3[k] = dx * (xu*xu + xu*xl + xl*xl) ;

            E2 += c.d2[k] ;
            E3 += c.d3[k] ;
            c.E2[k+1] = E2 ;
            c.E3[k+1] = E3 ;
        }
    }

    /* Store indices of lower and upper bins such that the coagulation product
     * falls in bins, k, in the range
     *   kmin <= k < kmax,
     * i.e.
     *    _me[kmin] <= (_m[i] + _me[j]) and (_m[i] + _me[j+1]) < _me[kmax]
     *
     * and the boundaries of the mass ranges in bin j that contribute to each
     * bin, k, for collisions with bin i (i <= j). Range k is
     *   [max(mmin, _me[k]), min(mmax, _me[k+1])] - _m[i],
     * where mmin/mmax are the minimum/maximum masses of the coagulation
     * products, so neighbouring ranges share a boundary.
     */
    c.nodes.clear() ;
    for (int j = 0; j < size; ++j)
        for (int i = 0; i <= j; ++i) {
            CoagulationCacheMHL::Pair& p = c.pair(i,j) ;

            double mass_l = _m[i] + _me[j] ;
            double mass_u = _m[i] + _me[j+1];

            int k = _grid_index(c, mass_l) ;
            assert(k != 0) ;
            k-- ;
            p.kmin = k ;
            while (k < size + 1 and _me[k] < mass_u) k++ ;
            p.kmax = k ;

            assert(p.kmin < p.kmax) ;

            p.node_start = c.nodes.size() ;

            double mmax = _m[i] + _me[j+1] ;
            for (int kk = p.kmin; kk <= p.kmax; ++kk) {
                double mk ;
                if (kk == p.kmin)
                    mk = _me[j] ;  // max(_m[i] + _me[j], _me[kmin]) - _m[i], exactly
                else if (kk - 1 >= size)
                    mk = mmax - _m[i] ;
                else
                    mk = std::min(mmax, _me[kk]) - _m[i] ;

                c.nodes.push_back({std::log(mk / _me[j]), (mk - _me[j]) / _me[j]}) ;
            }
        }
}

/* Helper for building the fragmentation tables */
class _MHLFragmentBuilder {
  public:
    using Node = CoagulationCacheMHL::Node ;

    _MHLFragmentBuilder(CoagulationCacheMHL& c)
      : _c(c), _me(c.me.get()), _size(c.size())
    { } ;

    Node node(int t, double m) const {
        return {std::log(m / _me[t]), (m - _me[t]) / _me[t]} ;
    }

    int scale_bin(double m_scale) const {
        return std::max(1, std::min(_grid_index(_c, m_scale), _size)) - 1 ;
    }

    /* add_remnant_run
     *
     * Split the (affine) remnant mass for target masses in [m_a, m_b] (inside
     * bin t), with remnant masses r_a and r_b at the ends, into pieces that
     * each fall within a single destination bin.
     */
    void add_remnant_run(int t, double m_a, double m_b, double r_a, double r_b) ;

    /* add_fragment_segment
     *
     * Close the segment of bin t that ends at m_b, with fragment scale bin s,
     * after its remnant runs have been added (merging it with the previous
     * segment of the pair if that has the same scale bin).
     */
    void add_fragment_segment(int t, std::size_t first_segment, double m_b, int s) {
        CoagulationCacheMHL::FragSegment seg =
            {s, node(t, m_b), int(_c.runs.size())} ;
        if (_c.segments.size() > first_segment && _c.segments.back().s == s)
            _c.segments.back() = seg ;   // extend the previous segment
        else
            _c.segments.push_back(seg) ;
    }

  private:
    CoagulationCacheMHL& _c ;
    const double* _me ;
    int _size ;
} ;

void _MHLFragmentBuilder::add_remnant_run(int t, double m_a, double m_b,
                                          double r_a, double r_b) {
    // Remnants are non-negative, so clip at zero
    double r_min = std::max(std::min(r_a, r_b), 0.) ;
    double r_max = std::max(r_a, r_b) ;
    if (r_max <= 0) return ;

    // m_rem = R0 + R1 * m
    double R1 = (r_b - r_a) / (m_b - m_a) ;
    double R0 = r_a - R1 * m_a ;

    // Inverse, exact at the ends of the segment
    auto m_of_r = [&](double r) {
        if (r == r_a) return m_a ;
        if (r == r_b) return m_b ;
        return (r - R0) / R1 ;
    } ;

    /* Destination bin k collects remnants with masses in [_me[k], _me[k+1]),
     * except that bin 0 extends down to zero mass and the last bin extends to
     * infinity.
     *
     * Pieces are generated in order of increasing remnant mass, each with the
     * range of target masses, [m_lo, m_hi], that maps onto it.
     */
    struct Piece { int k ; double m_lo, m_hi ; } ;
    std::vector<Piece> pcs ;

    if (R1 == 0) {
        // Constant remnant: one piece covering the whole segment
        int k = std::max(std::min(_grid_index(_c, r_max), _size), 1) - 1 ;
        pcs.push_back({k, m_a, m_b}) ;
    }
    else {
        int k = std::min(_grid_index(_c, r_min), _size) ;
        if (k > 0) k-- ;

        for (; k < _size; ++k) {
            double r_l = (k == 0)         ? 0 : _me[k] ;
            double r_u = (k + 1 == _size) ? std::numeric_limits<double>::max()
                                          : _me[k+1] ;

            double lo = std::max(r_l, r_min) ;
            double hi = std::min(r_u, r_max) ;

            if (hi > lo) {
                // Range of target masses that map onto bin k
                double m_lo = m_of_r(lo) ;
                double m_hi = m_of_r(hi) ;
                if (R1 < 0) std::swap(m_lo, m_hi) ;

                m_lo = std::max(m_lo, m_a) ;
                m_hi = std::min(m_hi, m_b) ;

                if (m_hi > m_lo)
                    pcs.push_back({k, m_lo, m_hi}) ;
            }

            if (r_u >= r_max) break ;
        }
    }
    if (pcs.empty()) return ;

    /* Store the run in walk order, from the largest remnant down. If R1 > 0
     * the remnant increases with the target mass and the walk is towards
     * lower target masses.
     */
    CoagulationCacheMHL::RemnantRun run ;
    run.R0 = R0 ;
    run.R1me = R1 * _me[t] ;
    run.dir = (R1 > 0) ? -1 : 1 ;
    run.piece_start = _c.pieces.size() ;
    for (auto p = pcs.rbegin(); p != pcs.rend(); ++p)
        _c.pieces.push_back({p->k, node(t, (R1 > 0) ? p->m_lo : p->m_hi)}) ;
    run.piece_end = _c.pieces.size() ;

    double m_start = (R1 > 0) ? pcs.back().m_hi : pcs.back().m_lo ;
    double m_end   = (R1 > 0) ? pcs.front().m_lo : pcs.front().m_hi ;
    run.start = node(t, m_start) ;
    run.end = node(t, m_end) ;

    /* The remnant falls to zero (or to round-off) at the end of the walk, so
     * the pieces below a cut can be replaced by an analytic tail.
     */
    run.tail = (R1 != 0) && (std::min(r_a, r_b) <= 0 || r_min < 1e-12 * r_max) ;
    run.inv_R1m0 = run.tail ? 1 / (R1 * m_end) : 0 ;

    _c.runs.push_back(run) ;
}

template<class Kernel, class Fragments>
void CoagulationRateMHL<Kernel,Fragments>::_set_fragment_properties(Fragments f) {
    CodeTiming::BlockTimer block =
        timer->StartNewTimer("CoagulationRateMHL::_set_fragment_properties");

    int size = _grain_sizes.size() ;
    CoagulationCacheMHL& c = _cache ;
    const double *_m = c.m.get(), *_me = c.me.get() ;

    _MHLFragmentBuilder builder(c) ;

    /* Store the (normalized) fragment product distributions, evaluated at
     * bin-centre masses */
    for (int s = 0; s < size; ++s) {
        int id_max = builder.scale_bin(f.max_fragment_mass(_m[s])) ;
        double Ctot = 0 ;
        for (int k = 0; k < size; ++k) {
            double C = 0 ;
            if (k <= id_max)
                C = f.cumulative_fragment_distribution(_me[k+1], _m[s]) -
                    f.cumulative_fragment_distribution(_me[k], _m[s]) ;
            assert(C >= 0) ;
            c.Cfrag[s*size + k] = C ;
            Ctot += C ;
        }
        for (int k = 0; k <= id_max; ++k)
            c.Cfrag[s*size + k] /= Ctot ;
    }

    /* Remnants and fragment segments for each pair of bins: target t >=
     * projectile p.
     *
     * The remnant mass and the fragment scale are affine in the target mass
     * between the breakpoints of the distribution, so each is represented
     * exactly by a line through two points inside the segment. Evaluating
     * inside the segment, rather than at its ends, picks up the correct side of
     * any jump at a breakpoint.
     */
    c.runs.clear() ;
    c.pieces.clear() ;
    c.segments.clear() ;

    const double eps = 1e-12 ;
    std::vector<double> bp, nodes, cuts ;
    for (int t = 0; t < size; ++t)
        for (int p = 0; p <= t; ++p) {
            CoagulationCacheMHL::Pair& pair = c.pair(p, t) ;
            pair.seg_start = c.segments.size() ;
            pair.run_start = c.runs.size() ;

            double m_lo = _me[t], m_hi = _me[t+1] ;
            double m_p = _m[p] ;

            f.breakpoints(m_p, bp) ;
            nodes.assign({m_lo, m_hi}) ;
            for (double b : bp)
                if (b > m_lo * (1 + eps) && b < m_hi * (1 - eps))
                    nodes.push_back(b) ;
            std::sort(nodes.begin(), nodes.end()) ;
            nodes.erase(std::unique(nodes.begin(), nodes.end(),
                                    [eps](double a, double b) { return b <= a * (1 + eps) ; }),
                        nodes.end()) ;

            std::size_t first_segment = c.segments.size() ;
            for (std::size_t n = 1; n < nodes.size(); ++n) {
                double m_a = nodes[n-1], m_b = nodes[n] ;

                // Lines through the points at 1/4 and 3/4 of the segment
                double m_1 = m_a + 0.25 * (m_b - m_a), m_2 = m_b - 0.25 * (m_b - m_a) ;
                double r_1 = f.remnant_mass(m_1, m_p), r_2 = f.remnant_mass(m_2, m_p) ;
                double s_1 = f.mass_scale(m_1, m_p),   s_2 = f.mass_scale(m_2, m_p) ;
                double R1 = (r_2 - r_1) / (m_2 - m_1), S1 = (s_2 - s_1) / (m_2 - m_1) ;
                auto remnant = [&](double m) { return r_1 + R1 * (m - m_1) ; } ;
                auto scale   = [&](double m) { return s_1 + S1 * (m - m_1) ; } ;

                /* Split further where the fragment scale crosses a bin edge, so
                 * that each piece has a single scale bin.
                 */
                cuts.assign({m_a, m_b}) ;
                if (S1 != 0) {
                    double lo = std::min(scale(m_a), scale(m_b)) ;
                    double hi = std::max(scale(m_a), scale(m_b)) ;
                    for (int k = _grid_index(c, lo); k < size + 1 && _me[k] < hi; ++k) {
                        double cut = m_1 + (_me[k] - s_1) / S1 ;
                        if (_me[k] > lo && cut > m_a && cut < m_b)
                            cuts.push_back(cut) ;
                    }
                    std::sort(cuts.begin(), cuts.end()) ;
                }

                for (std::size_t n = 1; n < cuts.size(); ++n) {
                    double c_a = cuts[n-1], c_b = cuts[n] ;
                    if (c_b <= c_a) continue ;
                    builder.add_remnant_run(t, c_a, c_b, remnant(c_a), remnant(c_b)) ;
                    builder.add_fragment_segment(t, first_segment, c_b,
                                                 builder.scale_bin(scale(0.5 * (c_a + c_b)))) ;
                }
            }
            pair.seg_end = c.segments.size() ;
        }

    c.upload() ;
}

template class CoagulationRateMHL<BirnstielKernel<false>,SimpleErosion> ;
template class CoagulationRateMHL<BirnstielKernel<true>,SimpleErosion> ;

template class CoagulationRateMHL<BirnstielKernelVertInt<false>,SimpleErosion> ;
template class CoagulationRateMHL<BirnstielKernelVertInt<true>,SimpleErosion> ;

template class CoagulationRateMHL<ConstantKernel,SimpleErosion> ;
