
#include <assert.h>

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
