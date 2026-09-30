
#ifndef _CUDISC_HEADERS_COAGULATION_SIZE_GRID_H_
#define _CUDISC_HEADERS_COAGULATION_SIZE_GRID_H_

#include <algorithm>
#include <cmath>
#include <fstream>

#include "field.h"
#include "grid.h"
#include "cuda_array.h"
#include "density_view.h"

struct Prims;
struct Prims1D;
struct Quants;

#ifdef REAL_TYPE
using RealType = REAL_TYPE ;
#else
using RealType = float ;
#endif

// Per-bin grain properties stored on the grid (size + density)
struct Grain {
    double a, rho;
};


class SizeGrid
{


protected:

    Grid& _g;
    int stride;
    
public:

    SizeGrid(Grid& g, RealType a_min, RealType a_max, int Nbins, RealType rho_daux=1)
      : _g(g),
        stride(Nbins),
        _mass_e(make_CudaArray<RealType>(Nbins+1)),
        _mass_c(make_CudaArray<RealType>(Nbins)),
        rho_d(rho_daux),
        num_bins(Nbins)
    {
        RealType l_min = std::log(a_min) ;
        RealType l_max = std::log(a_max) ;
        RealType dl = (l_max - l_min) / Nbins ;
        
        _mass_e[0] = 4*M_PI*rho_d/3 * a_min*a_min*a_min ;
        for (int idx = 0; idx < Nbins; ++idx) {
            _mass_e[idx+1] = 4*M_PI*rho_d/3 * std::exp(3*(l_min + (idx + 1)*dl)) ;
            _mass_c[idx] = 0.5 * (_mass_e[idx] + _mass_e[idx+1]) ;
        }

        init_grain_field() ;
    }

    SizeGrid(Grid& g, CudaArray<RealType>& a, int Nbins, RealType rho_daux=1)
      : _g(g),
        stride(Nbins),
        _mass_e(make_CudaArray<RealType>(Nbins+1)),
        _mass_c(make_CudaArray<RealType>(Nbins)),
        rho_d(rho_daux),
        num_bins(Nbins)
    {
        for (int idx = 0; idx < Nbins; ++idx) {
            _mass_c[idx] = 4*M_PI*rho_d/3 * std::pow(a[idx], 3.) ;
        }
        for (int idx = 0; idx < Nbins-1; ++idx) {
            _mass_e[idx+1] = 0.5 * (_mass_c[idx] + _mass_c[idx+1]) ;
        }
        _mass_e[0] = std::max(_mass_c[0] - (_mass_e[1] - _mass_c[0]), (RealType)0.);
        _mass_e[Nbins] = _mass_c[Nbins-1] + (_mass_c[Nbins-1] - _mass_e[Nbins-1]);

        init_grain_field() ;
    }

    virtual ~SizeGrid() = default ;

    int size() const {
        return num_bins ;
    }

    RealType min_mass() const { 
        return _mass_e[0] ;
    }

    RealType max_mass() const { 
        return _mass_e[num_bins] ;
    }

    RealType centre_mass(int idx) const {
        return _mass_c[idx] ;
    }
  
    RealType edge_mass(int idx) const {
        return _mass_e[idx] ;
    }

    // Provide access to arrays for convenience
    const RealType* grain_masses() const {
        return _mass_c.get() ;
    }

    RealType solid_density() const {
        return rho_d ;
    }
  
    /* grid_index
     *
     * Find i, such that m_{i-1} < mass < mass_i
     */
    int grid_index(RealType mass) const {
        return std::distance(_mass_e.get(),
                             std::lower_bound(_mass_e.get(), 
                                                _mass_e.get()+num_bins+1,
                                                mass)
                               ) ;
    }

    void write_ASCII(std::string filename) {
        std::ofstream f(filename) ;
        f << "# Cells=" << size() << "\n" ;
        f << "# mass size\n" ;
        for (int i=0; i < size()+1; i++) 
            f << edge_mass(i) << " " 
              << std::pow(3*edge_mass(i)/(4*M_PI*rho_d), 1/3.) << "\n" ;
    }

    void write_grid(std::string folder) {
        std::ofstream f(folder+"/grains.sizes") ;
        f << "# Cells=" << size() << "\n" ;
        f << "# mass size\n" ;
        for (int i=0; i < size()+1; i++) 
            f << edge_mass(i) << " " 
              << std::pow(3*edge_mass(i)/(4*M_PI*rho_d), 1/3.) << "\n" ;
    }

    // Per-cell grain properties field (size + density), shared by SizeGrid and its subclasses
    Field3D<Grain> grain_props = create_field3D<Grain>(_g, stride);

    template<typename out_type>
    void write_grain_props(std::filesystem::path dir, out_type out) {

        std::stringstream out_string ;
        out_string << out ;
        
        std::ofstream f(dir / ("grain_props_" + out_string.str() + ".dat"), std::ios::binary);
        
        int NR = _g.NR+2*_g.Nghost, NZ = _g.Nphi+2*_g.Nghost;

        f.write((char*) &NR, sizeof(int));
        f.write((char*) &NZ, sizeof(int));
        f.write((char*) &num_bins, sizeof(int));
        for (int i=0; i<_g.NR+2*_g.Nghost; i++) {
            for (int j=0; j<_g.Nphi+2*_g.Nghost; j++) {
                for (int k=0; k<num_bins; k++) {
                    f.write((char*) &grain_props(i,j,k).a, sizeof(double));
                    f.write((char*) &grain_props(i,j,k).rho, sizeof(double));
                }

            }
        }  
        f.close();
    }

    // Recompute grain size/density from the current dust/ice densities.
    // No-op by default: a plain SizeGrid has a fixed grain size/density set
    // at construction.
    //
    // Wd holds the dust (refractory) density and Wd_ice the ice density, each
    // as any element type supported by density_view. Wg (Prims or Prims1D)
    // is passed on so subclasses have access to t_stop (needed for
    // compaction/porosity).
    template<typename Tg, typename Td, typename Ti>
    void update_sizes(const Field<Tg>& Wg, Field3D<Td>& Wd, Field3D<Ti>& Wd_ice) {
        _update_sizes(Wg, density_view(Wd), density_view(Wd_ice)) ;
    }

protected:

    // Customisation points for update_sizes with ice, one per gas type.
    // No-op by default.
    virtual void _update_sizes(const Field<Prims>& /*Wg*/, DensityView /*rho_dust*/, DensityView /*rho_ice*/) {}
    virtual void _update_sizes(const Field<Prims1D>& /*Wg*/, DensityView /*rho_dust*/, DensityView /*rho_ice*/) {}

private:

    void init_grain_field() {
        for (int i=0; i<_g.NR+2*_g.Nghost; i++) {
            for (int j=0; j<_g.Nphi+2*_g.Nghost; j++) {
                for (int k=0; k<num_bins; k++) {
                    grain_props(i,j,k).a = std::pow(3./4./M_PI*_mass_c[k]/rho_d, 1./3.);
                    grain_props(i,j,k).rho = rho_d;
                }
            }
        }
    }

    CudaArray<RealType> _mass_e, _mass_c ;

    RealType rho_d=1;
    int num_bins;
    friend class SizeGridRef;
};

class SizeGridRef {

    protected:

        GridRef _g;
        int stride;
        RealType* _mass_c;

    public:

        SizeGridRef(SizeGrid& size) :
            _g(size._g),
            stride(size.stride),
            _mass_c(size._mass_c.get()),
            grain_props(size.grain_props)
        {}

        Field3DRef<Grain> grain_props;

        __host__ __device__ 
        RealType base_mass(int idx) const {
            return _mass_c[idx] ;
        }
} ;

#endif//_CUDISC_HEADERS_COAGULATION_SIZE_GRID_H_