
#ifndef _CUDISC_HEADERS_COAGULATION_SIZE_GRID_H_
#define _CUDISC_HEADERS_COAGULATION_SIZE_GRID_H_

#include <algorithm>
#include <cmath>
#include <fstream>

#include "field.h"
#include "grid.h"
#include "cuda_array.h"

#ifdef REAL_TYPE
using RealType = REAL_TYPE ;
#else
using RealType = float ;
#endif

class SizeGrid
{
public:

    SizeGrid(RealType a_min, RealType a_max, int Nbins, RealType rho_daux=1)
      : _mass_e(make_CudaArray<RealType>(Nbins+1)),
        _mass_c(make_CudaArray<RealType>(Nbins)),
        _a_c(make_CudaArray<RealType>(Nbins)),
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
            _a_c[idx] = std::pow(3./4./M_PI*_mass_c[idx]/rho_d,1./3.);
        }
    }

    SizeGrid(CudaArray<RealType>& a, int Nbins, RealType rho_daux=1)
      : _mass_e(make_CudaArray<RealType>(Nbins+1)),
        _mass_c(make_CudaArray<RealType>(Nbins)),
        _a_c(make_CudaArray<RealType>(Nbins)),
        rho_d(rho_daux),
        num_bins(Nbins)
    {
        for (int idx = 0; idx < Nbins; ++idx) {
            _mass_c[idx] = 4*M_PI*rho_d/3 * std::pow(a[idx], 3.) ;
            _a_c[idx] = a[idx];
        }
        for (int idx = 0; idx < Nbins-1; ++idx) {
            _mass_e[idx+1] = 0.5 * (_mass_c[idx] + _mass_c[idx+1]) ;
        }
        _mass_e[0] = std::max(_mass_c[0] - (_mass_e[1] - _mass_c[0]), (RealType)0.);
        _mass_e[Nbins] = _mass_c[Nbins-1] + (_mass_c[Nbins-1] - _mass_e[Nbins-1]);
    }


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

    RealType centre_size(int idx) const {
        return _a_c[idx];
    }

    // Provide access to arrays for convenience
    const RealType* grain_sizes() const {
        return _a_c.get() ;
    }
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

  private:

    CudaArray<RealType> _mass_e, _mass_c, _a_c ;

    RealType rho_d=1;
    int num_bins;
    friend class SizeGridIce;
    friend class SizeGridIceRef;
};

struct Ice {
    double a, rho;
};

class SizeGridIce : public SizeGrid {

    private:

        Grid& _g;
        int stride;
        RealType _rho_daux, _rho_m_ice;

        RealType _a_min, _a_max;

        friend class SizeGridIceRef;

    public:

        SizeGridIce(Grid& g, RealType a_min, RealType a_max, int Nbins, RealType rho_daux, RealType rho_m_ice) : 
            SizeGrid(a_min, a_max, Nbins, rho_daux),
            _g(g),
            stride(Nbins), _rho_daux(rho_daux),
            _rho_m_ice(rho_m_ice), _a_min(a_min), _a_max(a_max)
        {
            for (int i=0; i<_g.NR+2*_g.Nghost; i++) {
                for (int j=0; j<_g.Nphi+2*_g.Nghost; j++) {
                    for (int k=0; k<Nbins; k++) {
                        ice(i,j,k).a = centre_size(k);
                        ice(i,j,k).rho = rho_daux;
                    }
                }
            }
        }

        Field3D<Ice> ice = create_field3D<Ice>(_g, stride);

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

        RealType centre_size(int idx) const {
            return _a_c[idx];
        }

        // Provide access to arrays for convenience
        const RealType* grain_sizes() const {
            return _a_c.get() ;
        }
        const RealType* grain_masses() const {
            return _mass_c.get() ;
        }

        RealType solid_density() const {
            return rho_d ;
        }

        RealType ice_density() const {
            return _rho_m_ice;
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

} ;

class SizeGridIceRef {

    private:

        GridRef _g;
        int stride;
        RealType _rho_m_ice;

    public:

        SizeGridIceRef(SizeGridIce& size) :
            _g(size._g),
            stride(size.stride),
            _rho_m_ice(size._rho_m_ice),
            ice(size.ice)
        {}

        Field3DRef<Ice> ice;

        __host__ __device__
        RealType ice_density() const {
            return _rho_m_ice;
        }
   
} ;

#endif//_CUDISC_HEADERS_COAGULATION_SIZE_GRID_H_
