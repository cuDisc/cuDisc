
#ifndef _CUDISC_DUSTDYNAMICS_H_
#define _CUDISC_DUSTDYNAMICS_H_

#include "cuda_array.h"
#include "field.h"
#include "flags.h"
#include "grid.h"
#include "utils.h"
#include "icevapour.h"
#include <memory>

class SourcesBase ; 

struct Quants {
    double rho, mom_R, amom_phi, mom_Z;

    inline __host__ __device__ double& operator[](int i) {
        if (i==0) { return rho; } 
        if (i==1) { return mom_R; } 
        if (i==2) { return amom_phi; } 
        return mom_Z; 
    } ;
} ;

struct Prims {
    double rho, v_R, v_phi, v_Z;

    inline __host__ __device__ double& operator[](int i) {
        if (i==0) { return rho; } 
        if (i==1) { return v_R; } 
        if (i==2) { return v_phi; } 
        return v_Z; 
    } ;

    inline __host__ __device__ const double& operator[](int i) const {
        if (i==0) { return rho; }
        if (i==1) { return v_R; }
        if (i==2) { return v_phi; }
        return v_Z;
    } ;

} ;

__global__
void _set_boundaries(GridRef g, Field3DRef<Prims> w, int bound) ;

class DustDynamics {

    public:

        DustDynamics(Field3D<double>& D, const Field<double>& cs, SourcesBase& sources, double CFL_adv=0.4, double CFL_diff=0.1, double floor=1.e-40, double gas_floor=1.e-30) : 
               _DoDiffusion(true), _CFL_adv(CFL_adv), _CFL_diff(CFL_diff), _floor(floor), _gas_floor(gas_floor), _D(D), _cs(cs), _sources(sources)
            {};
        
        void disable_diffusion() {
            _DoDiffusion = false ;
        } ;

        void set_diffusion_parameter(Field3D<double>& D) {
            _DoDiffusion = true;
            _D = D;
        }

        void set_CFL_adv(double cfl) {
            _CFL_adv = cfl;
        }

        void set_CFL_diff(double cfl) {
            _CFL_diff = cfl;
        }

        void set_boundaries(int flag) {
            _boundary = flag ;
        }
        int get_boundaries() const {
        return _boundary ;
        }

        void compute_gas_floor_height(Grid& g, Field<Prims>& w_gas, CudaArray<double>& h) const;

        void floor_above(Grid&g, Field3D<Prims>& w_dust, Field<Prims>& w_gas, CudaArray<double>& h) const;

        void reinitialize_active(Grid& g, const Field3D<Prims>& w_dust,
                     const Field<Prims>& w_gas);
        void enforce_floor_for_inactive(Grid& g, Field3D<Prims>& w_dust, const Field<Prims>& w_gas) const;

        void operator() (Grid& g, Field3D<Prims>& w_dust, const Field<Prims>& w_gas, double dt) ;
        void operator() (Grid& g, Field3D<Prims>& w_dust, const Field<Prims>& w_gas, double dt, Molecule& mol) ;
        void operator() (Grid& g, Field3D<Prims>& w_dust, const Field<Prims>& w_gas, double dt, Molecule& mol, SizeGridIce& sizes) ;

        double get_CFL_limit(const Grid& g, const Field3D<Prims>& w, const Field<Prims>& w_gas) ;
        double get_CFL_limit(const Grid& g, const Field3D<Prims>& w, const Field<Prims>& w_gas, Molecule& mol) ;
        double get_CFL_limit_debug(const Grid& g, const Field3D<Prims>& w, const Field<Prims>& w_gas);
        // double get_CFL_limit_debug(const Grid& g, const Field3D<Quants>& q, const Field3D<double>& D) ;

    private:
        // Donor-cell stage: sets boundaries on w, computes conserved quantities q,
        // computes donor-cell fluxes, applies boundary/inactive flux masking.
        void donor_cell_update(
            Grid& g, Field3D<Prims>& w, const Field<Prims>& w_gas,
            Field3D<Quants>& q, Field3D<Quants>& fluxR, Field3D<Quants>& fluxZ,
            Field3D<int>& active, dim3 blocks, dim3 threads) ;
    
        // Van Leer stage: sets boundaries on w, computes Van Leer fluxes,
        // applies boundary/inactive flux masking. Does NOT call _update_quants.
        void van_leer_update(
            Grid& g, Field3D<Prims>& w, const Field<Prims>& w_gas,
            Field3D<Quants>& fluxR, Field3D<Quants>& fluxZ,
            Field3D<int>& active, dim3 blocks, dim3 threads) ;

        // Quants and source update
        template<bool apply_sources=true>
        void update_quants_and_sources(Grid& g, Field3D<Prims>& w, Field3D<Quants>& q_mids, Field3D<Quants>& q, const Field<Prims>& w_gas,
            double dt, Field3D<Quants>& fluxR, Field3D<Quants>& fluxZ,
            Field3D<int>& active, dim3 blocks, dim3 threads) ;


        bool _DoDiffusion = true ;
        double _CFL_adv;
        double _CFL_diff;
        double _floor;
        double _gas_floor;
        Field3DRef<double> _D;
        FieldConstRef<double> _cs;
        SourcesBase& _sources;
        mutable std::unique_ptr<Field3D<int>> _active;
        mutable std::unique_ptr<Field3D<int>> _active_vap;

        int _boundary = BoundaryFlags::open_R_inner | BoundaryFlags::open_R_outer;

} ;


#endif
