#ifndef _CUDISC_DUSTDYNAMICS1D_H_
#define _CUDISC_DUSTDYNAMICS1D_H_

#include "field.h"
#include "flags.h"
#include "star.h"
#include "coagulation/size_grid.h"

struct Prims1D {
    double Sig, v_R;

    inline __host__ __device__ double& operator[](int i) {
        if (i==0) { return Sig; } 
        return v_R; 
    } ;
} ;


class DustDyn1D {

     public:

        DustDyn1D(Field3D<double>& D, const Field<double>& cs, Star& star, SizeGrid& sizes, CudaArray<double>& nu, double CFL_adv=0.4, double CFL_diff=0.1, double floor=1.e-5, double gas_floor=1.e-10) : 
              _D(D), _cs(cs), _star(star), _sizes(sizes), _nu(nu), _CFL_adv(CFL_adv), _CFL_diff(CFL_diff), _floor(floor), _gas_floor(gas_floor) {};

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

        void operator() (Grid& g, Field3D<Prims1D>& W_d, const Field<Prims1D>& W_g, double dt) ;

        double get_CFL_limit(const Grid& g, const Field3D<Prims1D>& W_d, const Field<Prims1D>& W_g) ;

    private:

        Field3DRef<double> _D;
        FieldConstRef<double> _cs;
        Star& _star;
        SizeGrid& _sizes;
        CudaArray<double>& _nu;
        double _CFL_adv;
        double _CFL_diff;
        double _floor;
        double _gas_floor;

        int _boundary = BoundaryFlags::open_R_inner | BoundaryFlags::open_R_outer;

} ;

#endif//_CUDISC_HEADERS_DUSTDYNAMICS1D_H_