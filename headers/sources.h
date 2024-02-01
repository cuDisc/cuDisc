#ifndef _CUDISC_SOURCES_H_
#define _CUDISC_SOURCES_H_

#include "grid.h"
#include "field.h"
#include "dustdynamics.h"
#include "coagulation/size_grid.h"

void testTpcu(const Field3D<double> &t_stop, const Field3D<Quants> &q, 
                        double dt, int l, int m) ;

void source_term_updateTL(Grid& g, Field3D<Prims>& q, const Field<Prims>& w_gas, const Field<double>& T, 
                        const CudaArray<double>& s, double dt, double Mstar, double rho_m, double mu, int bound, double floor=1e-40) ;
                        
void source_term_update(Grid& g, Field3D<Prims>& q, const Field<Prims>& w_gas, const Field<double>& T, 
                        const SizeGrid& s, double dt, double Mstar, double rho_m, double mu, int bound, double floor) ;

void source_term_update(Grid& g, Field3D<Prims>& q, const Field<Prims>& w_gas, const Field<double>& T, const Field3D<double>& f_rad, 
                        const SizeGrid& s, double dt, double Mstar, double rho_m, double mu, int bound, double floor) ;


class SourcesBase {

    public:

        virtual void source_exp(Grid& g, Field3D<Prims>& w, Field3D<Quants>& u, double dt) = 0;

        virtual void source_imp(Grid& g, Field3D<Prims>& w, double dt) = 0;


} ;

class NoSources : public SourcesBase {

    public:

        void source_exp(Grid&, Field3D<Prims>&, Field3D<Quants>&, double) {};
        void source_imp(Grid&, Field3D<Prims>&, double) {};

} ;

class Sources : public SourcesBase {

    public:

        Sources(const Field<double>& T, const Field<Prims>& w_gas, const SizeGrid& s, double floor, double Mstar=1., double mu=2.4) :
            _Mstar(Mstar), _mu(mu), _sizes(s), _T(T), _w_gas(w_gas), _floor(floor) {};

        void source_exp(Grid& g, Field3D<Prims>& w, Field3D<Quants>& u, double dt);
        void source_imp(Grid& g, Field3D<Prims>& w, double dt);

    private:

        double _Mstar;
        double _mu; 
        const SizeGrid& _sizes;
        FieldConstRef<double> _T;
        FieldConstRef<Prims> _w_gas;
        double _floor;
} ;
class SourcesRad : public SourcesBase {

    public:

        SourcesRad(const Field<double>& T, const Field<Prims>& w_gas, const Field3D<double>& f_rad, const SizeGrid& s, double floor, double Mstar=1., double mu=2.4) :
            _Mstar(Mstar), _mu(mu), _sizes(s), _T(T), _w_gas(w_gas), _f_rad(f_rad), _floor(floor){};

        void source_exp(Grid& g, Field3D<Prims>& w, Field3D<Quants>& u, double dt);
        void source_imp(Grid& g, Field3D<Prims>& w, double dt);

    private:

        double _Mstar;
        double _mu; 
        const SizeGrid& _sizes;
        FieldConstRef<double> _T;
        FieldConstRef<Prims> _w_gas;
        Field3DConstRef<double> _f_rad;
        double _floor;

} ;

#endif