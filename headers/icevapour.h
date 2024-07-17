#ifndef _CUDISC_ICEVAPOUR_H
#define _CUDISC_ICEVAPOUR_H

#include "field.h"
#include "coagulation/size_grid.h"

struct Prims;

struct IceVap {
    double vap, ice, tot;
};

class Molecule {

    public:

        double m_mol;
        double T_bind;

        Molecule() {};

        Molecule(Grid& g, double m_mol, double T_bind) : m_mol(m_mol), T_bind(T_bind) {
            _rho = make_CudaArray<IceVap>((g.NR+2*g.Nghost)*(g.Nphi+2*g.Nghost));
            stride = g.Nphi+2*g.Nghost;
            for (int i=0; i<(g.NR+2*g.Nghost)*(g.Nphi+2*g.Nghost); i++) {
                _rho[i].ice = _rho[i].tot = _rho[i].vap = 0.;
            }
        }

        IceVap& rho(int i, int j) {
            return _rho[i*stride + j];
        }

        void set_T_bind(double T) {
            T_bind = T;
        }

    private:

        CudaArray<IceVap> _rho; 
        int stride;

        friend class MoleculeRef;
};

class MoleculeRef {

    public:

        double m_mol;
        double T_bind;

        MoleculeRef(Molecule& _mol) : 
            m_mol(_mol.m_mol), T_bind(_mol.T_bind), stride(_mol.stride), _rho(_mol._rho.get()) 
            {}

        __host__ __device__
        IceVap& rho(int i, int j) {
            return _rho[i*stride + j];
        }

    private:
        
        int stride;
        IceVap* _rho; 
        
};

class IceVapChem {

    private:

        GridRef _g;
        FieldConstRef<double> _T;
        Field3DRef<Prims> _W;
        FieldRef<Prims> _Wg;
        Field3DRef<double> _ice_grains;
        SizeGridIce& _sizes;
        MoleculeRef _mol;
        double _floor;


    public:

        IceVapChem(const Grid& g, const Field<double>& T, Field3D<Prims>& W_dust, Field<Prims>& W_gas, Field3D<double>& tracers, SizeGridIce& sizes, 
                        Molecule& mol, double floor = 1.e-100, double N_s = 1.5e15) :
                        _g(g), _T(T), _W(W_dust),  _Wg(W_gas), _ice_grains(tracers), _sizes(sizes), _mol(mol), _floor(floor), N_s(N_s)
                        {} ; 

        void imp_update(double dt);
        void change_molecule(Molecule& mol) {
            _mol = mol;
        }
        
        template<typename out_type>
        void write_mol(std::filesystem::path dir, out_type out) {

            std::stringstream out_string ;
            out_string << out ;
            
            std::ofstream f(dir / ("mol_" + out_string.str() + ".dat"), std::ios::binary);
            
            int NR = _g.NR+2*_g.Nghost, NZ = _g.Nphi+2*_g.Nghost, nspec = _W.Nd;

            f.write((char*) &NR, sizeof(int));
            f.write((char*) &NZ, sizeof(int));
            f.write((char*) &nspec, sizeof(int));
            for (int i=0; i<_g.NR+2*_g.Nghost; i++) {
                for (int j=0; j<_g.Nphi+2*_g.Nghost; j++) {
                    
                    f.write((char*) &_mol.rho(i,j).vap, sizeof(double));
                    for (int k=0; k<nspec; k++) {
                        f.write((char*) &_ice_grains(i,j,k), sizeof(double));
                        f.write((char*) &_sizes.ice(i,j,k).a, sizeof(double));
                        f.write((char*) &_sizes.ice(i,j,k).rho, sizeof(double));
                    }

                }
            }  
            f.close();
        }

        template<typename out_type>
        void read_mol(std::filesystem::path dir, out_type out) {

            std::stringstream out_string ;
            out_string << out ;
            
            std::ifstream f(dir / ("mol_" + out_string.str() + ".dat"), std::ios::binary);

            int NR = _g.NR+2*_g.Nghost, NZ = _g.Nphi+2*_g.Nghost, nspec = _W.Nd;

            f.read((char*) &NR, sizeof(int));
            f.read((char*) &NZ, sizeof(int));
            f.read((char*) &nspec, sizeof(int));

            for (int i=0; i<_g.NR+2*_g.Nghost; i++) {
                for (int j=0; j<_g.Nphi+2*_g.Nghost; j++) {
                    
                    f.read((char*) &_mol.rho(i,j).vap, sizeof(double));
                    double ice_tot = 0;
                    for (int k=0; k<nspec; k++) {
                        f.read((char*) &_ice_grains(i,j,k), sizeof(double));
                        f.read((char*) &_sizes.ice(i,j,k).a, sizeof(double));
                        f.read((char*) &_sizes.ice(i,j,k).rho, sizeof(double));
                        ice_tot += _ice_grains(i,j,k);
                    }
                    _mol.rho(i,j).ice = ice_tot;
                    _mol.rho(i,j).tot = _mol.rho(i,j).ice + _mol.rho(i,j).vap;

                }
            }  
            f.close();
        }

        double N_s;

};




#endif// _CUDISC_ICEVAPOUR_H