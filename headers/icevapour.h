#ifndef _CUDISC_ICEVAPOUR_H
#define _CUDISC_ICEVAPOUR_H

#include "field.h"
#include "bins.h"
#include "coagulation/size_grid.h"

struct Prims;
struct Quants;
class Molecule {

    private:

        int _ndust;
        Grid& _g;
    
    public:

        static Molecule nullMol;

        double m_mol;
        double T_bind;

        Molecule(Grid& g, double m_mol, double T_bind, int ndust) : _ndust(ndust), _g(g), m_mol(m_mol), T_bind(T_bind) {
            set_all(_g, vap, 0.);
            set_all(_g, ice, 0.);
        }

        Field<double> vap = create_field<double>(_g); 
        Field3D<double> ice = create_field3D<double>(_g,_ndust); 

        void set_T_bind(double T) {
            T_bind = T;
        }

};


class MoleculeRef {

    public:

        double m_mol;
        double T_bind;

        MoleculeRef(Molecule& mol) : m_mol(mol.m_mol), T_bind(mol.T_bind), vap(mol.vap), ice(mol.ice) {}

        FieldRef<double> vap; 
        Field3DRef<double> ice; 
        
};

class IceVapChem {

    private:

        GridRef _g;
        FieldConstRef<double> _T;
        Field3DConstRef<double> _J;
        Field3DRef<Prims> _W;
        FieldRef<Prims> _Wg;
        SizeGridIce& _sizes;
        MoleculeRef _mol;
        double _floor;
        int _Jbin_idx;


    public:

        IceVapChem(const Grid& g, const Field<double>& T, WavelengthBinner& bins, const Field3D<double>& J, Field3D<Prims>& W_dust, Field<Prims>& W_gas, SizeGridIce& sizes, 
                        Molecule& mol, double floor = 1.e-100, double N_s = 1.5e15) :
                        _g(g), _T(T), _J(J), _W(W_dust),  _Wg(W_gas), _sizes(sizes), _mol(mol), _floor(floor), N_s(N_s)
                        {
                            for (int i=0; i<bins.num_bands; i++) {
                                if (bins.bands[i] < 0.2) {
                                    _Jbin_idx = i+1;
                                }
                            }
                        } ; 

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
                    
                    f.write((char*) &_mol.vap(i,j), sizeof(double));
                    for (int k=0; k<nspec; k++) {
                        f.write((char*) &_mol.ice(i,j,k), sizeof(double));
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
                    
                    f.read((char*) &_mol.vap(i,j), sizeof(double));
                    double ice_tot = 0;
                    for (int k=0; k<nspec; k++) {
                        f.read((char*) &_mol.ice(i,j,k), sizeof(double));
                        f.read((char*) &_sizes.ice(i,j,k).a, sizeof(double));
                        f.read((char*) &_sizes.ice(i,j,k).rho, sizeof(double));
                    }
                }
            }  
            f.close();
        }

        double N_s;

};

void update_sizegrid(Grid& g, SizeGridIce& sizes, Field3D<Quants>& Qd, Field3D<Quants>& Qice);
void update_sizegrid(Grid& g, SizeGridIce& sizes, Field3D<Prims>& Qd, Field3D<double>& Qice);

#endif// _CUDISC_ICEVAPOUR_H