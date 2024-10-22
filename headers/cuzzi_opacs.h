#ifndef _CUDISC_CUZZI_OPACS_H_
#define _CUDISC_CUZZI_OPACS_H_

#include <iostream>
#include <fstream>
#include <field.h>
#include <string>
#include <cmath>
#include <vector>
#include <array>
#include "cuda_array.h"
#include "icevapour.h"
#include "interpolate.h"
#include "constants.h"
#include "coagulation/size_grid.h"
#include "dustdynamics.h"

struct RefIndx {
    double n, k;
};

struct Comp {
    double dens, mf;
    CudaArray<RefIndx> opt;
};

struct CompRef {
    double dens, mf;
    RefIndx* opt;
    CompRef(Comp& comp) : dens(comp.dens), mf(comp.mf), opt(comp.opt.get()) {}
};

class CuzziComp {
    public:
        Comp ice, sil, FeS, Fe, org;
        int n_comp = 5;
        
        CuzziComp() {};

        CuzziComp(int n_lam, double* lam);

        inline __host__ __device__ Comp& operator[](int i) {
            if (i==0) { return ice; } 
            if (i==1) { return sil; }
            if (i==2) { return FeS; }
            if (i==3) { return Fe; }
            return org;
        } ;
};

class DSHARPComp {
    public:
        Comp ice, sil, FeS, org;
        int n_comp = 4;
        
        DSHARPComp() {};

        DSHARPComp(int n_lam, double* lam);

        inline __host__ __device__ Comp& operator[](int i) {
            if (i==0) { return ice; } 
            if (i==1) { return sil; }
            if (i==2) { return FeS; }
            return org;
        } ;
};

class DSHARPwCOComp {
    public:
        Comp ice, sil, FeS, org, CO;
        int n_comp = 5;
        
        DSHARPwCOComp() {};

        DSHARPwCOComp(int n_lam, double* lam);

        inline __host__ __device__ Comp& operator[](int i) {
            if (i==0) { return ice; } 
            if (i==1) { return sil; }
            if (i==2) { return FeS; }
            if (i==3) { return org; }
            return CO;
        } ;
};

class DSHARPwCOCompRef {
    public:
        CompRef ice, sil, FeS, org, CO;
        int n_comp = 5;
        
        DSHARPwCOCompRef(DSHARPwCOComp& comp) : ice(comp.ice), sil(comp.sil), FeS(comp.FeS), org(comp.org), CO(comp.CO) {};

        inline __host__ __device__ CompRef& operator[](int i) {
            if (i==0) { return ice; } 
            if (i==1) { return sil; }
            if (i==2) { return FeS; }
            if (i==3) { return org; }
            return CO;
        } ;
};

template<class CompMix>
class CuzziOpacs {

    // arrays are indexed as (a,lam) for (i,j)

    public:
        int n_a;
        int n_lam;
        CompMix comp; 
        double por = 0.;

        CuzziOpacs() {};

        CuzziOpacs(int _n_a, int _n_lam, double lam_min, double lam_max);

        CuzziOpacs(SizeGrid& sizes, int _n_lam) : n_a(sizes.size()), n_lam(_n_lam) {
            
            for (int i=0; i<n_a; i++) {
                a_ptr[i] = static_cast<double>(sizes.centre_size(i)); 
            }
            lam_ptr = make_CudaArray<double>(n_lam);
            k_abs_ptr = make_CudaArray<double>(n_a*n_lam);
            k_sca_ptr = make_CudaArray<double>(n_a*n_lam);
            k_abs_g_ptr = make_CudaArray<double>(n_lam);
            k_sca_g_ptr = make_CudaArray<double>(n_lam);

        } 

        void set_porosity(double _por) {
            por = _por;
        }

        void calc_opacs(SizeGrid& sizes, double por=0.);

        double k_abs(int i, int j) const {
            return k_abs_ptr[i*n_lam + j];
        }
        double k_sca(int i, int j) const {
            return k_sca_ptr[i*n_lam + j];
        }
        double k_abs_g(int j) const {
            return k_abs_g_ptr[j];
        }
        double k_sca_g(int j) const {
            return k_sca_g_ptr[j];
        }
        double a(int i) const {
            return a_ptr[i];
        }
        double lam(int i) const {
            return lam_ptr[i];
        }

        double* k_abs() {
            return k_abs_ptr.get();
        }
        double* k_sca() {
            return k_sca_ptr.get();
        }
        double* a() {
            return a_ptr.get();
        }
        double* lam() {
            return lam_ptr.get();
        }

        void set_k_g_min_grain(double fac) {

            for (int i=0; i<n_lam; i++) { 
                k_abs_g_ptr[i] = fac*k_abs_ptr[i];
                k_sca_g_ptr[i] = fac*k_sca_ptr[i];
            }

        }

        void write_interp(std::filesystem::path filename) const; 

    private:
        std::string _filename;
        CudaArray<double> a_ptr;
        CudaArray<double> lam_ptr;
        CudaArray<double> k_abs_ptr;
        CudaArray<double> k_sca_ptr;
        CudaArray<double> k_abs_g_ptr;
        CudaArray<double> k_sca_g_ptr;

        // friend class CuzziOpacsRef<CompMix>;
} ;

// template<class CompMix>
// class CuzziOpacsRef {
//     public:
//         int n_a;
//         int n_lam;
//         CompMix comp; 
//         double por;

//         CuzziOpacsRef(CuzziOpacs<CompMix>& opac) : 
//             n_a(opac.n_a), n_lam(opac.n_lam), comp(opac.comp), por(opac.por), a_ptr(opac.a_ptr.get()),
//             lam_ptr(opac.lam_ptr.get()), k_abs_ptr(opac.k_abs_ptr.get()), k_sca_ptr(opac.k_sca_ptr.get()),
//             k_abs_g_ptr(opac.k_abs_g_ptr.get()), k_sca_g_ptr(opac.k_sca_g_ptr.get())
//             {} ;

//         __host__ __device__ 
//         double k_abs(int i, int j) const {
//             return k_abs_ptr[i*n_lam + j];
//         }
//         __host__ __device__ 
//         double k_sca(int i, int j) const {
//             return k_sca_ptr[i*n_lam + j];
//         }
//         __host__ __device__ 
//         double k_abs_g(int j) const {
//             return k_abs_g_ptr[j];
//         }
//         __host__ __device__ 
//         double k_sca_g(int j) const {
//             return k_sca_g_ptr[j];
//         }
//         __host__ __device__ 
//         double a(int i) const {
//             return a_ptr[i];
//         }
//         __host__ __device__ 
//         double lam(int i) const {
//             return lam_ptr[i];
//         }
//         __host__ __device__ 
//         double* k_abs() {
//             return k_abs_ptr;
//         }
//         __host__ __device__ 
//         double* k_sca() {
//             return k_sca_ptr;
//         }
//         __host__ __device__ 
//         double* a() {
//             return a_ptr;
//         }
//         __host__ __device__ 
//         double* lam() {
//             return lam_ptr;
//         }
     
//     private:
//         double* a_ptr;
//         double* lam_ptr;
//         double* k_abs_ptr;
//         double* k_sca_ptr;
//         double* k_abs_g_ptr;
//         double* k_sca_g_ptr;
// } ;

// template<typename T>
void calculate_total_rhokappa(Grid& g, SizeGridIce& sizes, Field3D<Prims>& qd, Field<Prims>& wg, Field<double>& rho_tot, CuzziOpacs<DSHARPwCOComp>& opacs,
                                    Field3D<double>& rhokappa_abs, Field3D<double>& rhokappa_sca, Molecule& mol);
void calculate_total_rhokappa(Grid& g, Grid& g_in, SizeGridIce& sizes, Field3D<double>& rho_d, Field<Prims>& wg, Field<double>& rho_tot, CuzziOpacs<DSHARPwCOComp>& opacs,
                                    Field3D<double>& rhokappa_abs, Field3D<double>& rhokappa_sca, Molecule& mol);

// class CuzziOpacs<DSHARPComp>;


#endif//_CUDISC_CUZZI_OPACS_H_