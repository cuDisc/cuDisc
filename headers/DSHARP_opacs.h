#ifndef _CUDISC_DSHARP_OPACS_H_
#define _CUDISC_DSHARP_OPACS_H_

#include <iostream>
#include <fstream>
#include <field.h>
#include <string>
#include <cmath>
#include <vector>
#include <array>
#include "cuda_array.h"
#include "interpolate.h"
#include "constants.h"
#include "coagulation/size_grid.h"
#include "dustdynamics.h"


const double c = 2.99792458e10;
const double h = 6.6260755e-27;
    
inline double planck(double wle, double T) {

    double B = std::pow(wle, -5.) * 1/(std::exp((h*c)/(wle*k_B*T))-1); 

    return B;
}

inline double dplanck(double nu, double T) {

    double dB = (2. * h * h * std::pow(nu, 4.))/(k_B*c*c) * std::exp((h*nu)/(k_B*T)) * std::pow(std::pow(std::exp((h*nu)/(k_B*T))-1,2.),-1) * (1/(T*T)) ; 

    if (dB != dB) { dB = 0; }

    return dB;
}

class DSHARP_opacsRef;

// struct opac_data {

//     int n;
//     int m;
//     Field<double> a = Field<double>(n,m)

// }

/* class DSHARP_opacs
 *
 * Reads in DSHARP opacity file
 *
 */
class DSHARP_opacs_MRN {
    public:
        int n;
        double* wavelengths;
        double* k_abs;
        double* k_sca;
        double* g_assym;
        double* wle_cm;

        DSHARP_opacs_MRN(std::string filename) : _filename(filename) {
            
            std::ifstream opac_f(_filename);
    
            opac_f >> n;

            double* _wavelengths = (double*)malloc(sizeof(double)*n);
            double* _wle_cm = (double*)malloc(sizeof(double)*n);
            double* _k_abs = (double*)malloc(sizeof(double)*n);
            double* _k_sca = (double*)malloc(sizeof(double)*n);
            double* _g_assym = (double*)malloc(sizeof(double)*n);

            for (int i=0; i<n; i++) { opac_f >> _wavelengths[i]; }
            for (int i=0; i<n; i++) { opac_f >> _k_abs[i]; }
            for (int i=0; i<n; i++) { opac_f >> _k_sca[i]; }
            for (int i=0; i<n; i++) { opac_f >> _g_assym[i]; }

            wavelengths = _wavelengths;
            k_abs = _k_abs;
            k_sca = _k_sca;
            g_assym = _g_assym;
            
            for (int i=0; i<n; i++) { _wle_cm[i] = _wavelengths[i]/1e4; }
            wle_cm = _wle_cm;

        }

        double planck_mean(double T) {
            double B[n];
            double int_1 = 0;
            double int_2 = 0;

            for (int i=0; i<n; i++) { 
                B[i] = planck(wle_cm[i], T); 
                if (i>0) { 
                    int_1 += 0.5*(B[i]*k_abs[i] + B[i-1]*k_abs[i-1]) * (wle_cm[i]-wle_cm[i-1]);
                    int_2 += 0.5*(B[i]+B[i-1]) * (wle_cm[i]-wle_cm[i-1]);
                }
            }
            
            double kappa_planck = int_1/int_2; 

            return kappa_planck; 
        }


        double rosseland_mean_iso(double T) {
            
            double nu[n];
            double dB_nu[n];
            double int_1 = 0;
            double int_2 = 0;

            for (int i=0; i<n; i++) { 
                nu[i] = (c*1e4) / wavelengths[i]; 
                dB_nu[i] = dplanck(nu[i], T); 
                std::cout << dB_nu[i] <<std::endl;
                if (i>0) { 
                    int_1 += -0.5*(dB_nu[i]*(std::pow(k_abs[i]+k_sca[i],-1)) + dB_nu[i-1]*(std::pow(k_abs[i-1]+k_sca[i-1],-1))) * (nu[i]-nu[i-1]);
                    int_2 += -0.5*(dB_nu[i]+dB_nu[i-1]) * (nu[i]-nu[i-1]);
                }
            }
            
            double kappa_rosseland = int_2/int_1; 

            return kappa_rosseland;
        }
        
    private:
        std::string _filename;
} ;

class DSHARP_opacs {

    // arrays are indexed as (a,lam) for (i,j)

    public:
        int n_a;
        int n_lam;

        DSHARP_opacs() {};

        DSHARP_opacs(int _n_a, int _n_lam) : n_a(_n_a), n_lam(_n_lam) {
            
            a_ptr = make_CudaArray<double>(n_a);
            lam_ptr = make_CudaArray<double>(n_lam);
            k_abs_ptr = make_CudaArray<double>(n_a*n_lam);
            k_sca_ptr = make_CudaArray<double>(n_a*n_lam);
            k_abs_g_ptr = make_CudaArray<double>(n_lam);
            k_sca_g_ptr = make_CudaArray<double>(n_lam);

        } 

        DSHARP_opacs(std::string filename, bool with_g=1) : _filename(filename) {
            
            std::ifstream opac_f(filename);

            opac_f >> n_a;
            opac_f >> n_lam;

            a_ptr = make_CudaArray<double>(n_a);
            lam_ptr = make_CudaArray<double>(n_lam);
            k_abs_ptr = make_CudaArray<double>(n_a*n_lam);
            k_sca_ptr = make_CudaArray<double>(n_a*n_lam);
            
            if (with_g == 1) {
                CudaArray<double> g = make_CudaArray<double>(n_a*n_lam);

                for (int i=0; i<n_a; i++) {
                    for (int j=0; j<n_lam; j++) {
                        opac_f >> a_ptr[i];
                        opac_f >> lam_ptr[j];
                        opac_f >> k_abs_ptr[i*n_lam+j];
                        opac_f >> k_sca_ptr[i*n_lam+j];
                        opac_f >> g[i*n_lam+j];
                    }
                }

                for (int i=0; i<n_a; i++) {
                    for (int j=0; j<n_lam; j++) {
                        k_sca_ptr[i*n_lam+j] *= (1.-g[i*n_lam+j]);
                    }
                }
            }
            
            else {
                for (int i=0; i<n_a; i++) {
                    for (int j=0; j<n_lam; j++) {
                        opac_f >> a_ptr[i];
                        opac_f >> lam_ptr[j];
                        opac_f >> k_abs_ptr[i*n_lam+j];
                        opac_f >> k_sca_ptr[i*n_lam+j];
                    }
                }
            }
        } 

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


        void generate_lam(double lam_min, double lam_max) {

            double dloglam = (std::log10(lam_max) - std::log10(lam_min)) / static_cast<double>(n_lam-1);

            for (int i=0; i<n_lam; i++) {
                lam_ptr[i] = std::pow(10., std::log10(lam_min) + dloglam*i); 
            }
        }

        void generate_a(SizeGrid& sizes) {

            for (int i=0; i<n_a; i++) {
                a_ptr[i] = static_cast<double>(sizes.centre_size(i)); 
            }
        }

        void set_k_g_min_grain(double fac) {

            for (int i=0; i<n_lam; i++) { 
                k_abs_g_ptr[i] = fac*k_abs_ptr[i];
                k_sca_g_ptr[i] = fac*k_sca_ptr[i];
            }

        }

        void interpolate_opacs(const DSHARP_opacs&) ;
        void write_interp(std::filesystem::path filename) const; 

    private:
        std::string _filename;
        CudaArray<double> a_ptr;
        CudaArray<double> lam_ptr;
        CudaArray<double> k_abs_ptr;
        CudaArray<double> k_sca_ptr;
        CudaArray<double> k_abs_g_ptr;
        CudaArray<double> k_sca_g_ptr;

        friend class DSHARP_opacsRef;
} ;

class DSHARP_opacsRef {
    public:
        int n_a;
        int n_lam;

        DSHARP_opacsRef(DSHARP_opacs& opac) : 
            n_a(opac.n_a), n_lam(opac.n_lam), a_ptr(opac.a_ptr.get()),
            lam_ptr(opac.lam_ptr.get()), k_abs_ptr(opac.k_abs_ptr.get()), k_sca_ptr(opac.k_sca_ptr.get()),
            k_abs_g_ptr(opac.k_abs_g_ptr.get()), k_sca_g_ptr(opac.k_sca_g_ptr.get())  
            {} ;

        __host__ __device__ 
        double k_abs(int i, int j) const {
            return k_abs_ptr[i*n_lam + j];
        }
        __host__ __device__ 
        double k_sca(int i, int j) const {
            return k_sca_ptr[i*n_lam + j];
        }
        __host__ __device__ 
        double k_abs_g(int j) const {
            return k_abs_g_ptr[j];
        }
        __host__ __device__ 
        double k_sca_g(int j) const {
            return k_sca_g_ptr[j];
        }
        __host__ __device__ 
        double a(int i) const {
            return a_ptr[i];
        }
        __host__ __device__ 
        double lam(int i) const {
            return lam_ptr[i];
        }
        __host__ __device__ 
        double* k_abs() {
            return k_abs_ptr;
        }
        __host__ __device__ 
        double* k_sca() {
            return k_sca_ptr;
        }
        __host__ __device__ 
        double* a() {
            return a_ptr;
        }
        __host__ __device__ 
        double* lam() {
            return lam_ptr;
        }
     
    private:
        double* a_ptr;
        double* lam_ptr;
        double* k_abs_ptr;
        double* k_sca_ptr;
        double* k_abs_g_ptr;
        double* k_sca_g_ptr;
} ;

void calculate_total_rhokappa(Grid& g, Field3D<Prims>& qd, Field<Prims>& wg, DSHARP_opacs& opacs,
                                    Field3D<double>& rhokappa_abs, Field3D<double>& rhokappa_sca, 
                                    double kgas_abs, double kgas_sca);

void calculate_total_rhokappa(Grid& g, Field3D<Prims>& qd, Field<Prims>& wg, DSHARP_opacs& opacs,
                                    Field3D<double>& rhokappa_abs, Field3D<double>& rhokappa_sca);

void calculate_grain_rhokappa(Grid& g, Field3D<Prims>& qd, DSHARP_opacs& opacs,
                                    Field3D<double>& rhokappa_abs_grain, Field3D<double>& rhokappa_sca_grain);

#endif//_CUDISC_DSHARP_OPACS_H_