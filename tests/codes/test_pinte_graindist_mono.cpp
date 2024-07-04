#include <iostream>
#include <cmath>
#include <sstream>
#include <fstream>
#include <string>
#include <filesystem>

#include "cuda_array.h"
#include "grid.h"
#include "field.h"
#include "constants.h"
#include "hydrostatic.h"
#include "stellar_irradiation.h"
#include "DSHARP_opacs.h"
#include "FLD.h"
#include "bins.h"
#include "file_io.h"
#include "errorfuncs.h"

void setup_init_JT(const Grid &g, Field<double> &heat, Field<double> &J, Field<double> &T) {

    // Sets initial radiative flux (J=cE_R where E_R is the radiative energy) for temperature calculations

    for (int i=0; i<g.NR + 2*g.Nghost; i++) {
        for (int j=0; j<g.Nphi + 2*g.Nghost; j++) {
            J(i,j) = heat(i,j) ;
            T(i,j) = 150. / std::sqrt(g.Rc(i)/au);
        }
    }
}


void init_density_structure(const Grid& g, Field3D<double>& rho, SizeGrid& sizes, double tau_max, double kappa_810) {

    double R0 = 100*au ;
    double rho0_R0 = (tau_max / kappa_810) /
        (std::pow(g.Re(0)/R0, -1.625) - std::pow(g.Re(g.NR+2*g.Nghost)/R0, -1.625)) ;

    double tau = 0 ;

    for (int i = 0; i < g.NR + 2*g.Nghost; i++) {
        double rho_0 = (rho0_R0 / g.dRe(i)) * 
            (std::pow(g.Re(i)/R0, -1.625) - std::pow(g.Re(i+1)/R0, -1.625)) ;

        double h = (10*au) * std::pow(g.Rc(i)/R0, 1.125) ;
        tau += rho_0 * g.dRe(i) * kappa_810 ;
        for (int k=0; k<100; k++) {
            double St = sizes.centre_size(k)*0.05 * (g.Rc(i)/au);
            double hd = h * std::sqrt(1/(1+St/1e-3));
            for (int j = 0; j < g.Nphi + 2*g.Nghost; j++) {
                rho(i,j,k) = rho_0 * std::exp(-0.5 * std::pow(g.Zc(i,j)/hd, 2)) ;
            }
        }
    }
    std::cout << "tau:" << tau << "\n";

}

void init_dust(Grid& g, Field3D<double>& rho, double Mtot, SizeGrid& sizes) {

    for (int i = 0; i < g.NR + 2*g.Nghost; i++) {

        double Sigtot = std::pow(g.Rc(i)/au, -1.625);

        double h = (10*au) * std::pow(g.Rc(i)/(100*au), 1.125) ;

        for (int j=0; j<g.Nphi+2*g.Nghost; j++) {  
            double ftot = 0;  

            for (int k=0; k<100; k++) {

                double St = sizes.centre_size(k)*0.05 * (g.Rc(i)/au);
                double hd = h * std::sqrt(1/(1+St/1e-3));
                double fk = std::pow(sizes.centre_size(k)/sizes.centre_size(0), 0.5) * std::exp(-std::pow(sizes.centre_size(k)/0.5, 5.));

                rho(i,j,k) = fk * Sigtot/(std::sqrt(2*M_PI)*hd) * std::exp(-(g.Zc(i,j)*g.Zc(i,j)/(2.*hd*hd)));

                ftot += fk;
            }
            for (int k=0; k<100; k++) {
                rho(i,j,k) /= ftot;
            }
        }
    }
    double M_dust=0;
    for (int i=0; i<g.NR + 2*g.Nghost; i++) {
        for (int j=g.Nghost; j<g.Nphi + 2*g.Nghost; j++) {
            for (int k=0; k<rho.Nd; k++) {
                M_dust += 4.*M_PI * rho(i,j,k) * g.volume(i,j);
            }
        }
    }
    for (int i=0; i<g.NR + 2*g.Nghost; i++) {
        for (int j=g.Nghost; j<g.Nphi + 2*g.Nghost; j++) {
            for (int k=0; k<rho.Nd; k++) {
                rho(i,j,k) *= Mtot/M_dust;
            }
        }
    }
}


void calculate_total_rhokappa(const Grid& g, Field3D<double>& rho, Field3D<double>& kabs, Field3D<double>& ksca, Field<double>& rhotot, DSHARP_opacs& opacs) {


    for (int i = 0; i < g.NR + 2*g.Nghost; i++) {
        for (int j = 0; j < g.Nphi + 2*g.Nghost; j++) {
            for (int k = 0; k < opacs.n_lam; k++) {

                double rhok_dust_abs = 0;
                double rhok_dust_sca = 0;
                double rho_ij = 0 ;

                for (int l=0; l<opacs.n_a; l++) { 
                    rhok_dust_abs += rho(i,j,l)*opacs.k_abs(l,k);
                    rhok_dust_sca += rho(i,j,l)*opacs.k_sca(l,k);
                    rho_ij += rho(i,j,l) ; 
                }

                kabs(i,j,k) = rhok_dust_abs;
                ksca(i,j,k) = rhok_dust_sca; 
                rhotot(i,j) = rho_ij;
            }
        }
    }
}

Grid setup_grid(double tau_max) {

    // Setup grid with small zones near inner boundary such that dtau < 1.
    double R0 = 0.1*au ;
    double R1 = 400*au ;
    double R_join = 0.15*au ;
    int Nouter = 200;

    // Work out number of cells in the inner grid:
    double dR0 = R0 / (1.625 * tau_max) ;
    double Delta = std::exp(std::log(R1/R_join) / Nouter) - 1 ;
    double delta = (R_join*Delta - dR0*(1+Delta)) / (R_join - R0*(1+Delta)) ;
    std::cout << "Delta, delta: " << Delta << " " << delta << "\n" ;

    int n_inner = 1;
    while ((R_join-R0)*delta > (std::pow(1+delta, n_inner) - 1)*dR0)
        n_inner++ ;
    std::cout << "Number of zones in inner region: " << n_inner << "\n" ;

    // Correct delta to make the join exact:
    std::cout << "dR0 target, actual: " << dR0/au << " " ;
    dR0 = (R_join-R0)*delta / (std::pow(1+delta, n_inner) - 1) ;
    std::cout << dR0/au << "\n" ;

    // Create radial grid:
    CudaArray<double> Re = make_CudaArray<double>(n_inner+Nouter+1) ;
    Re[0] = R0 ;
    for (int i=0; i < n_inner; i++) {
        Re[i+1] = Re[i] + dR0 ;
        dR0 *= 1 + delta ;
    }
    std::cout << "Rjoin (au): " << Re[n_inner]/au << "\n" ;
    for (int i=n_inner; i < n_inner+Nouter; i++) {
        Re[i+1] = Re[i] * (1 + Delta) ;
    }
    // Check the outer boundary:
    std::cout << "Outer radius (au): " << Re[n_inner+Nouter]/au << "\n" ;

    // Create azimuthal grid:
    int nPhi = 150 ;
    int Nghost = 1 ;
    double dphi = M_PI / (4*nPhi) ;

    CudaArray<double> phi = make_CudaArray<double>(nPhi + 2*Nghost + 1) ;
    
    for (int i=0; i <= nPhi + 2*Nghost; i++)
        phi[i] = (i-Nghost)*dphi;

    return Grid(n_inner+Nouter-2*Nghost, nPhi, Nghost, std::move(Re), std::move(phi)) ;
}


int main() {

    const char* run[2] = {"thin", "thick"};
    const double taus[2] = {1.22e3, 1.22e6};
    const double masses[2] = {3.01798e-8, 3.01798e-5};

    for (int i=0; i<2; i++) {
        
        std::filesystem::path path = __FILE__;
        path = (path.parent_path()).parent_path();
        std::filesystem::path dir = path / (std::string("outputs/pinte_mono/run_")+ run[i]);
        std::filesystem::create_directories(dir);

        std::cout << "Output directory: " << dir  << "\n";

        double tau_max = taus[i] ; 
        Grid g = setup_grid(tau_max);
        // g.write_grid(dir);

        int n_spec = 100;
        double a0 = 1e-5 ; // Grain size lower bound in cm
        double a1 = 1.   ;  // Grain size upper bound in cm
        SizeGrid sizes(a0, a1, n_spec, 3.5) ;

        double Cv = 100* 2.5*R_gas/2.4;

        int num_wavelengths = 100 ;
        int n_bands = 20;

        DSHARP_opacs opac_tab(path / "../codes/opacities/dustkappa_Draine.txt", false);

        DSHARP_opacs opacs(n_spec, num_wavelengths); 
        opacs.generate_lam(1.e-1,3.e3); 
        opacs.generate_a(sizes); 
        opacs.interpolate_opacs(opac_tab);

        Star star(GMsun, 1.105*Lsun, 4000) ;
        star.set_wavelengths(num_wavelengths, opacs.lam()) ;
        star.set_blackbody_fluxes() ;

        Field3D<double> rho = create_field3D<double>(g, sizes.size());
        Field<double> T = create_field<double>(g);
        Field<double> J = create_field<double>(g);
        Field<double> heating = create_field<double>(g) ; 
        Field3D<double> scattering = create_field3D<double>(g, opacs.n_lam) ; 
        Field3D<double> binned_scattering = create_field3D<double>(g, n_bands) ; 
        Field3D<double> kappa_abs = create_field3D<double>(g, opacs.n_lam);  
        Field3D<double> kappa_sca = create_field3D<double>(g, opacs.n_lam);  
        Field<double> rhotot = create_field<double>(g);

        WavelengthBinner bins(opacs.n_lam, opacs.lam(), n_bands);

        write_grids(dir, &g, &sizes, &opacs, &bins);

        init_dust(g, rho, masses[i]*Msun, sizes);

        calculate_total_rhokappa(g, rho, kappa_abs, kappa_sca, rhotot, opacs);

        compute_stellar_heating(star, g, kappa_abs, heating);

        setup_init_JT(g,heating,J,T);

        FLD_Solver FLD(1, 1e-5, 10000);
        FLD.set_precond_level(1);

        FLD.set_boundaries(BoundaryFlags::open_R_inner | 
                          BoundaryFlags::open_R_outer | 
                          BoundaryFlags::open_Z_outer) ;

        double tol=1;
        int n = 0;
        while (n<50 && tol>0.00001) {

            Field<double> oldT = create_field<double>(g);
            copy_field(g, T, oldT); 
            
            std::cout << "Iteration: " << n << "\n" ;  

            Field<double> rho_kappa_P = bins.planck_mean(g, kappa_abs, T);


            double dt_inittemp = 0;
            if (n==0) { dt_inittemp = 0; }

            FLD(g, dt_inittemp, Cv, rho_kappa_P, rho_kappa_P, rhotot, heating, T, J);

            std::cout << "T:" << T(1,1) << " " << T(30, 2) 
                      << " "<< T(1, g.Nphi) << " " <<  T(g.NR, g.Nphi)
                      << "\n" << std::endl ;

            tol = fracerr(g, oldT, T);
            std::cout << "Fractional error: "<< tol << "\n" << "\n";
            n += 1;
        }  

        write_temp(dir, 0, g, T, J);
    }
    return 0;
}
