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

void setup_init_JT(const Grid &g, Field<double> &heat, Field3D<double> &J, Field<double> &T) {

    // Sets initial radiative flux (J=cE_R where E_R is the radiative energy) for temperature calculations

    for (int i=0; i<g.NR + 2*g.Nghost; i++) {
        for (int j=0; j<g.Nphi + 2*g.Nghost; j++) {
            for (int k=0; k<J.Nd; k++) {
                J(i,j,k) = heat(i,j)/J.Nd ; 
            }    
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


void calculate_total_rhokappa(const Grid& g, Field3D<double>& rho, Field3D<double>& rhokabs, Field3D<double>& rhoksca, DSHARP_opacs& opacs) {


    for (int i = 0; i < g.NR + 2*g.Nghost; i++) {
        for (int j = 0; j < g.Nphi + 2*g.Nghost; j++) {
            for (int k = 0; k < opacs.n_lam; k++) {

                double rhok_dust_abs = 0;
                double rhok_dust_sca = 0;

                for (int l=0; l<opacs.n_a; l++) { 
                    rhok_dust_abs += rho(i,j,l)*opacs.k_abs(l,k);
                    rhok_dust_sca += rho(i,j,l)*opacs.k_sca(l,k);
                }

                rhokabs(i,j,k) = rhok_dust_abs;
                rhoksca(i,j,k) = rhok_dust_sca;
            }
        }
    }
}

void compute_total_density(Grid& g, Field3D<double>& rho, Field<double>& rhotot) {

    for (int i=0; i<g.NR + 2*g.Nghost; i++) {
        for (int j=0; j<g.Nphi + 2*g.Nghost; j++) {   
            double rho_tot_temp = 0.;
            for (int k=0; k<rho.Nd; k++) {
                rho_tot_temp += rho(i,j,k);
            }    
            rhotot(i,j) = rho_tot_temp;
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

        std::filesystem::path dir = std::string("../outputs/pinte/run_") + run[i];
        std::filesystem::create_directories(dir);

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

        DSHARP_opacs opac_tab("../../codes/opacities/dustkappa_Draine.txt", false);

        DSHARP_opacs opacs(n_spec, num_wavelengths); 
        opacs.generate_lam(1.e-1,3.e3); 
        opacs.generate_a(sizes); 
        opacs.interpolate_opacs(opac_tab);

        Star star(GMsun, 1.105*Lsun, 4000) ;
        star.set_wavelengths(num_wavelengths, opacs.lam()) ;
        star.set_blackbody_fluxes() ;

        Field3D<double> rho = create_field3D<double>(g, sizes.size());
        Field<double> rhotot = create_field<double>(g);
        Field<double> T = create_field<double>(g);
        Field3D<double> J = create_field3D<double>(g, n_bands);
        Field<double> heating = create_field<double>(g) ; 
        Field3D<double> scattering = create_field3D<double>(g, opacs.n_lam) ; 
        Field3D<double> binned_scattering = create_field3D<double>(g, n_bands) ; 
        Field3D<double> rho_kappa_abs = create_field3D<double>(g, opacs.n_lam);  
        Field3D<double> rho_kappa_sca = create_field3D<double>(g, opacs.n_lam);  
        Field3D<double> rho_kappa_abs_binned =  create_field3D<double>(g, n_bands);
        Field3D<double> rho_kappa_sca_binned =  create_field3D<double>(g, n_bands);

        WavelengthBinner bins(opacs.n_lam, opacs.lam(), n_bands);

        write_grids(dir, &g, &sizes, &opacs, &bins);

        init_dust(g, rho, masses[i]*Msun, sizes);

        calculate_total_rhokappa(g, rho, rho_kappa_abs, rho_kappa_sca, opacs);

        compute_total_density(g, rho, rhotot);

        compute_stellar_heating_with_scattering(star, g, rho_kappa_abs, rho_kappa_sca, heating, scattering);

        setup_init_JT(g,heating,J,T);

        FLD_Solver FLD(1, 1e-5, 10000);
        FLD.set_precond_level(1);

        FLD.set_boundaries(BoundaryFlags::open_R_inner | 
                        BoundaryFlags::open_R_outer | 
                        BoundaryFlags::open_Z_outer) ;

        double tol=1;
        int n = 0;
        while (n<10 && tol>0.00001) {

            Field<double> oldT = create_field<double>(g);
            Field3D<double> oldJ = create_field3D<double>(g,n_bands);
            copy_field(g, T, oldT); 
            copy_field(g, J, oldJ); 
            
            std::cout << "Iteration: " << n << "\n" ;  

            rho_kappa_abs_binned = bins.bin_planck(g, rho_kappa_abs, T);
            bin_central(g, rho_kappa_sca, rho_kappa_sca_binned, num_wavelengths, n_bands);

            binned_scattering = bins.bin_field(g, scattering, WavelengthBinner::SUM);


            double dt_inittemp = 0;
            if (n==0) { dt_inittemp = 0; }

            FLD.solve_multi_band(g, dt_inittemp, Cv, rho_kappa_abs_binned, rho_kappa_sca_binned, rhotot, heating, binned_scattering, bins.edges, T, J);

            std::cout << "T:" << T[T.index(1,1)] << " " << T[T.index(30, 2)] 
                    << " "<< T[T.index(1, g.Nphi)] << " " <<  T[T.index(g.NR, g.Nphi)]
                    << "\n" << std::endl ;

            tol = fracerr(g, oldT, T);
            std::cout << "Fractional error: "<< tol << "\n" << "\n";
            n += 1;
        }  

        write_temp(dir, 0, g, T, J);
    }
    return 0;
}
