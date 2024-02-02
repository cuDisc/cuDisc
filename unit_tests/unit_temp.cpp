#include <iostream>
#include <cmath>
#include <sstream>
#include <fstream>
#include <string>

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

void init_dust(Grid& g, Field3D<double>& rho, double Mtot, SizeGrid& sizes) {

    for (int i = 0; i < g.NR + 2*g.Nghost; i++) {

        double Sigtot = std::pow(g.Rc(i)/au, -1.625) * std::exp(-std::pow(3./(g.Rc(i)/au),10));

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
                rho(i,j,k) = rho(i,j,k)*Mtot/M_dust + 1e-100;
            }
        }
    }
}

double T_mid_bench[102] = {307.587408, 307.587408, 301.662721, 295.569876, 289.439440, 283.292300, 277.446215, 271.791091, 266.242574, 260.919363, 255.798959, 250.878610, 246.053191, 
                            241.526483, 224.518298, 220.720820, 217.340539, 214.474545, 212.399328, 211.766492, 214.288216, 220.697219, 227.989226, 208.598468, 191.742469, 173.712574, 
                            156.012222, 140.126486, 126.868327, 116.359785, 108.255819, 101.993240, 97.018691, 92.907731, 89.360209, 86.188018, 83.274457, 80.551091, 77.971042, 75.514718, 
                            73.157198, 70.896681, 68.718034, 66.621562, 64.600714, 62.649837, 60.772299, 58.958054, 57.210491, 55.524617, 53.898225, 52.332311, 50.820856, 49.365843, 47.963889, 
                            46.612565, 45.312903, 44.060281, 42.854632, 41.694846, 40.577229, 39.502440, 38.467653, 37.470753, 36.511713, 35.587775, 34.697892, 33.841042, 33.014889, 32.218569,
                            31.450097, 30.707219, 29.989251, 29.293702, 28.617959, 27.962553, 27.324835, 26.701847, 26.094508, 25.501419, 24.919248, 24.349276, 23.791830, 23.243138, 22.704242, 
                            22.177060, 21.657536, 21.146223, 20.645827, 20.152524, 19.667172, 19.194420, 18.733062, 18.287439, 17.864478, 17.460845, 17.066059, 16.639480, 16.078612, 15.118157, 
                            13.049811, 13.049811};

double T_surf_bench[102] = {295.638275, 295.638275, 290.277985, 284.509569, 278.759762, 273.090513, 267.553266, 262.149315, 256.992260, 252.030604, 247.150437, 242.500405, 238.005509, 
                            233.848324, 227.312449, 223.572136, 219.957094, 216.478557, 213.087765, 209.872141, 206.716194, 203.607888, 200.449463, 197.208940, 193.940038, 190.568102, 
                            187.162021, 183.725537, 180.366013, 177.019286, 173.734484, 170.509038, 167.394679, 164.334549, 161.346923, 158.434961, 155.622963, 152.865948, 150.176624, 
                            147.551871, 145.021842, 142.536712, 140.113426, 137.781541, 135.493730, 133.242092, 131.048226, 128.885419, 126.791986, 124.748340, 122.746503, 120.783001, 
                            118.851838, 116.983661, 115.153757, 113.360183, 111.580537, 109.844949, 108.156123, 106.498682, 104.870420, 103.250390, 101.676492, 100.143032, 98.635265, 
                            97.141088, 95.657595, 94.214862, 92.815418, 91.436756, 90.056073, 88.690587, 87.365574, 86.075652, 84.806282, 83.528038, 82.267655, 81.048292, 79.855837, 
                            78.679605, 77.492487, 76.324071, 75.198730, 74.097536, 73.003250, 71.898936, 70.813783, 69.773896, 68.755449, 67.733490, 66.706905, 65.699831, 64.734196, 
                            63.793175, 62.838731, 61.884481, 60.953418, 60.055196, 59.174280, 58.270132, 57.345321, 57.345321};

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

int main() {

    std::cout << "Test temp...\n";

    Grid::params p;
    p.NR = 100;
    p.Nphi = 100;
    p.Nghost = 1;

    p.Rmin = 1.*au;
    p.Rmax = 100.*au;
    p.theta_min = 0. ;
    p.theta_power = 0.75;
    p.theta_max = M_PI/4.;

    p.R_spacing = RadialSpacing::log ;
    p.theta_spacing = ThetaSpacing::power;

    Grid g(p);

    int n_spec = 100;
    double a0 = 1e-5 ; // Grain size lower bound in cm
    double a1 = 1.   ;  // Grain size upper bound in cm
    SizeGrid sizes(a0, a1, n_spec, 3.5) ;

    double Cv = 2.5*R_gas/2.4;

    int num_wavelengths = 100 ;
    int n_bands = 20;

    DSHARP_opacs opac_tab("./codes/opacities/dustkappa_Draine.txt", false);

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

    init_dust(g, rho,1.e-4*Msun, sizes);

    calculate_total_rhokappa(g, rho, rho_kappa_abs, rho_kappa_sca, opacs);

    compute_total_density(g, rho, rhotot);

    compute_stellar_heating_with_scattering(star, g, rho_kappa_abs, rho_kappa_sca, heating, scattering);

    setup_init_JT(g,heating,J,T);

    FLD_Solver FLD(1, 1e-5, 10000);
    FLD.set_precond_level(0);

    FLD.set_boundaries(BoundaryFlags::open_R_inner | 
                       BoundaryFlags::open_R_outer | 
                       BoundaryFlags::open_Z_outer) ;

    double tol=1;
    int n = 0;
    std::cout.setstate(std::ios_base::failbit);
    while (n<50 && tol>0.000001) {

        Field<double> oldT = create_field<double>(g);
        copy_field(g, T, oldT); 
        
        std::cout << "Iteration: " << n << "\n" ;  

        rho_kappa_abs_binned = bins.bin_planck(g, rho_kappa_abs, T);
        bin_central(g, rho_kappa_sca, rho_kappa_sca_binned, num_wavelengths, n_bands);

        binned_scattering = bins.bin_field(g, scattering, WavelengthBinner::SUM);


        double dt_inittemp = 0.;
        FLD.solve_multi_band(g, dt_inittemp, Cv, rho_kappa_abs_binned, rho_kappa_sca_binned, rhotot, heating, binned_scattering, bins.edges, T, J);

        tol = fracerr(g, oldT, T);

    }  
    std::cout.clear();

    double L2_mid = 0, L2_surf = 0;
    for (int i=0; i<g.NR+2*g.Nghost; i++) {
        L2_mid += (std::pow(T(i,1)-T_mid_bench[i], 2.)/(g.NR+2*g.Nghost));
        L2_surf += (std::pow(T(i,60)-T_surf_bench[i], 2.)/(g.NR+2*g.Nghost));
    }
    L2_mid = std::sqrt(L2_mid);
    L2_surf = std::sqrt(L2_surf);

    if (L2_mid <= 1e-5) {printf("\nL2_mid = %g, pass\n", L2_mid);}
    else {printf("\nL2_mid = %g, fail\n", L2_mid);}
    if (L2_surf <= 1e-5) {printf("L2_surf = %g, pass\n\n", L2_surf);}
    else {printf("L2_surf = %g, fail\n\n", L2_surf);}
}
