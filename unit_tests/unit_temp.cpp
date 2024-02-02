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

double T_mid_bench[102] = {307.562943, 307.562943, 301.620814, 295.517878, 289.381782, 283.231728, 277.383776, 271.728409, 266.178363, 260.859758, 255.745839, 250.835220, 246.017804, 241.481783, 224.409284, 220.496279, 216.985788, 214.093288, 212.188818, 211.808566, 214.464455, 220.846331, 228.094907, 208.730048, 191.880958, 173.820906, 156.059908, 140.101013, 126.774003, 116.200043, 108.033795, 101.717711, 96.702100, 92.562783, 88.999289, 85.818004, 82.903901, 80.182482, 77.609487, 75.161505, 72.814873, 70.565984, 68.398847, 66.315981, 64.307605, 62.370748, 60.505430, 58.703256, 56.967222, 55.290612, 53.673612, 52.115126, 50.610437, 49.162548, 47.766827, 46.422872, 45.130567, 43.885688, 42.688594, 41.537107, 40.427749, 39.360965, 38.333124, 37.342582, 36.388705, 35.468798, 34.581967, 33.727110, 32.902201, 32.106225, 31.337512, 30.594138, 29.875239, 29.178368, 28.502356, 27.846913, 27.209460, 26.588227, 25.984447, 25.395910, 24.819954, 24.258697, 23.710751, 23.172366, 22.644605, 22.127563, 21.616470, 21.111454, 20.614332, 20.121073, 19.633505, 19.156995, 18.691361, 18.243244, 17.819526, 17.416609, 17.023961, 16.600042, 16.041242, 15.083613, 13.020229, 13.020229};

double T_surf_bench[102] = {295.610455, 295.610455, 290.227536, 284.445519, 278.688104, 273.013066, 267.469003, 262.063398, 256.902439, 251.939974, 247.057248, 242.410377, 237.917573, 233.760849, 227.233605, 223.502472, 219.900175, 216.432199, 213.046280, 209.823840, 206.648908, 203.514872, 200.332133, 197.077576, 193.799438, 190.428842, 187.024904, 183.595604, 180.241414, 176.903716, 173.625403, 170.409508, 167.300245, 164.247629, 161.265732, 158.361350, 155.553484, 152.801961, 150.116861, 147.497370, 144.970285, 142.488979, 140.068854, 137.740150, 135.456147, 133.207321, 131.016007, 128.855572, 126.764791, 124.722966, 122.722977, 120.760691, 118.831286, 116.964382, 115.135662, 113.343464, 111.564810, 109.830581, 108.142928, 106.486454, 104.859127, 103.239982, 101.666984, 100.134259, 98.627166, 97.133694, 95.650748, 94.208364, 92.809429, 91.431038, 90.050967, 88.685659, 87.360291, 86.070657, 84.801770, 83.523857, 82.263271, 81.042996, 79.850318, 78.675110, 77.488580, 76.319915, 75.193303, 74.091503, 72.999565, 71.895720, 70.810348, 69.768434, 68.748702, 67.729912, 66.703683, 65.694059, 64.727598, 63.785056, 62.834158, 61.880557, 60.946441, 60.045967, 59.168306, 58.266343, 57.341698, 57.341698};

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

    FLD_Solver FLD(1, 1e-7, 10000);
    FLD.set_precond_level(0);

    FLD.set_boundaries(BoundaryFlags::open_R_inner | 
                       BoundaryFlags::open_R_outer | 
                       BoundaryFlags::open_Z_outer) ;

    double tol=1;
    int n = 0;
    std::cout.setstate(std::ios_base::failbit);
    while (n<20 && tol>0.0001) {

        Field<double> oldT = create_field<double>(g);
        copy_field(g, T, oldT); 
        
        std::cout << "Iteration: " << n << "\n" ;  

        rho_kappa_abs_binned = bins.bin_planck(g, rho_kappa_abs, T);
        bin_central(g, rho_kappa_sca, rho_kappa_sca_binned, num_wavelengths, n_bands);

        binned_scattering = bins.bin_field(g, scattering, WavelengthBinner::SUM);


        double dt_inittemp = 0.;
        FLD.solve_multi_band(g, dt_inittemp, Cv, rho_kappa_abs_binned, rho_kappa_sca_binned, rhotot, heating, binned_scattering, bins.edges, T, J);

        tol = fracerr(g, oldT, T);
        std::cout << tol << "\n";
        n++;
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

    // for (int i=0; i<g.NR + 2*g.Nghost; i++) {
    //     printf("%1.6f, ", T(i,1));
    // }
    // std::cout << "\n";
    // for (int i=0; i<g.NR + 2*g.Nghost; i++) {
    //     printf("%1.6f, ", T(i,60));
    // }
    // std::cout << "\n";
}
