#include <iostream>
#include <cmath>
#include <sstream>
#include <fstream>
#include <string>
#include <iomanip>

#include "cuda_array.h"
#include "grid.h"
#include "field.h"
#include "constants.h"
#include "file_io.h"

#include "coagulation/coagulation.h"
#include "coagulation/integration.h"


void set_up(Grid& g, Field<Prims>& wg, Field<double>& cs, Field3D<Prims>& qd, SizeGrid& sizes) {

    double M_star = 1.;

    for (int i=0; i<g.NR+2*g.Nghost; i++) {
        for (int j=0; j<g.Nphi+2*g.Nghost; j++) {

            wg(i,j).rho = 5e-13;
            wg(i,j).v_R = 0.;
            wg(i,j).v_phi = g.Rc(i) * std::pow(GMsun*M_star/std::pow(g.Rc(i)*g.Rc(i)+g.Zc(i,j)*g.Zc(i,j),1.5), 0.5) * std::pow(0.9996, 0.5);
            wg(i,j).v_Z = 0.;
        
            cs(i,j) = 35000;
        }
    }

    for (int i=0; i<g.NR+2*g.Nghost; i++) {
        for (int j=0; j<g.Nphi+2*g.Nghost; j++) {
            for (int k=0; k<qd.Nd; k++) {    
                qd(i,j,k).rho = std::pow(sizes.centre_size(k)/sizes.centre_size(0), 0.5) * std::exp(-std::pow(sizes.centre_size(k)/0.02, 10.));
            }
        }
    }
    double rho_dust=0;
    for (int i=0; i<g.NR+2*g.Nghost; i++) {
        for (int k=0; k<qd.Nd; k++) {
            rho_dust += qd(i,2,k).rho;
        }
    }
    
    for (int i=0; i<g.NR+2*g.Nghost; i++) {
        for (int j=0; j<g.Nphi+2*g.Nghost; j++) {
            
            double vk = std::sqrt(GMsun*M_star/g.Rc(i));

            for (int k=0; k < sizes.size(); k++) {
                qd(i,j,k).rho = qd(i,j,k).rho * 0.01*wg(i,j).rho/rho_dust + 1.e-40;
                qd(i,j,k).v_R  =  0.;
                qd(i,j,k).v_phi = vk;
                qd(i,j,k).v_Z   = 0;
            }
        }
    }
}

double rho_bench[100] = {1.01172e-17, 6.14468e-18, 5.22943e-18, 5.07278e-18, 4.89298e-18, 4.70835e-18, 4.5615e-18, 
                        4.518e-18, 4.44084e-18, 4.39954e-18, 4.38231e-18, 4.38283e-18, 4.39778e-18, 4.42472e-18, 
                        4.46224e-18, 4.50836e-18, 4.56224e-18, 4.62307e-18, 4.69022e-18, 4.7631e-18, 4.84128e-18, 
                        4.92456e-18, 5.01257e-18, 5.10507e-18, 5.20187e-18, 5.30274e-18, 5.40748e-18, 5.51587e-18, 
                        5.62772e-18, 5.74263e-18, 5.86045e-18, 5.98071e-18, 6.10302e-18, 6.22693e-18, 6.35182e-18, 
                        6.47698e-18, 6.60166e-18, 6.7249e-18, 6.84571e-18, 6.9628e-18, 7.07479e-18, 7.18022e-18, 
                        7.27715e-18, 7.36397e-18, 7.43798e-18, 7.49713e-18, 7.53875e-18, 7.56056e-18, 7.55935e-18, 
                        7.53222e-18, 7.47604e-18, 6.9193e-18, 5.92419e-18, 6.03152e-18, 6.53026e-18, 7.16145e-18, 
                        7.82226e-18, 8.49126e-18, 9.15495e-18, 9.81558e-18, 1.04826e-17, 1.11829e-17, 1.1895e-17, 
                        1.26529e-17, 1.34689e-17, 1.43669e-17, 1.53614e-17, 1.64751e-17, 1.77189e-17, 1.91186e-17, 
                        2.06978e-17, 2.24537e-17, 2.43725e-17, 2.62347e-17, 2.8712e-17, 3.11464e-17, 3.44122e-17, 
                        3.6666e-17, 3.84513e-17, 3.99633e-17, 3.9531e-17, 3.73046e-17, 3.35239e-17, 2.76661e-17, 
                        2.0809e-17, 1.42185e-17, 8.46591e-18, 4.35038e-18, 1.91013e-18, 6.84909e-19, 1.98854e-19, 
                        4.59633e-20, 8.44969e-21, 1.21433e-21, 1.38931e-22, 1.30316e-23, 1.00176e-24, 6.45616e-26, 
                        3.44367e-27, 5e-53};


int main() {

    std::cout << "Test coag... ";
    std::cout.flush() ;
    
    Grid::params p;
    p.NR = 1;
    p.Nphi = 1;
    p.Nghost = 2;

    p.Rmin = 19.*au;
    p.Rmax = 21.*au;
    p.theta_min = 0. ;
    p.theta_power = 0.75;
    p.theta_max = 0.001;

    p.R_spacing = RadialSpacing::log ;
    p.theta_spacing = ThetaSpacing::power;

    Grid g(p);

    // Setup a size distribution

    int n_spec = 100;
    double rho_p = 1.6;
    double a0 = 5e-5 ; // Grain size lower bound in cm
    double a1 = 0.1   ;  // Grain size upper bound in cm
    SizeGrid sizes(a0, a1, n_spec, rho_p) ;

    // Disc & Star parameters
    
    double mu = 2.4, alpha = 1.e-3;

    // Create gas and dust fields (Qs = Quantities; object holds density and three-momenta, Ws = Primitives; object holds density and three-velocity)

    Field3D<Prims> Ws_d = create_field3D<Prims>(g, n_spec); // Dust quantities 
    Field<Prims> Ws_g = create_field<Prims>(g); // Gas primitives
    Field<double> T = create_field<double>(g); // Temperature
    Field<double> cs = create_field<double>(g); // Sound speed
    Field<double> alpha2D = create_field<double>(g);

    for (int i=0; i<g.NR+2*g.Nghost; i++) {
        for (int j=0; j<g.Nphi+2*g.Nghost; j++) {
            alpha2D(i,j) = alpha;
        }
    }

    set_up(g, Ws_g, cs, Ws_d, sizes);

    // Set up coagulation kernel
    BirnstielKernel kernel = BirnstielKernel(g, sizes, Ws_d, Ws_g, cs, alpha2D, mu, 1.);
    kernel.set_fragmentation_threshold(100.);
    BS32Integration<CoagulationRate<BirnstielKernel, SimpleErosion>>
        coagulation_integrate(
            create_coagulation_rate(
                sizes, 
                kernel, 
                SimpleErosion(1,11/6.,sizes.min_mass())), 
            1e-2, 1e-10
        ) ;

    double dt_coag = 0;

    std::cout.setstate(std::ios_base::failbit);
    coagulation_integrate.integrate(g, Ws_d, Ws_g, 5e5*year, dt_coag, 1e-40) ;
    std::cout.clear();

    double L2 = 0;
    for (int i=0; i<sizes.size(); i++) {
        L2 += (std::pow(Ws_d(2,2,i).rho-rho_bench[i], 2.)/(sizes.size()));
    }
    L2 = std::sqrt(L2);

    if (L2 <= 2.e-23) {printf("Pass.\n");}
    else {printf("\n\tL2 = %g, fail.\n", L2);}

    return 0;
} 