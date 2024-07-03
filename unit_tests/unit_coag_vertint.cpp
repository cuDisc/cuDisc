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


void set_up(Grid& g, Field<Prims1D>& wg, Field<double>& cs, Field3D<Prims1D>& qd, SizeGrid& sizes) {


    for (int i=0; i<g.NR+2*g.Nghost; i++) {
        for (int j=0; j<g.Nphi+2*g.Nghost; j++) {

            wg(i,j).Sig = 10.;
            wg(i,j).v_R = 0.;
        
            cs(i,j) = 35000;
        }
    }

    for (int i=0; i<g.NR+2*g.Nghost; i++) {
        for (int j=0; j<g.Nphi+2*g.Nghost; j++) {
            for (int k=0; k<qd.Nd; k++) {    
                qd(i,j,k).Sig = std::pow(sizes.centre_size(k)/sizes.centre_size(0), 0.5) * std::exp(-std::pow(sizes.centre_size(k)/0.02, 10.));
            }
        }
    }
    double Sig_dust=0;
    for (int k=0; k<qd.Nd; k++) {
        Sig_dust += qd(2,2,k).Sig;
    }
    
    for (int i=0; i<g.NR+2*g.Nghost; i++) {
        for (int j=0; j<g.Nphi+2*g.Nghost; j++) {
            for (int k=0; k < sizes.size(); k++) {
                qd(i,j,k).Sig *= 0.01*wg(i,j).Sig/Sig_dust + 1.e-40;
                qd(i,j,k).v_R = 0.;
            }
        }
    }
}

double Sig_bench[100] = {0.00188614, 0.00106862, 0.000888368, 0.000839277, 0.000794924, 0.000759651, 0.000733208, 0.00072963, 0.00071768, 0.000711879, 0.000709859, 0.000710564, 0.000713451, 0.000718108, 0.00072438, 0.000731921, 0.000740624, 0.000750379, 0.000761092, 0.000772676, 0.000785068, 0.000798226, 0.000812089, 0.000826608, 0.000841734, 0.000857415, 0.00087359, 0.000890197, 0.000907161, 0.000924378, 0.000941762, 0.000959171, 0.000976466, 0.000993478, 0.00101, 0.00102578, 0.00104056, 0.00105402, 0.00106581, 0.00107551, 0.00108269, 0.00108686, 0.00108748, 0.00108403, 0.00107585, 0.0010624, 0.00104312, 0.000891653, 0.000816768, 0.000828595, 0.000880748, 0.000946277, 0.00101713, 0.00109186, 0.00116893, 0.00125061, 0.00133831, 0.00143523, 0.00154281, 0.00166444, 0.00180085, 0.00195538, 0.00212281, 0.0023071, 0.00250694, 0.00271994, 0.00293715, 0.00314276, 0.003309, 0.0034006, 0.00337374, 0.00319731, 0.00285117, 0.00236651, 0.00180574, 0.00123866, 0.000767325, 0.000404916, 0.000180594, 6.79211e-05, 2.04038e-05, 4.877e-06, 9.19761e-07, 1.33553e-07, 1.50837e-08, 1.3669e-09, 9.91855e-11, 5.93512e-12, 3.00947e-13, 1.26211e-14, 4.45631e-16, 1.3287e-17, 3.39607e-19, 7.27415e-21, 1.32119e-22, 2.06779e-24, 2.71638e-26, 3.03511e-28, 2.91614e-30, 1e-39};


int main() {


    std::filesystem::path dir = std::string("./codes/outputs/unit_coag_vertint");
    std::filesystem::create_directories(dir);

    std::cout << "Test coag vertically-integrated... ";
    std::cout.flush() ;
    
    Grid::params p;
    p.NR = 1;
    p.Nphi = 1;
    p.Nghost = 2;

    p.Rmin = 19.*au;
    p.Rmax = 21.*au;
    p.theta_min = -0.001 ;
    p.theta_max = 0.001;

    p.R_spacing = RadialSpacing::log ;

    Grid g(p);

    // Setup a size distribution

    int n_spec = 100;
    double rho_p = 1.6;
    double a0 = 5e-5 ; // Grain size lower bound in cm
    double a1 = 0.1   ;  // Grain size upper bound in cm
    SizeGrid sizes(a0, a1, n_spec, rho_p) ;

    write_grids(dir, &g, &sizes);

    // Disc & Star parameters
    
    double mu = 2.4, alpha = 1.e-3;

    // Create gas and dust fields (Qs = Quantities; object holds density and three-momenta, Ws = Primitives; object holds density and three-velocity)

    Field3D<Prims1D> Ws_d = create_field3D<Prims1D>(g, n_spec); // Dust quantities 
    Field<Prims1D> Ws_g = create_field<Prims1D>(g); // Gas primitives
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
    BirnstielKernelVertInt kernel = BirnstielKernelVertInt(g, sizes, Ws_d, Ws_g, cs, alpha2D, mu, 1.);
    kernel.set_fragmentation_threshold(100.);
    BS32Integration<CoagulationRate<decltype(kernel), SimpleErosion>>
        coagulation_integrate(
            create_coagulation_rate(
                sizes, 
                kernel, 
                SimpleErosion(1,11/6.,sizes.min_mass())), 
            1e-2, 1e-10
        ) ;

    double dt_coag = 0;

    std::cout.setstate(std::ios_base::failbit);
    coagulation_integrate.integrate(g, Ws_d, Ws_g, 1e5*year, dt_coag, 1e-40) ;
    std::cout.clear();

    double L2 = 0;
    for (int i=0; i<sizes.size(); i++) {
        L2 += (std::pow(Ws_d(2,2,i).Sig-Sig_bench[i], 2.)/(sizes.size()));
    }
    L2 = std::sqrt(L2);
    if (L2 <= 2.e-9) {printf("Pass.\n");}
    else {printf("\n\tL2 = %g, fail.\n", L2);}

    return 0;
} 