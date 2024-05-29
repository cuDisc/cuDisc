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

double Sig_bench[100] = {0.001897035681, 0.001070227601, 0.0008884412249, 0.0008378119723, 0.0007927426385, 0.0007572934984, 0.0007307442676, 
                        0.0007271452571, 0.0007150952201, 0.0007091749165, 0.0007070004654, 0.000707525382, 0.0007102065419, 0.000714631457, 
                        0.0007206427145, 0.0007278929399, 0.0007362729952, 0.0007456678349, 0.0007559803298, 0.0007671189718, 0.0007790127982, 
                        0.000791615233, 0.0008048554397, 0.0008186757508, 0.0008330173347, 0.0008478140651, 0.0008629938335, 0.0008784785185, 
                        0.0008941757108, 0.0009099638524, 0.0009257352935, 0.0009413278091, 0.0009565775125, 0.0009712907917, 0.0009852320781, 
                        0.0009981355348, 0.001009707837, 0.00101960656, 0.00102746746, 0.001032861944, 0.001035335911, 0.001034404789, 
                        0.001029517574, 0.001020148527, 0.001005585322, 0.0009493085827, 0.0008038984786, 0.0007841608907, 0.0008221655496, 
                        0.0008828848535, 0.0009506695935, 0.001021305504, 0.001093009275, 0.001165853525, 0.001241529846, 0.001322918018, 
                        0.00141095008, 0.001508360172, 0.001616162939, 0.001737515768, 0.00187287469, 0.002025531353, 0.002190179254, 
                        0.002371047806, 0.002566851289, 0.002775088356, 0.002986620909, 0.003185013158, 0.003342471128, 0.00342227187, 
                        0.003383413473, 0.003190829522, 0.002834835052, 0.002336746749, 0.001775509786, 0.001206740936, 0.0007433139464, 
                        0.0003879576061, 0.0001716662164, 6.375430573e-05, 1.895986709e-05, 4.471770959e-06, 8.342022391e-07, 1.196559244e-07, 
                        1.339618936e-08, 1.203524393e-09, 8.687795738e-11, 5.172259039e-12, 2.613834924e-13, 1.092425088e-14, 3.847282324e-16, 
                        1.144183139e-17, 2.918001183e-19, 6.237774373e-21, 1.131017646e-22, 1.7671532e-24, 2.318191473e-26, 2.586961717e-28, 
                        2.482434943e-30, 1e-39};


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
    BS32Integration<CoagulationRate<BirnstielKernelVertInt, SimpleErosion>>
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

    if (L2 <= 2.e-13) {printf("Pass.\n");}
    else {printf("\n\tL2 = %g, fail.\n", L2);}

    return 0;
} 