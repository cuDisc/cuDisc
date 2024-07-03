#include <iostream>
#include <cmath>
#include <sstream>
#include <fstream>
#include <string>
#include <chrono>
#include <iomanip>

#include "cuda_array.h"
#include "grid.h"
#include "field.h"
#include "constants.h"
#include "file_io.h"

#include "coagulation/coagulation.h"
#include "coagulation/integration.h"


/*
Dynamics + Coag + FLD for a dustpy comparison
*/


void set_up_gas(Grid& g, Field<Prims1D>& wg, CudaArray<double>& Sig_g, Field<double>& T, Field<double>& cs) {

    double M_star = 1.;
    double p = -2.25;
    double q = -0.5;

    for (int i=0; i<g.NR+2*g.Nghost; i++) {
        for (int j=0; j<g.Nphi+2*g.Nghost; j++) {

            double h_g = 1.048*au; 

            Sig_g[i] = 1.69779673e+001;
            double eta = - std::pow(h_g/g.Rc(i), 2) * std::exp(-std::pow(0.2/(g.Rc(i)/au),10)) * (p + q + ((q+3)/2)*std::pow(g.Zc(i,j)/h_g, 2) + 10*std::pow(0.2/(g.Rc(i)/au),10));

            wg(i,j).Sig = Sig_g[i];
            wg(i,j).v_R = 0.;
            wg(i,j).v_phi = g.Rc(i) * std::pow(GMsun*M_star/std::pow(g.Rc(i)*g.Rc(i)+g.Zc(i,j)*g.Zc(i,j),1.5), 0.5) * std::pow(1 - eta, 0.5);
            wg(i,j).v_Z = 0.;
        
            T(i,j) = 35.59275531 ;
            cs(i,j) = 34987.90450412;
        }
    }

}
void set_up_dust(Grid& g, Field3D<Prims1D>& qd, SizeGrid& sizes, double M_gas) {

    double d_to_g = 0.01;
    double M_star = 1.;
    double M_dust=0;

    for (int i=0; i<g.NR+2*g.Nghost; i++) {
        for (int j=0; j<g.Nphi+2*g.Nghost; j++) {
            for (int k=0; k<qd.Nd; k++) {    
                qd(i,j,k).Sig = std::pow(sizes.centre_size(k)/sizes.centre_size(0), 0.5) * std::exp(-std::pow(sizes.centre_size(k)/0.02, 10.));
            }
        }
    }
    for (int i=0; i<g.NR+2*g.Nghost; i++) {
        for (int k=0; k<qd.Nd; k++) {
            M_dust += 2.*M_PI * qd(i,2,k).Sig * g.Rc(i) * g.dRe(i);
        }
    }
    
    for (int i=0; i<g.NR+2*g.Nghost; i++) {
        for (int j=0; j<g.Nphi+2*g.Nghost; j++) {
            
            double vk = std::sqrt(GMsun*M_star/g.Rc(i));

            for (int k=0; k < sizes.size(); k++) {
                qd(i,j,k).Sig = qd(i,j,k).Sig * d_to_g*M_gas/M_dust + 1.e-40;
                qd(i,j,k).v_R  =  0.;
                qd(i,j,k).v_phi = vk;
                qd(i,j,k).v_Z   = 0;
            }
        }
    }

}

int main() {

    std::filesystem::path path = __FILE__;
    path = (path.parent_path()).parent_path();
    std::filesystem::path dir = path / std::string("outputs/coag_dustpy");
    std::filesystem::create_directories(dir);

    std::cout << "Output directory: " << dir  << "\n";

    // Set up spatial grid 

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

    int n_spec = 200;
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
    CudaArray<double> Sig_g = make_CudaArray<double>(g.NR+2*g.Nghost); // Gas surface density
    Field<double> T = create_field<double>(g); // Temperature
    Field<double> cs = create_field<double>(g); // Sound speed
    Field<double> cs2 = create_field<double>(g); // Sound speed squared
    Field<double> alpha2D = create_field<double>(g);

    for (int i=0; i<g.NR+2*g.Nghost; i++) {
        for (int j=0; j<g.Nphi+2*g.Nghost; j++) {
            alpha2D(i,j) = alpha;
        }
    }

    // Set up initial dust and gas variables

    set_up_gas(g, Ws_g, Sig_g, T, cs);

    double M_gas=0;

    for (int i=0; i<g.NR+2*g.Nghost; i++ ) { M_gas += Sig_g[i]*2.*M_PI*g.Rc(i)*g.dRe(i);}
    std::cout << "Initial gas mass: " << M_gas/Msun << " M_sun\n";

    set_up_dust(g, Ws_d, sizes, M_gas);

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

    std::chrono::_V2::system_clock::time_point start,stop;
    start = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds duration;
    double dt_coag = 0;

    coagulation_integrate.integrate(g, Ws_d, Ws_g, 1e5*year, dt_coag, 1e-40) ;
    write_prims1D(dir, 0, g, Ws_d, Ws_g);

    stop = std::chrono::high_resolution_clock::now();
    
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time taken: " << duration.count()/(1.e6*60.) << " mins" << std::endl;  
    return 0;
} 