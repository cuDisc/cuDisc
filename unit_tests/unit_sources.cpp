#include <iostream>
#include <cmath>
#include <sstream>
#include <fstream>
#include <string>
#include <chrono>

#include "dustdynamics.h"
#include "sources.h"
#include "cuda_array.h"
#include "grid.h"
#include "field.h"
#include "constants.h"
#include "gas1d.h"
#include "hydrostatic.h"
#include "file_io.h"
#include "errorfuncs.h"

double rhobench[] = {3.38824e-13, 4.78601e-13, 6.76042e-13, 9.54936e-13, 1.34888e-12, 1.90535e-12, 2.69139e-12, 3.80172e-12, 5.37015e-12, 7.58575e-12, 1.07158e-11, 1.51382e-11, 2.13881e-11, 3.02251e-11, 4.27478e-11, 6.06822e-11, 2.01842e-11, 2.24911e-18, 2.24911e-18, 2.24911e-18};
double vRbench[] = {-9.62968e-06, -1.92137e-05, -3.83364e-05, -7.64913e-05, -0.00015262, -0.000304517, -0.000607591, -0.0012123, -0.00241886, -0.00482627, -0.00962967, -0.0192137, -0.0383364, -0.0764912, -0.198354, -0.78966, -3.14368, 0, 0, 0};

void set_up_gas(Grid& g, CudaArray<double>& Sig_g, CudaArray<double>& nu, Field<double>& T, Field<double>& cs, Field<double>& cs2, double alpha, Star& star) {
  
    double r_c = 30*au;
    double mu = 2.4;
    double Mtot = 0.;
    double Mdisc = 0.07*Msun;

    for (int i=0; i<g.NR+2*g.Nghost; i++) {

        Sig_g[i] =  std::pow(g.Rc(i)/r_c, -1.) * std::exp(-g.Rc(i)/r_c);
        Mtot += M_PI*Sig_g[i]*(g.Re(i+1)*g.Re(i+1) - g.Re(i)*g.Re(i));
        for (int j=0; j<g.Nphi+2*g.Nghost; j++) {
            T(i,j) = std::pow(6.25e-3 * star.L / (M_PI * g.Rc(i)*g.Rc(i) * sigma_SB), 0.25);
            cs(i,j) = std::sqrt(k_B*T(i,j) / (mu*m_H));
            cs2(i,j) = k_B*T(i,j) / (mu*m_H);
            nu[i] = alpha * cs(i,j) * cs(i,j) / std::sqrt(star.GM/std::pow(g.Rc(i), 3.));
        }
    
    }
    
    for (int i=0; i<g.NR+2*g.Nghost; i++) {
        Sig_g[i] *= Mdisc/Mtot + 1e-30;
    }

}

void set_up_dust(Grid& g, Field3D<Prims>& qd, Field<Prims>& wg, Field3D<double>& D, SizeGrid& sizes, double alpha, Field<double>& cs, double floor, double gfloor, double Mstar) {

    double d_to_g = 0.01;
    double Sc = 1.;

    for (int i=0; i<g.NR+2*g.Nghost; i++) {

        for (int j=0; j<g.Nphi+2*g.Nghost; j++) {
            double rho_tot = 0;
            for (int k=0; k<qd.Nd; k++) {
                // Initialise dust with MRN profile and exponential cut off at 0.1 micron
                qd(i,j,k).rho = std::pow(sizes.centre_size(k)/sizes.centre_size(0), 0.5) * std::exp(-std::pow(sizes.centre_size(k)/1., 10.));
                D(i,j,k) = wg(i,j).rho * (alpha * cs(i,j) * cs(i,j) / std::sqrt(GMsun/std::pow(g.Rc(i), 3.))) / Sc ;
                rho_tot += qd(i,j,k).rho;
            }
            for (int k=0; k<qd.Nd; k++) {
                if (wg(i,j).rho <= 1.1*gfloor) {
                    qd(i,j,k).rho = 0.1*wg(i,j).rho*floor;
                }
                else {
                    qd(i,j,k).rho *= d_to_g*wg(i,j).rho/rho_tot ;
                }
            }
        }
    }
    
    for (int i=0; i<g.NR+2*g.Nghost; i++) {
        for (int j=0; j<g.Nphi+2*g.Nghost; j++) {

            double vk = std::sqrt(GMsun*Mstar/g.Rc(i));

            for (int k=0; k < sizes.size(); k++) {
                qd(i,j,k).rho = std::max(qd(i,j,k).rho, 0.1*floor*wg(i,j).rho);

                // Set initial dust velocity to Keplerian orbit
              
                qd(i,j,k).v_R   = 0.;
                qd(i,j,k).v_phi = vk;
                qd(i,j,k).v_Z   = 0.;

            }
        }
    }

}

void compute_cs2(const Grid &g, Field<double> &T, Field<double> &cs2, double mu) {

    // Calculates square of the sound speed

    for (int i=0; i<g.NR + 2*g.Nghost; i++) {
        for (int j=0; j<g.Nphi + 2*g.Nghost; j++) {
            cs2(i,j) = R_gas * T(i,j) / mu;
        }
    }
}

void compute_nu(const Grid &g, CudaArray<double> &nu, Field<double> &cs2, double Mstar, double alpha) {
    for (int i=0; i<g.NR + 2*g.Nghost; i++) {
        double Om = std::sqrt(GMsun * Mstar / (g.Rc(i)*g.Rc(i)*g.Rc(i)));
        nu[i] = alpha * cs2(i,2) / Om;
    }
}

void compute_D(const Grid &g, Field3D<double> &D, Field<Prims> &wg, Field<double> &cs2, double Mstar, double alpha, double Sc) {

    // Calculates the dust diffusion constant

    for (int i=0; i<g.NR + 2*g.Nghost; i++) {
        double Om = std::sqrt(GMsun * Mstar / (g.Rc(i)*g.Rc(i)*g.Rc(i)));
        for (int j=0; j<g.Nphi + 2*g.Nghost; j++) {
            for (int k=0; k<D.Nd; k++) {
                D(i,j,k) = wg(i,j).rho * alpha * cs2(i,j) / (Sc*Om) ;
            }
        }
    }
}

void compute_nu(const Grid &g, CudaArray<double> &nu, double nu0, double Mstar, double alpha) {
    for (int i=0; i<g.NR + 2*g.Nghost; i++) {
        nu[i] = alpha * nu0 * (g.Rc(i)/au) / std::sqrt(Mstar);
    }
}

void cs2_to_cs(Grid& g, Field<double> &cs, Field<double> &cs2) {
    for (int i=0; i<g.NR + 2*g.Nghost; i++) {
        for (int j=0; j<g.Nphi + 2*g.Nghost; j++) {
            cs(i,j) = std::sqrt(cs2(i,j));
        }
    }
}

int main() {

    std::cout << "Test 2D sources... ";
    std::cout.flush() ;

    // Set up spatial grid 

    Grid::params p;
    p.NR = 20;
    p.Nphi = 100;
    p.Nghost = 2;

    p.Rmin = 1.*au;
    p.Rmax = 5.*au;
    p.theta_min = 0. ;
    p.theta_power = 0.75;
    p.theta_max = M_PI/6.;

    p.R_spacing = RadialSpacing::log ;
    p.theta_spacing = ThetaSpacing::power;

    Grid g(p);

    // Setup a size distribution

    double rho_p = 1.6;
    double a0 = 1e-5 ; // Grain size lower bound in cm
    double a1 = 10.   ;  // Grain size upper bound in cm
    int n_spec = 20;

    SizeGrid sizes(a0, a1, n_spec, rho_p) ;

    // Disc & Star parameters
    
    double mu = 2.4, M_star = 1., alpha = 1.e-3, T_star=4500., R_star = 1.7*Rsun;
    double L_star = 4.*M_PI*sigma_SB*std::pow(T_star, 4.)*std::pow(R_star, 2.);

    // Create star

    Star star(GMsun*M_star, L_star, T_star);

    // Create gas and dust fields

    Field3D<Prims> Ws_d = create_field3D<Prims>(g, n_spec); // Dust quantities 
    Field<Prims> Ws_g = create_field<Prims>(g); // Gas primitives
    CudaArray<double> Sig_g = make_CudaArray<double>(g.NR+2*g.Nghost); // Gas surface density
    CudaArray<double> nu = make_CudaArray<double>(g.NR+2*g.Nghost); // Kinematic viscosity
    Field<double> T = create_field<double>(g); // Temperature
    Field<double> cs = create_field<double>(g); // Sound speed
    Field<double> cs2 = create_field<double>(g); // Sound speed squared
    Field<double> alpha2D = create_field<double>(g); // alpha 2D
    Field3D<double> D = create_field3D<double>(g, n_spec); // Dust diffusion constant 

    // Set up initial dust and gas variables

    set_up_gas(g, Sig_g, nu, T, cs, cs2, alpha, star);

    int gas_boundary = BoundaryFlags::open_R_inner | BoundaryFlags::open_R_outer | BoundaryFlags::open_Z_outer;
    double gas_floor = 1e-100;
    double floor = 1.e-10;

    compute_hydrostatic_equilibrium(star, g, Ws_g, cs2, Sig_g, gas_floor);
    calc_gas_velocities(g, Sig_g, Ws_g, cs2, nu, alpha, star, gas_boundary, gas_floor);   
    for (int i=0; i<g.NR + 2*g.Nghost; i++) {
        for (int j=0; j<g.Nphi + 2*g.Nghost; j++) {
            alpha2D(i,j) = alpha;
        }
    }

    set_up_dust(g, Ws_d, Ws_g, D, sizes, alpha, cs, floor, gas_floor, M_star);
    
    double t = 0, dt;

    // Initialise diffusion-advection solver

    Sources src(T, Ws_g, sizes, floor, M_star, mu);
    DustDynamics dyn(D, cs, src, 0.4, 0.2, floor, gas_floor);

    // Set up boundary conditions

    int boundary = BoundaryFlags::open_R_inner | BoundaryFlags::open_R_outer | BoundaryFlags::open_Z_outer;

    dyn.set_boundaries(boundary);

    int count = 0;

    double dt_CFL = 100.;

    // Main timestep iteration

    while (t < 10*year) {

        dt = dt_CFL; // Set time-step according to CFL condition or proximity to selected time snapshots
        
        dyn(g, Ws_d, Ws_g, dt); // Diffusion-advection update

        // Gas updates

        count += 1;
        t += dt;

        if (count < 1000) {
            dt_CFL = std::min(dyn.get_CFL_limit(g, Ws_d, Ws_g), 1.1*dt); // Calculate new CFL condition time-step 
        }
        else {
            dt_CFL = dyn.get_CFL_limit(g, Ws_d, Ws_g);
        }
    }

    double L2 = 0;
    for (int k=0; k<sizes.size(); k++) {
        L2 += (std::pow(Ws_d(2,2,k).rho-rhobench[k], 2.)/sizes.size());
        L2 += (std::pow(Ws_d(2,2,k).v_R-vRbench[k], 2.)/sizes.size());
    }
    L2 = std::sqrt(L2);
    if (L2 <= 1.e-6) {printf("Pass.\n");}
    else {printf("\n\tL2 = %g, fail.\n", L2);}
    return 0;
} 
