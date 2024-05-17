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

#include "coagulation/coagulation.h"
#include "coagulation/integration.h"
#include "coagulation/fragments.h"


/*
Dynamics + Coag for a primordial disc with vertically isothermal temperature profile 
*/

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
                qd(i,j,k).rho = std::pow(sizes.centre_size(k)/sizes.centre_size(0), 0.5) * std::exp(-std::pow(sizes.centre_size(k)/1e-5, 10.));
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

    std::filesystem::path dir = std::string("./codes/outputs/isoPD");
    std::filesystem::create_directories(dir);

    // Set up spatial grid 

    Grid::params p;
    p.NR = 100;
    p.Nphi = 100;
    p.Nghost = 2;

    p.Rmin = 5.*au;
    p.Rmax = 500.*au;
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
    int n_spec = 7.*3.*std::log10(a1/a0) + 1;
    double v_frag = 100.; // Fragmentation threshold

    std::cout << "Number of dust species: "<< n_spec << "\n";
    SizeGrid sizes(a0, a1, n_spec, rho_p) ;

    write_grids(dir, &g, &sizes); // Write grids to file

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
    Field<double> J = create_field<double>(g); // Dummy J
    Field<double> cs = create_field<double>(g); // Sound speed
    Field<double> cs2 = create_field<double>(g); // Sound speed squared
    Field<double> alpha2D = create_field<double>(g); // alpha 2D
    Field3D<double> D = create_field3D<double>(g, n_spec); // Dust diffusion constant 

    // Set up initial dust and gas variables

    set_up_gas(g, Sig_g, nu, T, cs, cs2, alpha, star);

    double M_gas=0, M_dust=0;

    for (int i=g.Nghost; i<g.NR+g.Nghost; i++ ) { M_gas += Sig_g[i]*2.*M_PI*g.Rc(i)*g.dRe(i);}
    std::cout << "Initial gas mass: " << M_gas/Msun << " M_sun\n";
        
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

    for (int i=g.Nghost; i<g.NR + g.Nghost; i++) {
        for (int j=g.Nghost; j<g.Nphi + g.Nghost; j++) { 
            for (int k=0; k<Ws_d.Nd; k++) {
                M_dust += 4.*M_PI * Ws_d(i,j,k).rho * g.volume(i,j); // 4pi comes from 2pi in azimuth and 2 for symmetry about midplane
            }
        }
    }

    // Set up coagulation kernel, storing the fragmentation velocity

    BirnstielKernel kernel = BirnstielKernel(g, sizes, Ws_d, Ws_g, cs, alpha2D, mu);
    kernel.set_fragmentation_threshold(v_frag);

    // Setup the integrator
    BS32Integration<CoagulationRate<BirnstielKernel, SimpleErosion>>
        coagulation_integrate(
            create_coagulation_rate(
                sizes, 
                kernel, 
                SimpleErosion(1,11/6.,sizes.min_mass())), 
            1e-2, 1e-10
        ) ;

    std::cout << "Initial dust mass: " << M_dust/Msun << " M_sun\n";

    // Choose times to store data
    
    double t = 0, dt;
    const int ntimes = 4;  
    double ts[ntimes] = {10*year, 100*year, 1000*year, 1e4*year};

    std::ofstream f_times((dir / "2Dtimes.txt"));
    f_times << 0. << "\n";
    for (int i=0; i<ntimes; i++) {
        f_times << ts[i] << "\n";
    } 
    f_times.close();

    // Initialise diffusion-advection solver

    Sources src(T, Ws_g, sizes, floor, M_star, mu);
    DustDynamics dyn(D, cs, src, 0.4, 0.2, floor, gas_floor);

    double dt_CFL = dyn.get_CFL_limit(g, Ws_d, Ws_g);

    std::cout << dt_CFL << "\n";

    // Set up boundary conditions

    int boundary = BoundaryFlags::open_R_inner | BoundaryFlags::open_R_outer | BoundaryFlags::open_Z_outer;

    dyn.set_boundaries(boundary);

    std::chrono::_V2::system_clock::time_point start,stop;
    start = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds duration;
    double yps;
    int count = 0;
    double t_coag = 0, dt_coag = 0, t_temp = 0, dt_1perc = year;

    dt_CFL = 1;

    int Nout = 1;

    double dummy = 0;

    std::ifstream f(dir / ("restart_params.dat"), std::ios::binary);
    double t_restart=0;

    if (f) {

        // This block is used for reading in restart configurations if running on a cluster that requires restarting the code
        
        read_restart_file(dir / ("restart_params.dat"), count, t, dt_CFL, t_coag, t_temp, dt_coag, dt_1perc, dummy);

        std::cout << "Restart params: " << count << " " << t/year << " " << dt_CFL/year << "\n";

        read_restart_prims(dir, Ws_d, Ws_g, Sig_g);

        compute_cs2(g,T,cs2,mu);
        cs2_to_cs(g, cs, cs2);
        compute_D(g, D, Ws_g, cs2, M_star, alpha, 1.);
        compute_nu(g, nu, cs2, M_star, alpha);
        t_restart = t;
    }
    else {
        compute_nu(g, nu, cs2, M_star, alpha);
        compute_D(g, D, Ws_g, cs2, M_star, alpha, 1.);
        write_prims(dir, 0, g, Ws_d, Ws_g, Sig_g);
        write_temp(dir, 0, g, T) ; 
    }


    // Main timestep iteration

    for (double ti : ts) {

        if (t > ti) {
            Nout += 1;
            continue;
        }

        while (t < ti) {    

            if (!(count%1000)) {
                std::cout << "t = " << t/year << " years\n";
                std::cout << "dt = " <<dt_CFL/year << " years\n";
                stop = std::chrono::high_resolution_clock::now();
                yps = ((t-t_restart)/year) / std::chrono::duration_cast<std::chrono::seconds>(stop - start).count();
                std::cout << "Years per second: " << yps << "\n";
            }

            dt = std::min(dt_CFL, ti-t); // Set time-step according to CFL condition or proximity to selected time snapshots
            
            dyn(g, Ws_d, Ws_g, dt); // Diffusion-advection update

            // Gas updates

            update_gas_sigma(g, Sig_g, dt, nu, gas_boundary, gas_floor);
            compute_hydrostatic_equilibrium(star, g, Ws_g, cs2, Sig_g, Ws_d, gas_floor);
            calc_gas_velocities(g, Sig_g, Ws_g, cs2, nu, alpha, star, gas_boundary, gas_floor);  
            compute_D(g, D, Ws_g, cs2, M_star, alpha, 1.);

            // Coagulation update when 1 internal coagulation time-step has passed in the global simulation time

            if ((t+dt >= t_coag+dt_coag)|| (t+2*dt >= t_coag+dt_coag && dt < dt_coag) || dt == ti-t) {
                std::cout << "Coag step at count = " << count << "\n";
                // Reset coagulation kernel with updated quantities
                kernel = BirnstielKernel(g, sizes, Ws_d, Ws_g, cs, alpha2D, mu);
                kernel.set_fragmentation_threshold(v_frag);
                coagulation_integrate.set_kernel(kernel);
                // Run coagulation internal integration (routine calculates its own sub-steps to integrate over the timestep passed into it)
                coagulation_integrate.integrate(g, Ws_d, Ws_g, (t+dt)-t_coag, dt_coag, floor) ;
                t_coag = t+dt;
            } 

            count += 1;
            t += dt;

            if (count < 1000) {
                dt_CFL = std::min(dyn.get_CFL_limit(g, Ws_d, Ws_g), 1.025*dt); // Calculate new CFL condition time-step 
            }
            else {
                dt_CFL = dyn.get_CFL_limit(g, Ws_d, Ws_g);
            }
                
            // Uncomment this section for writing restart files for jobs on clusters that need to be re-batched after a certain amount of time; here a restart file is written after 20 hrs
            // if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count()/3600. > 20.) {
            //     std::cout << "Writing restart at t = " << t/year << " years.\n" ;
            //     write_restart_file(dir / ("restart_params.dat"), count, t, dt_CFL, t_coag, t_temp, dt_coag, dt_1perc, dummy);
            //     write_restart_prims(dir, g, Ws_d, Ws_g, Sig_g);  
            //     return 0;
            // } 

        }

        // Record densities to file at time snapshots

        write_prims(dir, Nout, g, Ws_d, Ws_g, Sig_g);  
        //write_temp(dir, Nout, g, T) ; Skip because constant
        Nout+=1;
    }
    
    // This is used for telling your job submission script that the final snapshot has been reached, meaning no more restarts are necessary
    std::ofstream fin(dir / ("finished"));
    fin.close();

    stop = std::chrono::high_resolution_clock::now();
    std::cout << count << " timesteps\n" ;
    
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time taken: " << duration.count()/(1.e6*60.) << " mins" << std::endl;  
    return 0;
} 
