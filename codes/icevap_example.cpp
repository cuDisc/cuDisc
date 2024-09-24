#include <iostream>
#include <cmath>
#include <sstream>
#include <fstream>
#include <string>
#include <chrono>

#include "sources.h"
#include "cuda_array.h"
#include "grid.h"
#include "field.h"
#include "constants.h"
#include "gas1d.h"
#include "hydrostatic.h"
#include "stellar_irradiation.h"
#include "DSHARP_opacs.h"
#include "FLD.h"
#include "bins.h"
#include "file_io.h"
#include "errorfuncs.h"
#include "dustdynamics.h"

#include "coagulation/coagulation.h"
#include "coagulation/integration.h"
#include "coagulation/fragments.h"

#include "icevapour.h"


/*
Dynamics + Coag + FLD + Ice-Vapour chem for a steady state transition disc
*/


void setup_init_J(const Grid &g, Field<double> &heat, Field3D<double> &J) {

    // Sets initial radiative flux (J=cE_R where E_R is the radiative energy) for temperature calculations

    for (int i=0; i<g.NR + 2*g.Nghost; i++) {
        for (int j=0; j<g.Nphi + 2*g.Nghost; j++) {
            for (int k=0; k<J.Nd; k++) {
                J[J.index(i,j,k)] = heat[heat.index(i,j)]/J.Nd ; 
            }    
        }
    }

}

void set_up_gas(Grid& g, Field<Prims>& wg, CudaArray<double>& Sig_g, CudaArray<double>& nu, Field<double>& T, Field<double>& cs, Field<double>& cs2, double alpha, Star& star) {

    // Constant gas surface density and velocities set according to Takeuchi and Lin 2002 (https://iopscience.iop.org/article/10.1086/344437/pdf)

    double rho_0 = 18.e-10;
    double M_star = 1.;
    double h_0 = 3.33e-2*au;
    double p = -2.25;
    double q = -0.5;
    double mu = 2.4;

    for (int i=0; i<g.NR+2*g.Nghost; i++) {
        for (int j=0; j<g.Nphi+2*g.Nghost; j++) {

            double h_g = h_0 * std::pow(g.Rc(i)/au, (q+3)/2);

            // Sig_g[i] = std::sqrt(2.*M_PI) * rho_0*h_0*std::pow(g.Rc(i)/au, p+(q+3)/2.)*std::exp(-std::pow(5./(g.Rc(i)/au),10)) + 1.e-30;
            Sig_g[i] = std::sqrt(2.*M_PI) * rho_0*h_0*std::pow(g.Rc(i)/au, p+(q+3.)/2.)*std::exp(- (g.Rc(i)/(30*au))) + 1.e-30;
            
            double eta = - std::pow(h_g/g.Rc(i), 2) * (p + q + ((q+3)/2)*std::pow(g.Zc(i,j)/h_g, 2));

            wg(i,j).v_R = 0.;
            wg(i,j).v_phi = g.Rc(i) * std::pow(GMsun*M_star/std::pow(g.Rc(i)*g.Rc(i)+g.Zc(i,j)*g.Zc(i,j),1.5), 0.5) * std::pow(1 - eta, 0.5);
            wg(i,j).v_Z = 0.;
        
            // T(i,j) = mu*m_H/k_B * (GMsun*M_star/std::pow(g.Rc(i), 3)) * h_0*h_0 * std::pow(g.Rc(i)/au, q+3);
            T(i,j) = std::pow(6.25e-3 * star.L / (M_PI * g.Rc(i)*g.Rc(i) * sigma_SB), 0.25);
            cs(i,j) = std::sqrt(k_B*T(i,j) / (mu*m_H));
            cs2(i,j) = k_B*T(i,j) / (mu*m_H);
            nu[i] = alpha * cs(i,j) * cs(i,j) / std::sqrt(GMsun/std::pow(g.Rc(i), 3.));
        }
    
    }
    
}

void init_dust(Grid& g, Field3D<Prims>& qd, Field<Prims>& wg, CudaArray<double>& Sig_g, SizeGrid& sizes, Field<double>& cs, double alpha, double Mstar, double u_f, double gfloor) {

    CudaArray<double> P = make_CudaArray<double>(g.NR+2*g.Nghost);
    double dtg = 0.01;
    double mdusttot = 0;
    double Mdisc = 0.0402929*Msun;

    for (int i=0; i<g.NR+2*g.Nghost+1; i++) {
        
        double v_k = std::sqrt(GMsun*Mstar/g.Re(i));

        if (i==0) {
            P[i] = Sig_g[i] * cs(i,2) * v_k / (g.Re(i) * std::sqrt(2.*M_PI));
        }

        else if (i==g.NR+2*g.Nghost) {
            P[i] = Sig_g[i-1] * cs(i,2) * v_k / (g.Re(i) * std::sqrt(2.*M_PI));
        }

        else {
            double Sig_g_e = Sig_g[i] - (g.Rc(i)-g.Re(i))*(Sig_g[i]-Sig_g[i-1]) / g.dRc(i-1);

            P[i] = Sig_g_e * cs(i,2) * v_k / (g.Re(i) * std::sqrt(2.*M_PI));
        }
    
    }

    for (int i=0; i<g.NR+2*g.Nghost; i++) {
        double Omk = std::sqrt(GMsun*Mstar/(g.Rc(i)*g.Rc(i)*g.Rc(i)));
        double hg = cs(i,2)/Omk;
        double a_frag = std::max(1e-5, 1./(M_PI) * Sig_g[i]/(sizes.solid_density()*alpha) * std::pow(u_f/cs(i,2), 2.));

        double dlnPdlnR = (log(P[i+1]) - log(P[i])) / (log(g.Re(i+1)) - log(g.Re(i))); 
        double a_drift = std::max(1e-5, 0.55 * (2*(0.01*Sig_g[i])/(M_PI*sizes.solid_density())) * ((Omk*Omk*g.Rc(i)*g.Rc(i))/(cs(i,2)*cs(i,2))) * (1./std::abs(dlnPdlnR)));

        double eta = -(hg*hg/(g.Rc(i)*g.Rc(i))) * dlnPdlnR;

        for (int j=0; j<g.Nphi+2*g.Nghost; j++) {  
            double Sigtot = 0;  
            for (int k=0; k < sizes.size(); k++) {
  
                if (wg(i,j).rho < 10*gfloor) {

                    qd(i,j,k).rho = 1e-12*wg(i,j).rho;
                    qd(i,j,k).v_R   = 0.;
                    qd(i,j,k).v_phi = wg(i,j).v_phi;  
                    qd(i,j,k).v_Z = 0.;
                }
                else {
                    double St = sizes.solid_density() * sizes.centre_size(k) * (M_PI/2.) / Sig_g[i];
                    double hp = hg * std::sqrt(1/(1+St/alpha));

                    double Sigk = std::pow(sizes.centre_size(k)/sizes.centre_size(0), 0.5) * std::exp(-std::pow(sizes.centre_size(k)/std::min(a_frag,a_drift), 5.));

                    qd(i,j,k).rho = Sigk * (0.01*Sig_g[i])/(std::sqrt(2*M_PI)*hp) * std::exp(-(g.Zc(i,j)*g.Zc(i,j)/(2.*hp*hp)));
                    qd(i,j,k).v_R   = (wg(i,j).v_R - eta * Omk*g.Rc(i) * St / (1 + St*St)) ;
                    qd(i,j,k).v_phi = Omk*g.Rc(i); 
                    qd(i,j,k).v_Z = - Omk * St * g.Zc(i,j); 
                    Sigtot += Sigk;
                }

            }
            if (wg(i,j).rho >= 10*gfloor) {
                for (int k=0; k < sizes.size(); k++) {
                    qd(i,j,k).rho = qd(i,j,k).rho/Sigtot + 1e-12*wg(i,j).rho;
                }
            }
        }
    }
}

void set_up_dust(Grid& g, Field3D<Prims>& qd, Field<Prims>& wg, CudaArray<double>& Sig_g, Field3D<double>& D, SizeGrid& sizes, double alpha, Field<double>& cs, double M_gas, double floor) {

    double d_to_g = 0.01;
    double M_star = 1.;
    double h_0 = 3.33e-2*au;
    double p = -2.25;
    double q = -0.5;
    double Sc = 1.;
    double M_dust=0;

    for (int i=0; i<g.NR+2*g.Nghost; i++) {

        double h_g = cs(i,2)/std::sqrt(GMsun/(g.Rc(i)*g.Rc(i)*g.Rc(i)));
        for (int j=0; j<g.Nphi+2*g.Nghost; j++) {
            for (int k=0; k<qd.Nd; k++) {
                // Initialise dust with MRN profile and exponential cut off for large grains
                double Sig_d = std::pow(sizes.centre_size(k)/sizes.centre_size(0), 0.5) * std::exp(-std::pow(sizes.centre_size(k)/0.2, 10.)) * Sig_g[i] * std::exp(-g.Rc(i)/(10.*au));
                qd(i,j,k).rho = Sig_d/(std::sqrt(2.*M_PI)*h_g) * std::exp(-g.Zc(i,j)*g.Zc(i,j)/(2.*h_g*h_g));
                D(i,j,k) = wg(i,j).rho * (alpha * cs(i,j) * cs(i,j) / std::sqrt(GMsun/std::pow(g.Rc(i), 3.))) / Sc ;
            }
        }
    }
    for (int i=g.Nghost; i<g.NR + g.Nghost; i++) {
        for (int j=g.Nghost; j<g.Nphi + g.Nghost; j++) {
            for (int k=0; k<qd.Nd; k++) {
                M_dust += 4.*M_PI * qd(i,j,k).rho * g.volume(i,j);
            }
        }
    }
    
    for (int i=0; i<g.NR+2*g.Nghost; i++) {
        for (int j=0; j<g.Nphi+2*g.Nghost; j++) {

            double h_g = h_0 * std::pow(g.Rc(i)/au, (q+3)/2);
            double eta = - std::pow(h_g/g.Rc(i), 2) * std::exp(-std::pow(5./(g.Rc(i)/au),10)) * (p + q + ((q+3)/2)*std::pow(g.Zc(i,j)/h_g, 2) + 10*std::pow(5./(g.Rc(i)/au),10));
            double vk = std::sqrt(GMsun*M_star/g.Rc(i));
            double az = vk*vk * g.Zc(i,j) / (g.Rc(i)*g.Rc(i)) ;

            for (int k=0; k < sizes.size(); k++) {
                qd(i,j,k).rho = qd(i,j,k).rho * d_to_g*M_gas/M_dust ;
                qd(i,j,k).rho = std::max(qd(i,j,k).rho, floor*wg(i,j).rho);

                // Set initial dust velocities through standard drift velocity equations
                
                double St = (vk/g.Rc(i))*sizes.solid_density() * sizes.centre_size(k) / (cs(i,j) * wg(i,j).rho * std::sqrt(8./M_PI)) ;

                qd(i,j,k).v_R   = 0.;// (- eta * vk * St / (1 + St*St)) ;
                qd(i,j,k).v_phi = vk;
                qd(i,j,k).v_Z   = 0.;//- (vk/g.Rc(i)) * St * g.Zc(i,j);

                // if (qd(i,j,k).rho <= 1.e-40) {
                //     qd(i,j,k).v_R   = 0;
                //     qd(i,j,k).v_phi = 0;
                //     qd(i,j,k).v_Z = 0;
                // }
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
                
                // if (wg(i,j).rho <= 1.e-30) {
                //     D(i,j,k) = 1e-50 * D(i,j,k);
                // }
            }
        }
    }
}

void compute_nu(const Grid &g, CudaArray<double> &nu, double nu0, double Mstar, double alpha) {
    for (int i=0; i<g.NR + 2*g.Nghost; i++) {
        nu[i] = alpha * nu0 * (g.Rc(i)/au) / std::sqrt(Mstar);
    }
}

void compute_D(const Grid &g, Field3D<double> &D, Field<Prims> &wg, CudaArray<double> &nu, double Sc) {

    // Calculates the dust diffusion constant

    for (int i=0; i<g.NR + 2*g.Nghost; i++) {
        for (int j=0; j<g.Nphi + 2*g.Nghost; j++) {
            for (int k=0; k<D.Nd; k++) {
                D(i,j,k) = wg(i,j).rho * nu[i] / Sc ;
                // if (wg(i,j).rho <= 1.e-30) {
                //     D(i,j,k) = 1e-50 * D(i,j,k);
                // }
            }
        }
    }
}

void compute_total_density(Grid& g, Field<Prims>& w_g, Field3D<Prims>& w_d, Field<double>& rho_tot) {

    for (int i=0; i<g.NR + 2*g.Nghost; i++) {
        for (int j=0; j<g.Nphi + 2*g.Nghost; j++) {   
            double rho_tot_temp = 0.;
            for (int k=0; k<w_d.Nd; k++) {
                rho_tot_temp += w_d(i,j,k).rho;
            }    
            rho_tot(i,j) = w_g(i,j).rho + rho_tot_temp ;
        }
    }
}

void compute_total_density(Grid& g, Field<Prims>& w_g, Field3D<Prims>& w_d, Field<double>& rho_tot, Molecule& mol) {

    for (int i=0; i<g.NR + 2*g.Nghost; i++) {
        for (int j=0; j<g.Nphi + 2*g.Nghost; j++) {   
            double rho_tot_temp = 0.;
            for (int k=0; k<w_d.Nd; k++) {
                rho_tot_temp += w_d(i,j,k).rho + mol.ice(i,j,k);
            }    
            rho_tot(i,j) = w_g(i,j).rho + rho_tot_temp + mol.vap(i,j);
        }
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

    std::filesystem::path dir = std::string("./codes/outputs/icevap_example/0");
    std::filesystem::create_directories(dir);

    // Set up spatial grid 

    Grid::params p;
    p.NR = 250;
    p.Nphi = 200;
    p.Nghost = 2;

    p.Rmin = 5.*au;
    p.Rmax = 200.*au;
    p.theta_min = 0. ;
    p.theta_power = 0.75;
    p.theta_max = M_PI/4.;

    p.R_spacing = RadialSpacing::log ;
    p.theta_spacing = ThetaSpacing::power;

    Grid g(p);

    // Setup a size distribution

    int n_spec = 160;
    double rho_p = 1.7;
    double a0 = 1e-5 ; // Grain size lower bound in cm
    double a1 = 300.   ;  // Grain size upper bound in cm
    SizeGridIce sizes(g, a0, a1, n_spec, rho_p, 0.9) ;

    // Read in opacity table 

    DSHARP_opacs opac_tab("./codes/opacities/dustkappa_DSHARP500_wg.txt");

    // Interpolate opacities onto specified (grain size, wavelength) grid

    int num_wavelengths = 200; // This is the number of wavelengths used for stellar heating calculations

    DSHARP_opacs opacs(n_spec, num_wavelengths); // Create opacity object for number of dust sizes and number of wavelengths in grid

    opacs.generate_lam(1.e-1,1.e5); // Generate log grid of wavelengths in microns
    opacs.generate_a(sizes); // Copy grain sizes to opacity object
    opacs.interpolate_opacs(opac_tab);

    int n_bands = 20; // This is the number of bands used for the FLD routine
    WavelengthBinner bins(num_wavelengths, opacs.lam(), n_bands); // Bin large wavelength grid into smaller number of bands

    opacs.set_k_g_min_grain(1e-10);

    write_grids(dir, &g, &sizes, &opacs, &bins);

    // Create rho*kappa fields for absorption and scattering 

    Field3D<double> rhok_abs = create_field3D<double>(g, num_wavelengths);
    Field3D<double> rhok_sca = create_field3D<double>(g, num_wavelengths);
    Field3D<double> rhok_abs_binned = create_field3D<double>(g, n_bands);
    Field3D<double> rhok_sca_binned = create_field3D<double>(g, n_bands);

    // Disc & Star parameters
    
    double mu = 2.4, M_star = 1., alpha = 1.e-4, T_star=4500., R_star = 1.7*Rsun, Cv = 2.5*R_gas/mu;
    double L_star = 4.*M_PI*sigma_SB*std::pow(T_star, 4.)*std::pow(R_star, 2.);

    // Create star

    Star star(GMsun*M_star, L_star, T_star);
    star.set_wavelengths(num_wavelengths, opacs.lam());
    star.set_blackbody_fluxes();
    
    // Create fields for temperature solver

    Field<double> heat = create_field<double>(g);
    Field3D<double> J = create_field3D<double>(g, n_bands);
    Field3D<double> scattering = create_field3D<double>(g, num_wavelengths);
    Field3D<double> binned_scattering = create_field3D<double>(g, n_bands);
    Field<double> rho_tot = create_field<double>(g);
    
    // Create gas and dust fields (Qs = Quantities; object holds density and three-momenta, Ws = Primitives; object holds density and three-velocity)

    Field3D<Prims> Ws_d = create_field3D<Prims>(g, n_spec); // Dust quantities 
    Field<Prims> Ws_g = create_field<Prims>(g); // Gas primitives
    CudaArray<double> Sig_g = make_CudaArray<double>(g.NR+2*g.Nghost); // Gas surface density
    CudaArray<double> Sigdot_wind = make_CudaArray<double>(g.NR+2*g.Nghost); // Gas surface density

    CudaArray<double> nu = make_CudaArray<double>(g.NR+2*g.Nghost); // Kinematic viscosity
    CudaArray<double> dummysig = make_CudaArray<double>(g.NR+2*g.Nghost); // Kinematic viscosity
    Field<double> T = create_field<double>(g); // Temperature
    Field<double> cs = create_field<double>(g); // Sound speed
    Field<double> cs2 = create_field<double>(g); // Sound speed squared
    Field<double> alpha2D = create_field<double>(g); // alpha 2D
    Field3D<double> D = create_field3D<double>(g, n_spec); // Dust diffusion constant 

    // Set up initial dust and gas variables

    set_up_gas(g, Ws_g, Sig_g, nu, T, cs, cs2, alpha, star);

    double M_gas=0, M_dust=0;

    for (int i=g.Nghost; i<g.NR+g.Nghost; i++ ) { M_gas += Sig_g[i]*2.*M_PI*g.Rc(i)*g.dRe(i);}
    std::cout << "Initial gas mass: " << M_gas/Msun << " M_sun\n";
        
    int gas_boundary = BoundaryFlags::open_R_inner | BoundaryFlags::open_R_outer | BoundaryFlags::open_Z_outer;
    double gas_floor = 1e-100;
    double floor = 1.e-12;

    compute_hydrostatic_equilibrium(star, g, Ws_g, cs2, Sig_g);
    calc_gas_velocities_full(g, Sig_g, Ws_g, cs2, nu, alpha, star, gas_boundary, gas_floor);   
    for (int i=0; i<g.NR + 2*g.Nghost; i++) {
        for (int j=0; j<g.Nphi + 2*g.Nghost; j++) {
            alpha2D(i,j) = alpha;
        }
    }

    init_dust(g, Ws_d, Ws_g, Sig_g, sizes, cs, alpha, M_star, 1000, gas_floor);

    for (int i=g.Nghost; i<g.NR + g.Nghost; i++) {
        for (int j=g.Nghost; j<g.Nphi + g.Nghost; j++) { 
            for (int k=0; k<Ws_d.Nd; k++) {
                M_dust += 4.*M_PI * Ws_d(i,j,k).rho * g.volume(i,j); // 4pi comes from 2pi in azimuth and 2 for symmetry about midplane
            }
        }
    }

    std::cout << "Initial dust mass: " << M_dust/Msun << " M_sun\n";

    // Set up molecule

    Molecule CO(g, 28*m_H, 850, sizes.size());
    IceVapChem COchem(g, T, bins, J, Ws_d, Ws_g, sizes, CO, 1.e-100*floor);

    // Initialise coag solver

    BirnstielKernelIce kernel = BirnstielKernelIce(g, sizes, Ws_d, Ws_g, cs, alpha2D, mu, M_star);
    BS32Integration<CoagulationRate<decltype(kernel), SimpleErosion>>
        coagulation_integrate(
            create_coagulation_rate(
                sizes, 
                kernel, 
                SimpleErosion(1,11/6.,sizes.min_mass())), 
            1e-2, 1e-10
        ) ;


    // Initialise temperature solver

    FLD_Solver FLD(10, 1e-5, 5000);

    FLD.set_boundaries(BoundaryFlags::open_R_inner | 
                       BoundaryFlags::open_R_outer | 
                       BoundaryFlags::open_Z_outer) ;

    double tol=1;
    int n = 0;

    // Initialise diffusion-advection solver

    SourcesIce src(T, Ws_g, sizes, floor, M_star, mu);
    DustDynamics dyn(D, cs, src, 0.4, 0.2, floor, gas_floor);

    double dt_CFL = dyn.get_CFL_limit(g, Ws_d, Ws_g);

    std::cout << dt_CFL << "\n";
    
    // Choose times to store data
    
    double t = 0, dt;
    const int ntimes = 8;  
    double ts[ntimes] = {0.1*year, 1*year, 10*year, 100*year, 1e3*year};

    // Set up boundary conditions

    int boundary = BoundaryFlags::open_R_inner | BoundaryFlags::open_R_outer | BoundaryFlags::open_Z_outer;

    dyn.set_boundaries(boundary);

    std::chrono::_V2::system_clock::time_point start,stop;
    start = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds duration;
    double yps;
    int count = 0;
    double t_coag = 0, t_temp = 0, err = 1., dt_1perc = year;

    dt_CFL = 1e5;

    int Nout = 1;

    double dummy = 0;

    double dt_coag = 0;

    std::ifstream f(dir / ("restart_params.dat"), std::ios::binary);
    double t_restart=0;

    if (f) {
        read_restart_file(dir / ("restart_params.dat"), count, t, dt_CFL, t_coag, t_temp, dt_coag, dt_1perc, dummy);
        std::cout << "Restart params: " << count << " " << t/year << " " << dt_CFL/year << "\n";

        read_prims(dir, "restart", Ws_d, Ws_g, Sig_g);
        read_temp(dir, "restart", T, J);
        COchem.read_mol(dir, "restart"); 

        compute_cs2(g,T,cs2,mu);
        cs2_to_cs(g, cs, cs2);
        compute_D(g, D, Ws_g, cs2, M_star, alpha, 1.);
        compute_nu(g, nu, cs2, M_star, alpha);
        t_restart = t;
    }
    else {
        std::cout << "Computing initial temperature structure\n"; 

        for (int i=0; i<4; i++) {

            n=0;
            tol = 1;

            init_dust(g, Ws_d, Ws_g, Sig_g, sizes, cs, alpha, M_star, 1000, gas_floor);
            while (n<10 && tol>0.0001) {

                Field<double> oldT = create_field<double>(g);
                copy_field(g, T, oldT); 
                
                std::cout << "Iteration: " << n << "\n" ;  

                calculate_total_rhokappa(g, Ws_d, Ws_g, rho_tot, opacs, rhok_abs, rhok_sca);

                rhok_abs_binned = bins.bin_planck(g, rhok_abs, T);
                bin_central(g, rhok_sca, rhok_sca_binned, num_wavelengths, n_bands);

                compute_stellar_heating_with_scattering(star, g, rhok_abs, rhok_sca, heat, scattering);
                add_viscous_heating(star, g, Ws_g, nu, heat);
                binned_scattering = bins.bin_field(g, scattering, bins.SUM);

                if (i==0) {
                    setup_init_J(g,heat,J);
                }

                FLD.solve_multi_band(g, 0, Cv, rhok_abs_binned, rhok_sca_binned, rho_tot, heat, binned_scattering, bins.edges, T, J);

                compute_cs2(g,T,cs2,mu);
                compute_hydrostatic_equilibrium(star, g, Ws_g, cs2, Sig_g, gas_floor);
                cs2_to_cs(g, cs, cs2);

                std::cout << "T:" << T[T.index(1,1)] << " " << T[T.index(g.NR, 1)] 
                        << " "<< T[T.index(1, g.Nphi)] << " " <<  T[T.index(g.NR, g.Nphi)]
                        << "\n" << std::endl ;

                tol = fracerr(g, oldT, T);
                std::cout << "Fractional error: "<< tol << "\n" << "\n";

                n += 1;
            }
        }
        
        calc_gas_velocities_full(g, Sig_g, Ws_g, cs2, nu, alpha, star, gas_boundary, gas_floor);   

        compute_nu(g, nu, cs2, M_star, alpha);
        compute_D(g, D, Ws_g, cs2, M_star, alpha, 1.);

        // Initialise CO vapour

        for (int i=g.Nghost; i<g.NR+g.Nghost; i++) {
            for (int j=g.Nghost; j<g.Nphi+g.Nghost; j++) {
                CO.vap(i,j) = ((2.*Ws_g(i,j).rho/(2.8*m_H)) * 2e-4)*28*m_H;
                for (int k=0; k<Ws_d.Nd; k++) {
                    CO.ice(i,j,k) = 0;
                }
            }
        }

        double dt_chem=0.1*year;
        for (int i=0; i<20; i++) {    
            COchem.imp_update(dt_chem*2.);
        }

        n = 0;
        tol = 1;
        while (n<20 && tol>0.0001) {

            Field<double> oldT = create_field<double>(g);
            copy_field(g, T, oldT); 
            
            std::cout << "Iteration: " << n << "\n" ;  

            calculate_total_rhokappa(g, Ws_d, Ws_g, rho_tot, opacs, rhok_abs, rhok_sca, CO);

            rhok_abs_binned = bins.bin_planck(g, rhok_abs, T);
            bin_central(g, rhok_sca, rhok_sca_binned, num_wavelengths, n_bands);

            compute_stellar_heating_with_scattering(star, g, rhok_abs, rhok_sca, heat, scattering);
            add_viscous_heating(star, g, Ws_g, nu, heat);
            binned_scattering = bins.bin_field(g, scattering, bins.SUM);

            FLD.solve_multi_band(g, 0, Cv, rhok_abs_binned, rhok_sca_binned, rho_tot, heat, binned_scattering, bins.edges, T, J);

            compute_cs2(g,T,cs2,mu);
            compute_hydrostatic_equilibrium(star, g, Ws_g, cs2, Sig_g, CO, gas_floor, 1e-100*floor);
            cs2_to_cs(g, cs, cs2);

            std::cout << "T:" << T[T.index(1,1)] << " " << T[T.index(g.NR, 1)] 
                    << " "<< T[T.index(1, g.Nphi)] << " " <<  T[T.index(g.NR, g.Nphi)]
                    << "\n" << std::endl ;

            tol = L2err(g, oldT, T);
            std::cout << "Fractional error: "<< tol << "\n" << "\n";

            n += 1;
        }
        
        calc_gas_velocities_full(g, Sig_g, Ws_g, cs2, nu, alpha, star, gas_boundary, gas_floor);   

        compute_nu(g, nu, cs2, M_star, alpha);
        compute_D(g, D, Ws_g, cs2, M_star, alpha, 1.);    
                        
        write_prims(dir, 0, g, Ws_d, Ws_g, Sig_g);
        write_temp(dir, 0, g, T, J);
        COchem.write_mol(dir, 0);   
        dt_CFL = 1; 
        t_coag=0*year;
    }

    double dt_temp_max = 5000*year;

    // Main timestep iteration

    for (double ti : ts) {

        if (t > ti) {
            continue;
        }

        while (t < ti) {  

            if (!(count%100)) {
                std::cout << "t = " << t/year << " years\n";
                std::cout << "dt = " <<dt_CFL/year << " years\n";
                stop = std::chrono::high_resolution_clock::now();
                yps = ((t)/year) / std::chrono::duration_cast<std::chrono::seconds>(stop - start).count();
                std::cout << "Years per second: " << yps << "\n";
                std::cout << "No. of cells per second: " <<   (double)((g.NR+2*g.Nghost)*(g.Nphi+2*g.Nghost)*n_spec*count)/ (double)(std::chrono::duration_cast<std::chrono::seconds>(stop - start).count()) << "\n" ;
            }

            dt = std::min(dt_CFL, ti-t); // Set time-step according to CFL condition or proximity to selected time snapshots
            
            dyn(g, Ws_d, Ws_g, dt, CO, sizes); // Diffusion-advection update

            update_gas_sigma(g, Sig_g, dt, nu, gas_boundary, gas_floor);
            compute_hydrostatic_equilibrium(star, g, Ws_g, cs2, Sig_g, CO, gas_floor);
            compute_D(g, D, Ws_g, cs2, M_star, alpha, 1.);    

            if ( ((t+dt >= t_coag+dt_coag)|| (t+2*dt >= t_coag+dt_coag && dt < dt_coag) || ((t+dt)-t_coag)>50.*year || dt == ti-t )) {

                std::cout << "Coag step at count = " << count << "\n";
                coagulation_integrate.integrate_tracers(g, Ws_d, Ws_g, CO, (t+dt)-t_coag, dt_coag, floor) ;
                t_coag = t+dt;
            } 

            // Temperature update
        
            if (count == 1 || (t+dt)-t_temp > .1*dt_1perc || ((t+dt)-t_temp)>dt_temp_max || dt == ti-t) {

                bool exit = false;

                std::cout << "Temp step at count = " << count << "\n";

                Field<double> oldT = create_field<double>(g);
                Field3D<double> oldJ = create_field3D<double>(g, n_bands);
                copy_field(g, T, oldT); 
                copy_field(g, J, oldJ); 
                
                err = 1.;
                int Tcount = 0;

                while (err > 0.1) {

                    if (Tcount == 3) {
                        copy_field(g, oldT, T); 
                        copy_field(g, oldJ, J);

                        exit = true;
                        std::cout << "Temp break: move ahead.\n";
                        break;
                    }

                    if (Tcount > 0) { 
                        FLD.set_precond_level(1);
                        FLD.set_tolerance(std::pow(10.,-2));
                        copy_field(g, oldT, T); 
                        copy_field(g, oldJ, J);
                    }

                    calculate_total_rhokappa(g, Ws_d, Ws_g, rho_tot, opacs, rhok_abs, rhok_sca, CO);

                    rhok_abs_binned = bins.bin_planck(g, rhok_abs, T);
                    bin_central(g, rhok_sca, rhok_sca_binned, num_wavelengths, n_bands);

                    compute_stellar_heating_with_scattering(star, g, rhok_abs, rhok_sca, heat, scattering);
                    add_viscous_heating(star, g, Ws_g, nu, heat);
                    binned_scattering = bins.bin_field(g, scattering, bins.SUM);

                    FLD.solve_multi_band(g, (t+dt)-t_temp, Cv, rhok_abs_binned, rhok_sca_binned, rho_tot, heat, binned_scattering, bins.edges, T, J);
                    err = fracerr(g, oldT, T);
                    Tcount += 1;
                }
                FLD.set_precond_level(0);
                FLD.set_tolerance(1e-5);
                if (exit == true) { 
        
                }
                else {
                    compute_cs2(g,T,cs2,mu);
                    compute_hydrostatic_equilibrium(star, g, Ws_g, cs2, Sig_g, CO, gas_floor, 1e-100*floor);
                    compute_D(g, D, Ws_g, cs2, M_star, alpha, 1.);
                    compute_nu(g, nu, cs2, M_star, alpha);
                    calc_gas_velocities_full(g, Sig_g, Ws_g, cs2, nu, alpha, star, gas_boundary, gas_floor);   

                    std::cout << "T:" << T[T.index(1,1)] << " " << T[T.index(g.NR, 1)] 
                            << " "<< T[T.index(1, g.Nphi)] << " " <<  T[T.index(g.NR, g.Nphi)]
                            << "\n" << std::endl ;

                    err = fracerr(g, oldT, T);
                    dt_1perc = ((t+dt)-t_temp) * 0.01/err;
                    std::cout << "Error: " << err << " " << "dt_1perc: " << dt_1perc/year << " years " << "Time: " << (t+dt)/year << " years\n";

                    t_temp = t+dt;
                    dt_temp_max = 5000*year;
                }
            }

            COchem.imp_update(dt);

            count += 1;
            t += dt;
            dt_CFL = std::min(dyn.get_CFL_limit(g, Ws_d, Ws_g), 2.*dt); 

            // if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count()/3600. > 29.) {
            //     std::cout << "Writing restart at t = " << t/year << " years.\n" ;
            //     write_restart_file(dir / ("restart_params.dat"), count, t, dt_CFL, t_coag, t_temp, dt_coag, dt_1perc, dummy);
            //     write_init_dens(dir / ("dens_restart.dat"), g, Ws_d, Ws_g, Sig_g, 1, n_spec);
            //     write_init_temp(dir / ("temp_restart.dat"), g, T, J);
            //     return 0;
            // } 

        }

        // Append densities and temperature to files at time snapshots

        write_prims(dir, Nout, g, Ws_d, Ws_g, Sig_g);
        write_temp(dir, Nout, g, T, J);
        COchem.write_mol(dir, Nout);

        Nout+=1;

    }


    std::ofstream fin(dir / ("finished"));
    fin.close();


    stop = std::chrono::high_resolution_clock::now();
    std::cout << count << " timesteps\n" ;
    
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time taken: " << duration.count()/(1.e6*60.) << " mins" << std::endl;  
    std::cout << "No of cells : " << (g.NR+2*g.Nghost)*(g.Nphi+2*g.Nghost)*n_spec << std::endl;  
    std::cout << "No. of cells per second: " <<   ((g.NR+2*g.Nghost)*(g.Nphi+2*g.Nghost)*n_spec*count)/ (duration.count()/1.e6) << "\n" ;
    return 0;
} 
