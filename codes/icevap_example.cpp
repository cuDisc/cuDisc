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

void set_up_gas(Grid& g, CudaArray<double>& Sig_g, CudaArray<double>& nu, Field<double>& T, Field<double>& cs, Field<double>& cs2, double alpha, Star& star) {
  
    double r_c = 30*au;
    double mu = 2.4;
    double Mtot = 0.;
    double Mdisc = 0.07*Msun;

    for (int i=0; i<g.NR+2*g.Nghost; i++) {

        Sig_g[i] =  std::pow(g.Rc(i)/au, -1.) * std::exp(-g.Rc(i)/r_c);
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
    
void init_dust(Grid& g, Field3D<Prims>& wd, Field<Prims>& wg, CudaArray<double>& Sig_g, SizeGrid& sizes, Field<double>& cs, CudaArray<double>& nu, double Mstar, double u_f, double d_to_g, double gfloor) {

    auto dtg = [d_to_g](double R) {
        return d_to_g;
    };

    double Sc = 1.;
    CudaArray<double> P = make_CudaArray<double>(g.NR+2*g.Nghost);

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

    for (int i=0; i<g.NR + 2*g.Nghost; i++) {
        double Sig_kk[wd.Nd];
        double Sig_kktot = 0.;
        double Om = std::sqrt(GMsun*Mstar/(g.Rc(i)*g.Rc(i)*g.Rc(i)));

        double a_frag = std::max(1e-5,0.5* 2./(M_PI) * Sig_g[i]/(sizes.solid_density()*Om*nu[i]) * std::pow(u_f, 2.));

        double dlnPdlnR = (log(P[i+1]) - log(P[i])) / (log(g.Re(i+1)) - log(g.Re(i))); 
        double eta = - cs(i,2)*cs(i,2)/(Om*Om*g.Rc(i)*g.Rc(i)) * dlnPdlnR;
        double a_drift = std::max(1e-5, 0.5*(2*(dtg(g.Rc(i))*Sig_g[i])/(M_PI*sizes.solid_density())) * ((Om*Om*g.Rc(i)*g.Rc(i))/(cs(i,2)*cs(i,2))) * (1./std::abs(dlnPdlnR)));

        for (int k=0; k<wd.Nd; k++) {
            Sig_kk[k] = std::pow(sizes.centre_size(k)/sizes.centre_size(0), 0.5) * std::exp(-std::pow(sizes.centre_size(k)/(1e-4),5.));
            Sig_kktot += Sig_kk[k];
        }

        for (int k=0; k<wd.Nd; k++) {
            double Sig_k=0;
            wd(i,g.Nghost,k).rho = 1.e-2;
            for (int j=g.Nghost; j<g.Nphi + g.Nghost-1; j++) {
                double St = std::max(1e-5,sizes.solid_density() * sizes.centre_size(k) / (wg(i,j).rho*1.59577*cs(i,j)) * Om);
                wd(i,j+1,k).rho = std::exp(std::log(wg(i,j+1).rho * (wd(i,j,k).rho/wg(i,j).rho)*(1.-g.dZc(i,j)*St*Om*g.Zc(i,j)/nu[i])));  
                if (wd(i,j+1,k).rho != wd(i,j+1,k).rho) {wd(i,j+1,k).rho = 0.;}
                Sig_k += 2.*wd(i,j,k).rho*g.dZe(i,j);
            }
            Sig_k += 2.*wd(i,g.Nphi + g.Nghost-1,k).rho*g.dZe(i,g.Nphi + g.Nghost-1);

            for (int j=g.Nghost; j<g.Nphi + g.Nghost; j++) {
                wd(i,j,k).rho *= dtg(g.Rc(i)) * Sig_g[i] * (Sig_kk[k]/Sig_kktot)/ Sig_k;//* std::exp(-std::pow(1.*au/g.Rc(i),.8));
                if (wd(i,j,k).rho != wd(i,j,k).rho) {wd(i,j,k).rho = 0.;}
            }
            
            wd(i,0,k).rho = wd(i,g.Nghost+1,k).rho;
            wd(i,1,k).rho = wd(i,g.Nghost,k).rho;
            wd(i,g.Nphi+2*g.Nghost-1,k).rho = wd(i,g.Nphi+g.Nghost-1,k).rho;
            wd(i,g.Nphi+2*g.Nghost-2,k).rho = wd(i,g.Nphi+g.Nghost-1,k).rho;

            for (int j=0; j<g.Nphi + 2*g.Nghost; j++) {
                // Set initial dust velocities through standard drift velocity equations
                double St = sizes.solid_density() * sizes.centre_size(k) / (wg(i,j).rho*1.59577*cs(i,j)) * Om;
                wd(i,j,k).v_R   = (wg(i,j).v_R - eta * Om*g.Rc(i) * St) / (1 + St*St) ;
                wd(i,j,k).v_phi = Om*g.Rc(i); 
                wd(i,j,k).v_Z = - Om * St * g.Zc(i,j); 
            }
        }
    }

    for (int i=0; i<g.NR+2*g.Nghost; i++) {
        for (int j=0; j<g.Nphi+2*g.Nghost; j++) {
            for (int k=0; k < sizes.size(); k++) {
                double Om = std::sqrt(GMsun*Mstar/(g.Rc(i)*g.Rc(i)*g.Rc(i)));
                if (wd(i,j,k).rho < 1e-12*wg(i,j).rho) {
                    wd(i,j,k).rho = 1e-12*wg(i,j).rho;
                    wd(i,j,k).v_R   = 0;
                    wd(i,j,k).v_phi = Om*g.Rc(i); 
                    wd(i,j,k).v_Z = 0;
                }   
            }
        }
    }
}

void compute_cs2(const Grid &g, Field<double> &T, Field<double> &cs2, Field<double> &mu) {

    // Calculates square of the sound speed

    for (int i=0; i<g.NR + 2*g.Nghost; i++) {
        for (int j=0; j<g.Nphi + 2*g.Nghost; j++) {
            cs2(i,j) = R_gas * T(i,j) / mu(i,j);
        }
    }
}

void compute_alpha(Grid& g, CudaArray<double>& nuCA, Field<double>& alpha2D, Field<double>& cs2, double M_star) {

    for (int i=0; i<g.NR + 2*g.Nghost; i++) {
        double Omk = std::sqrt(GMsun*M_star/g.Rc(i))/g.Rc(i);
        for (int j=0; j<g.Nphi + 2*g.Nghost; j++) {
            alpha2D(i,j) = nuCA[i] * Omk / cs2(i,j);
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

void compute_Cv(const Grid &g, Field<double> &Cv, Field<double>& mu) {

    // Specific heat at const. vol.

    for (int i=0; i<g.NR + 2*g.Nghost; i++) {
        for (int j=0; j<g.Nphi + 2*g.Nghost; j++) {
            Cv(i,j) = 2.5*R_gas / mu(i,j);
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

    std::filesystem::path dir = std::string("./codes/outputs/icevap_example");
    std::filesystem::create_directories(dir);

    // Set up spatial grid 

    Grid::params p;
    p.NR = 100;
    p.Nphi = 100;
    p.Nghost = 2;

    p.Rmin = 5.*au;
    p.Rmax = 200.*au;
    p.theta_min = 0. ;
    p.theta_power = 0.75;
    p.theta_max = M_PI/8.;

    p.R_spacing = RadialSpacing::log ;
    p.theta_spacing = ThetaSpacing::power;

    Grid g(p);

    // Setup a size distribution

    double rho_p = 1.675;
    double a0 = 1e-5 ; // Grain size lower bound in cm
    double a1 = 200.   ;  // Grain size upper bound in cm
    int n_spec = 7.*3.*std::log10(a1/a0) + 1;
    SizeGridIce sizes(g, a0, a1, n_spec, rho_p, 0.89) ;

    double v_frag = 200.;

    // Set up opacities and wavelength bins

    int num_wavelengths = 200; // This is the number of wavelengths used for stellar heating calculations

    CuzziOpacs<DSHARPwCOComp> opacs(sizes.size(), num_wavelengths, 1.e-1, 1.e5);
    opacs.calc_opacs(sizes);

    int n_bands = 20; // This is the number of bands used for the FLD routine
    WavelengthBinner bins(num_wavelengths, opacs.lam(), n_bands); // Bin large wavelength grid into smaller number of bands

    write_grids(dir, &g, &sizes, &opacs, &bins);

    // Create rho*kappa fields for absorption and scattering 

    Field3D<double> rhok_abs = create_field3D<double>(g, num_wavelengths);
    Field3D<double> rhok_sca = create_field3D<double>(g, num_wavelengths);
    Field3D<double> rhok_abs_binned = create_field3D<double>(g, n_bands);
    Field3D<double> rhok_sca_binned = create_field3D<double>(g, n_bands);

    // Disc & Star parameters
    
    double mu_HHe = 2.4, M_star = 1., alpha = 1.e-3, T_star=4500., R_star = 1.7*Rsun, Cv = 2.5*R_gas/mu_HHe;
    double L_star = 4.*M_PI*sigma_SB*std::pow(T_star, 4.)*std::pow(R_star, 2.);

    Field<double> mu2D = create_field<double>(g);
    Field<double> Cv2D = create_field<double>(g);
    set_all(g, mu2D, mu_HHe);
    set_all(g, Cv2D, Cv);


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

    CudaArray<double> nu = make_CudaArray<double>(g.NR+2*g.Nghost); // Kinematic viscosity
    Field<double> T = create_field<double>(g); // Temperature
    Field<double> cs = create_field<double>(g); // Sound speed
    Field<double> cs2 = create_field<double>(g); // Sound speed squared
    Field<double> alpha2D = create_field<double>(g); // alpha 2D
    Field3D<double> D = create_field3D<double>(g, n_spec); // Dust diffusion constant 

    // Set up initial dust and gas variables

    set_up_gas(g, Sig_g, nu, T, cs, cs2, alpha, star);

    double M_gas=0, M_dust=0;

    for (int i=g.Nghost; i<g.NR+g.Nghost; i++ ) { M_gas += Sig_g[i]*2.*M_PI*g.Rc(i)*g.dRe(i);}
    std::cout << "Initial gas mass: " << M_gas/Msun << " M_sun\n";
        
    int gas_boundary = BoundaryFlags::const_Mdot_R_inner | BoundaryFlags::const_Mdot_R_outer | BoundaryFlags::open_Z_outer;
    double gas_floor = 1e-100;
    double floor = 1.e-10;

    compute_hydrostatic_equilibrium(star, g, Ws_g, cs2, Sig_g);
    calc_gas_velocities(g, Sig_g, Ws_g, cs2, nu, alpha, star, gas_boundary, gas_floor);   
    compute_alpha(g, nu, alpha2D, cs2, M_star);

    // init_dust(g, Ws_d, Ws_g, Sig_g, sizes, cs, alpha, M_star, v_frag, gas_floor);
    init_dust(g, Ws_d, Ws_g, Sig_g, sizes, cs, nu, M_star, v_frag, 0.01, gas_floor);

    for (int i=g.Nghost; i<g.NR + g.Nghost; i++) {
        for (int j=g.Nghost; j<g.Nphi + g.Nghost; j++) { 
            for (int k=0; k<Ws_d.Nd; k++) {
                M_dust += 4.*M_PI * Ws_d(i,j,k).rho * g.volume(i,j); // 4pi comes from 2pi in azimuth and 2 for symmetry about midplane
            }
        }
    }

    std::cout << "Initial dust mass: " << M_dust/Msun << " M_sun\n";

    // Set up molecule

    CudaArray<double> h_phdiss = make_CudaArray<double>(g.NR+2*g.Nghost);
    Field<double> F_UV = create_field<double>(g); // UV_field

    Molecule CO(g, 28*m_H, 850, sizes.size());
    IceVapChem COchem(g, T, bins, J, Ws_d, Ws_g, sizes, CO, h_phdiss, F_UV, mu2D, mu_HHe, floor);

    // Initialise coag solver

    BirnstielKernelIce kernel = BirnstielKernelIce(g, sizes, Ws_d, Ws_g, cs, alpha2D, mu2D, M_star);
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

    SourcesIce src(T, Ws_g, sizes, floor, M_star, mu2D);
    DustDynamics dyn(D, cs, src, 0.4, 0.2, floor, gas_floor);

    double dt_CFL = dyn.get_CFL_limit(g, Ws_d, Ws_g);

    std::cout << dt_CFL << "\n";
    
    // Choose times to store data
    
    double t = 0, dt;
    const int ntimes = 6;  
    double ts[ntimes] = {1*year, 10*year, 100*year, 1e3*year, 1e4*year, 1e5*year};

    // Set up boundary conditions

    int boundary = BoundaryFlags::open_R_inner | BoundaryFlags::open_R_outer | BoundaryFlags::open_Z_outer;

    dyn.set_boundaries(boundary);

    std::chrono::_V2::system_clock::time_point start,stop;
    start = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds duration;
    double yps;
    int count = 0;
    double t_coag = 0, t_temp = 0, t_chem = 0, err = 1., dt_1perc = year, dt_1percchem = 1;

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
        // COchem.read_restart_file(dir, t_chem, dt_1percchem);

        compute_cs2(g,T,cs2,mu2D);
        cs2_to_cs(g, cs, cs2);
        compute_D(g, D, Ws_g, cs2, M_star, alpha, 1.);
        compute_nu(g, nu, cs2, M_star, alpha);

        update_sizegrid(g, sizes, Ws_d, CO.ice);

        calculate_total_rhokappa(g, sizes, Ws_d, Ws_g, rho_tot, opacs, rhok_abs, rhok_sca, CO);
        compute_stellar_UV_field(star, g, rhok_abs, rhok_sca, F_UV);
        
        calc_photodiss_surface(g, Ws_g, h_phdiss, 1.3e21);

        t_restart = t;
    }
    else {
        std::cout << "Computing initial temperature structure\n"; 

        // for (int i=0; i<4; i++) {

        n=0;
        tol = 1;


        while (n<20 && tol>0.0001) {

            Field<double> oldT = create_field<double>(g);
            copy_field(g, T, oldT); 
            
            std::cout << "Iteration: " << n << "\n" ;  

            // init_dust(g, Ws_d, Ws_g, Sig_g, sizes, cs, alpha, M_star, v_frag, gas_floor);
            init_dust(g, Ws_d, Ws_g, Sig_g, sizes, cs, nu, M_star, v_frag, 0.01, gas_floor);

            calculate_total_rhokappa(g, sizes, Ws_d, Ws_g, rho_tot, opacs, rhok_abs, rhok_sca, CO);

            rhok_abs_binned = bins.bin_planck(g, rhok_abs, T);
            bin_central(g, rhok_sca, rhok_sca_binned, num_wavelengths, n_bands);

            compute_stellar_heating_with_scattering(star, g, rhok_abs, rhok_sca, heat, scattering);
            add_viscous_heating(star, g, Ws_g, nu, heat);
            binned_scattering = bins.bin_field(g, scattering, bins.SUM);

            if (n==0) {
                setup_init_J(g,heat,J);
            }

            FLD.solve_multi_band(g, 0, Cv2D, rhok_abs_binned, rhok_sca_binned, rho_tot, heat, binned_scattering, bins.edges, T, J);

            compute_cs2(g,T,cs2,mu2D);
            compute_hydrostatic_equilibrium(star, g, Ws_g, cs2, Sig_g, gas_floor);
            cs2_to_cs(g, cs, cs2);

            std::cout << "T:" << T[T.index(1,1)] << " " << T[T.index(g.NR, 1)] 
                    << " "<< T[T.index(1, g.Nphi)] << " " <<  T[T.index(g.NR, g.Nphi)]
                    << "\n" << std::endl ;

            tol = fracerr(g, oldT, T);
            std::cout << "Fractional error: "<< tol << "\n" << "\n";

            n += 1;
        }
        
        calculate_total_rhokappa(g, sizes, Ws_d, Ws_g, rho_tot, opacs, rhok_abs, rhok_sca, CO);
        compute_stellar_UV_field(star, g, rhok_abs, rhok_sca, F_UV);
        
        calc_photodiss_surface(g, Ws_g, h_phdiss, 1.3e21);

        calc_gas_velocities(g, Sig_g, Ws_g, cs2, nu, alpha, star, gas_boundary, gas_floor);   

        compute_nu(g, nu, cs2, M_star, alpha);
        compute_alpha(g, nu, alpha2D, cs2, M_star);
        compute_D(g, D, Ws_g, cs2, M_star, alpha, 1.);

        // Initialise CO

        for (int i=0; i<g.NR+2*g.Nghost; i++) {
            for (int j=0; j<g.Nphi+2*g.Nghost; j++) {
                CO.vap(i,j) = ((2.*Ws_g(i,j).rho/(2.4*m_H) * 1.e-4)*28*m_H);
                for (int k=0; k<Ws_d.Nd; k++) {
                    CO.ice(i,j,k) = 1e-100;
                }
            }
        }

        double dt_ice = 0.1*year;
        for (int i=0; i<100; i++) {
            COchem.imp_update(dt_ice,dt_1percchem);
            dt_ice *= 2.;
        }

        compute_Cv(g, Cv2D, mu2D);

        n = 0;
        tol = 1;
        while (n<20 && tol>0.0001) {

            Field<double> oldT = create_field<double>(g);
            copy_field(g, T, oldT); 
            
            std::cout << "Iteration: " << n << "\n" ;  

            calculate_total_rhokappa(g, sizes, Ws_d, Ws_g, rho_tot, opacs, rhok_abs, rhok_sca, CO);

            rhok_abs_binned = bins.bin_planck(g, rhok_abs, T);
            bin_central(g, rhok_sca, rhok_sca_binned, num_wavelengths, n_bands);

            compute_stellar_heating_with_scattering(star, g, rhok_abs, rhok_sca, heat, scattering);
            add_viscous_heating(star, g, Ws_g, nu, heat);
            binned_scattering = bins.bin_field(g, scattering, bins.SUM);

            FLD.solve_multi_band(g, 0, Cv2D, rhok_abs_binned, rhok_sca_binned, rho_tot, heat, binned_scattering, bins.edges, T, J);

            compute_cs2(g,T,cs2,mu2D);
            compute_hydrostatic_equilibrium(star, g, Ws_g, cs2, Sig_g, CO, gas_floor, floor);
            cs2_to_cs(g, cs, cs2);

            std::cout << "T:" << T[T.index(1,1)] << " " << T[T.index(g.NR, 1)] 
                    << " "<< T[T.index(1, g.Nphi)] << " " <<  T[T.index(g.NR, g.Nphi)]
                    << "\n" << std::endl ;

            tol = L2err(g, oldT, T);
            std::cout << "Fractional error: "<< tol << "\n" << "\n";

            n += 1;
        }
        
        calc_gas_velocities(g, Sig_g, Ws_g, cs2, nu, alpha, star, gas_boundary, gas_floor);   

        compute_nu(g, nu, cs2, M_star, alpha);
        compute_alpha(g, nu, alpha2D, cs2, M_star);
        compute_D(g, D, Ws_g, cs2, M_star, alpha, 1.);    
                        
        write_prims(dir, 0, g, Ws_d, Ws_g, Sig_g);
        write_temp(dir, 0, g, T, J);
        COchem.write_mol(dir, 0);   
        dt_CFL = 1; 
        t_coag=0*year;
    }

    double dt_temp_max = 5000*year;
    double dt_chem_max = 100*year;

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
                yps = ((t-t_restart)/year) / std::chrono::duration_cast<std::chrono::seconds>(stop - start).count();
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
                double dt_coag_0 = dt_coag;
                coagulation_integrate.integrate_tracers(g, Ws_d, Ws_g, CO, (t+dt)-t_coag, dt_coag, floor) ;
                if (dt_coag > 0.) {    
                    t_coag = t+dt;
                }
                else {
                    dt_coag = dt_coag_0;
                }
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

                    calculate_total_rhokappa(g, sizes, Ws_d, Ws_g, rho_tot, opacs, rhok_abs, rhok_sca, CO);

                    rhok_abs_binned = bins.bin_planck(g, rhok_abs, T);
                    bin_central(g, rhok_sca, rhok_sca_binned, num_wavelengths, n_bands);

                    compute_stellar_heating_with_scattering(star, g, rhok_abs, rhok_sca, heat, scattering);
                    add_viscous_heating(star, g, Ws_g, nu, heat);
                    binned_scattering = bins.bin_field(g, scattering, bins.SUM);

                    FLD.solve_multi_band(g, (t+dt)-t_temp, Cv2D, rhok_abs_binned, rhok_sca_binned, rho_tot, heat, binned_scattering, bins.edges, T, J);
                    err = fracerr(g, oldT, T);
                    Tcount += 1;
                }
                FLD.set_precond_level(0);
                FLD.set_tolerance(1e-5);
                if (exit == true) { 
        
                }
                else {
                    compute_cs2(g,T,cs2,mu2D);
                    compute_hydrostatic_equilibrium(star, g, Ws_g, cs2, Sig_g, CO, gas_floor, floor);
                    compute_D(g, D, Ws_g, cs2, M_star, alpha, 1.);
                    compute_nu(g, nu, cs2, M_star, alpha);
                    compute_alpha(g, nu, alpha2D, cs2, M_star);
                    calc_gas_velocities(g, Sig_g, Ws_g, cs2, nu, alpha, star, gas_boundary, gas_floor);   

                    compute_stellar_UV_field(star, g, rhok_abs, rhok_sca, F_UV);
        
                    calc_photodiss_surface(g, Ws_g, h_phdiss, 1.3e21);

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

            if (((t+dt)-t_chem > .1*dt_1percchem || ((t+dt)-t_chem)>dt_chem_max || dt == ti-t)) {
                std::cout << "Ice-vap step at count = " << count << "\n";

                double dt_ice = (t+dt)-t_chem;
                COchem.imp_update(dt_ice, dt_1percchem);
                compute_Cv(g, Cv2D, mu2D);
                t_chem += dt_ice;
                std::cout << "dt_1perc: " << dt_1percchem/year << " years " << "Time: " << (t+dt)/year << " years\n";
            }

            count += 1;
            t += dt;
            if (count < 200) {
                dt_CFL = std::min(dyn.get_CFL_limit(g, Ws_d, Ws_g), 1.1*dt); // Calculate new CFL condition time-step 
            }
            else {
                dt_CFL = dyn.get_CFL_limit(g, Ws_d, Ws_g);
            }

            if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count()/3600. > 29.) {
                std::cout << "Writing restart at t = " << t/year << " years.\n" ;
                write_restart_file(dir / ("restart_params.dat"), count, t, dt_CFL, t_coag, t_temp, dt_coag, dt_1perc, t_chem, dt_1percchem);
                write_prims(dir, "restart", g, Ws_d, Ws_g, Sig_g);
                write_temp(dir, "restart", g, T, J) ;
                COchem.write_mol(dir,"restart");
                return 0;
            }

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
