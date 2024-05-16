#include <iostream>
#include <string>

#include "constants.h"
#include "dustdynamics1D.h"
#include "dustdynamics.h"
#include "gas1d.h"
#include "file_io.h"

#include "coagulation/integration.h"
#include "coagulation/kernels.h"
#include "coagulation/coagulation.h"

double calc_mass(Grid& g, Field3D<Prims1D>& q) {

    double mass=0;

    for (int i=g.Nghost; i<g.NR+g.Nghost; i++) {
        for (int j=g.Nghost; j<g.Nphi+g.Nghost; j++) {
            for (int k=0; k<q.Nd; k++) {
                mass += 2.*M_PI*g.Rc(i)*q(i,j,k).Sig * g.dRe(i);
            }
        }
    }

    return mass;
}

void set_up(Grid& g, Field3D<Prims1D>& W_d, Field<Prims1D>& W_g, SizeGrid& sizes, Field<double>& T, 
            Field<double>& cs, CudaArray<double>& nu, Field<double>& alpha2D, Field3D<double>& D, double alpha, Star& star) {

    double M_disc = 0.07*Msun;
    double dtg = 0.01;
    double r_c = 30*au;
    double Mtot=0;

    for (int i=0; i<g.NR+2*g.Nghost; i++) {
  
        W_g(i,g.Nghost).Sig = std::pow(g.Rc(i)/r_c, -1) * std::exp(-g.Rc(i)/r_c);

        Mtot += 2.*M_PI*g.Rc(i)*W_g(i,g.Nghost).Sig*g.dRe(i);
        

        double v_k = std::sqrt(star.GM/g.Rc(i));
        T(i,g.Nghost) = std::pow(6.25e-3 * star.L / (M_PI * g.Rc(i)*g.Rc(i) * sigma_SB), 0.25);
        cs(i,g.Nghost) = std::sqrt(k_B*T(i,g.Nghost)/(2.4*m_H));

        W_g(i,g.Nghost).v_R = 0;
        
        nu[i] = alpha * cs(i,g.Nghost)*cs(i,g.Nghost) * (g.Rc(i)/v_k);
        alpha2D(i,g.Nghost) = alpha;
    }
    
    for (int i=0; i<g.NR+2*g.Nghost; i++) {
        W_g(i,g.Nghost).Sig *=  M_disc/Mtot;
    }

    double Sc = 1;
    double Mdtot = 0;
    for (int i=0; i<g.NR+2*g.Nghost; i++) {
        double Sig_tot = 0;
        for (int k=0; k<W_d.Nd; k++) {
            W_d(i,g.Nghost,k).Sig = std::pow(sizes.centre_size(k)/sizes.centre_size(0), 0.5) * std::exp(-std::pow(sizes.centre_size(k)/1e-5, 10.));
            Sig_tot += W_d(i,g.Nghost,k).Sig;
            Mdtot += 2.*M_PI*g.Rc(i)*W_d(i,g.Nghost,k).Sig*g.dRe(i);
            W_d(i,g.Nghost,k).v_R = W_g(i,g.Nghost).v_R;
            D(i,g.Nghost,k) =  W_g(i,g.Nghost).Sig * nu[i]/Sc;
        }
        for (int k=0; k<W_d.Nd; k++) {
            W_d(i,g.Nghost,k).Sig *= dtg*W_g(i,g.Nghost).Sig / Sig_tot;
        }
    }

}


int main() {

    std::filesystem::path dir = std::string("./codes/outputs/1Ddisc_multgrain");
    std::filesystem::create_directories(dir);
    
    Grid::params p;
    p.NR = 150;
    p.Nphi = 1;
    p.Nghost = 2;

    p.Rmin = 1.*au;
    p.Rmax = 1000.*au;
    p.R_spacing = RadialSpacing::log;

    p.theta_min = -M_PI / 20.;
    p.theta_max = M_PI / 20.;

    Grid g(p);

    double alpha = 1.e-3;
    double rho_s = 1.67;
    double a0 = 1.e-5;
    double a_max = 10.;
    int n_spec = 3*7*std::log10(a_max/a0) + 1;
    std::cout << n_spec << "\n";
    double v_frag = 100.;
    SizeGrid sizes(a0, a_max, n_spec, rho_s);

    write_grids(dir, &g, &sizes);

    double M_star = 0.7, T_star=4500., R_star = 1.7*Rsun;
    double L_star = 4.*M_PI*sigma_SB*std::pow(T_star, 4.)*std::pow(R_star, 2.);
    Star star(GMsun*M_star, L_star, T_star);

    Field<Prims1D> W_g  = create_field<Prims1D>(g);
    Field3D<Prims1D> W_d = create_field3D<Prims1D>(g, sizes.size());
    Field3D<double> D = create_field3D<double>(g, sizes.size());
    Field<double> T = create_field<double>(g);
    Field<double> cs = create_field<double>(g);
    Field<double> alpha2D = create_field<double>(g);
    CudaArray<double> nu = make_CudaArray<double>(g.NR+2*g.Nghost);

    double floor = 1e-15;
    double gas_floor = 1e-20;
    int gas_boundary = BoundaryFlags::open_R_inner | BoundaryFlags::open_R_outer;

    set_up(g, W_d, W_g, sizes, T, cs, nu, alpha2D, D, alpha, star);

    DustDyn1D dyn(D, cs, star, sizes, nu, 0.4, 0.1, floor, gas_floor);
    int boundary = BoundaryFlags::open_R_inner | BoundaryFlags::open_R_outer;
    dyn.set_boundaries(boundary);

    BirnstielKernelVertInt kernel = BirnstielKernelVertInt(g, sizes, W_d, W_g, cs, alpha2D, 2.4, M_star);
    kernel.set_fragmentation_threshold(v_frag);
    BS32Integration<CoagulationRate<BirnstielKernelVertInt, SimpleErosion>>
        coagulation_integrate(
            create_coagulation_rate(
                sizes, 
                kernel, 
                SimpleErosion(1,11/6.,sizes.min_mass())), 
            1e-2, 1e-10
        ) ;

        
    double t = 0, dt;
    const int ntimes = 9;  
    double ts[ntimes] = {100*year,1000*year,1e4*year,1e5*year,2e5*year, 5e5*year,1e6*year,2e6*year,5e6*year};

    std::chrono::_V2::system_clock::time_point start,stop;
    start = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds duration;
    double yps;
    int count = 0;
    double t_coag = 0, dt_coag = 0;

    double dt_CFL = 1;

    int Nout = 1;

    calc_v_gas(g, W_g, nu, star.GM, gas_floor);
    write_prims1D(dir, 0, g, W_d, W_g);

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
                yps = ((t)/year) / std::chrono::duration_cast<std::chrono::seconds>(stop - start).count();
                std::cout << "Years per second: " << yps << "\n";
                printf("%1.12g\n", calc_mass(g, W_d));
            }
            dt = std::min(dt_CFL, ti-t);
            dyn(g, W_d, W_g, dt);

            update_gas_sigma(g, W_g, dt, nu, gas_boundary, gas_floor);
            calc_v_gas(g, W_g, nu, star.GM, gas_floor);

            if ((t+dt >= t_coag+dt_coag)|| (t+2*dt >= t_coag+dt_coag && dt < dt_coag) || dt == ti-t) {
                std::cout << "Coag step at count = " << count << "\n";
                // Reset coagulation kernel with updated quantities
                kernel = BirnstielKernelVertInt(g, sizes, W_d, W_g, cs, alpha2D, 2.4);
                kernel.set_fragmentation_threshold(v_frag);
                coagulation_integrate.set_kernel(kernel);
                // Run coagulation internal integration (routine calculates its own sub-steps to integrate over the timestep passed into it)
                coagulation_integrate.integrate(g, W_d, W_g, (t+dt)-t_coag, dt_coag, floor) ;
                t_coag = t+dt;
            } 

            count += 1;
            t += dt;
            dt_CFL = std::min(dyn.get_CFL_limit(g, W_d, W_g),1.2*dt);

        }
        write_prims1D(dir, Nout, g, W_d, W_g);
        Nout +=1;
    }

    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time taken: " << duration.count()/(1.e6*60.) << " mins" << std::endl;  

    return 0;
}