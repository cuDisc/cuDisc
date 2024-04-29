#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <chrono>

#include "dustdynamics.h"
#include "grid.h"
#include "field.h"
#include "gas1d.h"
#include "star.h"
#include "cuda_array.h"
#include "constants.h"
#include "flags.h"
#include "file_io.h"

/*
    1D disc evolution for gas and dust using 2-population model (Birnstiel et al. 2012 https://www.aanda.org/articles/aa/pdf/2012/03/aa18136-11.pdf) for dust evolution
    Also includes example PE wind profile from Owen et al. 2011 https://academic.oup.com/mnras/article/412/1/13/984147 
*/

void setup_gas(Grid& g, CudaArray<double>& Sig_g, CudaArray<double>& u_gas, CudaArray<double>& nu, double alpha, Star& star) {

    double r_c = 30*au;
    double Sig_0 = 0.;
    double q = 0.5;
    double p = 1.;
    double mu = 2.4;
    double T0 = std::pow(6.25e-3 * star.L / (M_PI *au*au * sigma_SB), 0.25);

    double Mtot = 0.;
    double Mdisc = 0.07*Msun;

    double c_s, v_k, T;

    for (int i=0; i<g.NR+2*g.Nghost; i++) {

        Sig_g[i] = std::pow(g.Rc(i)/r_c, -p) * std::exp(-std::pow(g.Rc(i)/r_c, 2-p));
        Mtot += 2.*M_PI*g.Rc(i)*Sig_g[i]*g.dRe(i);
        v_k = std::sqrt(star.GM/g.Rc(i));
        T = T0*std::pow(g.Rc(i)/au, -q);
        c_s = std::sqrt(k_B*T/(mu*m_H));

        u_gas[i] = -3. * alpha * c_s*c_s / v_k * (2-p-q);
        
        nu[i] = alpha * c_s*c_s * (g.Rc(i)/v_k);
    }
    
    for (int i=0; i<g.NR+2*g.Nghost; i++) {
        Sig_0 = Mdisc/Mtot;
        Sig_g[i] *= Sig_0;
    }
    std::cout << Sig_0 << "\n";
}

void setup_dust(Grid& g, CudaArray<double>& Sig_d, CudaArray<double>& Sig_g) {

    for (int i=0; i<g.NR+2*g.Nghost; i++) {
        Sig_d[i] = 0.01*Sig_g[i];
    }

}

void Sigdot_w_JO(Grid& g, CudaArray<double>& Sigdot_w, double logLx, double Mstar) {

    double Mdot_w = 6.25e-9*std::pow(Mstar,-0.068)*std::pow(std::pow(10.,logLx)/std::pow(10.,30.),1.14) * (Msun/year);
    double a = 0.15138, b = -1.2182, c = 3.4046, d = -3.5717, e = -0.32762, f = 3.6064, g_ = -2.4918;

    CudaArray<double> Signorm = make_CudaArray<double>(g.NR+2*g.Nghost);

    for  (int i=0; i<g.NR+2*g.Nghost; i++) {

        if (0.85 * (g.Rc(i)/au) / Mstar > 0.7) {
            double x = 0.85 * (g.Rc(i)/au) / Mstar;
            double l10x = std::log10(x);
            double lx = std::log(x);
            double l10 = std::log(10.);
            
            Signorm[i] = (std::pow(10, a*std::pow(l10x,6)+b*std::pow(l10x,5)+c*std::pow(l10x,4)+d*std::pow(l10x,3)+e*std::pow(l10x,2)+f*l10x+g_)
                            * (6.*a*std::pow(lx, 5.)/(x*x*std::pow(l10,7.)) + 5.*b*std::pow(lx, 4.)/(x*x*std::pow(l10,6.)) +
                        4.*c*std::pow(lx, 3.)/(x*x*std::pow(l10,5.)) + 3.*d*std::pow(lx, 2.)/(x*x*std::pow(l10,4.)) + 
                        2.*e*lx/(x*x*std::pow(l10,3.)) + f/(x*x*l10*l10) ) * std::exp(-std::pow(x/100,10.))) * (Msun/(au*au*year));
        }
        else {
            Signorm[i] = 0.;
        }
    }
    double Mdottot = 0;
    for  (int i=0; i<g.NR+2*g.Nghost; i++) {
        Mdottot += 2.*M_PI*g.Rc(i)*Signorm[i]*g.dRe(i);
    }

    for  (int i=0; i<g.NR+2*g.Nghost; i++) {
        Sigdot_w[i] = Signorm[i] * Mdot_w/Mdottot ;
    }
}


void find_Rcav(Grid& g, CudaArray<double>& Sig_g, double& Rcav) {

    for (int i=0; i<g.NR; i++) {
        if (Sig_g[i] > 1.e-30) {
            Rcav = g.Rc(i);
            break;
        }
    }   
}

int main() {

    std::filesystem::path dir = std::string("./codes/outputs/1Ddisc");
    std::filesystem::create_directories(dir);
    
    Grid::params p;
    p.NR = 300;
    p.Nphi = 1;
    p.Nghost = 2;

    p.Rmin = 0.1*au;
    p.Rmax = 1000.*au;
    p.R_spacing = RadialSpacing::log;

    p.theta_min = -M_PI / 20.;
    p.theta_max = M_PI / 20.;

    Grid g(p);
    write_grid(dir, g);

    double alpha = 1.e-3;
    double rho_s = 1.25;
    double a0 = 1.e-5;
    double u_f = 1000.;
    double Rcav=0.;

    double M_star = 0.7, T_star=4500., R_star = 1.7*Rsun;
    double L_star = 4.*M_PI*sigma_SB*std::pow(T_star, 4.)*std::pow(R_star, 2.);
    Star star(GMsun*M_star, L_star, T_star);

    CudaArray<double> Sig_g = make_CudaArray<double>(g.NR+2*g.Nghost);
    CudaArray<double> Sigdot_w = make_CudaArray<double>(g.NR+2*g.Nghost);
    CudaArray<double> u_g = make_CudaArray<double>(g.NR+2*g.Nghost);
    CudaArray<double> nu = make_CudaArray<double>(g.NR+2*g.Nghost);

    CudaArray<double> Sig_d = make_CudaArray<double>(g.NR+2*g.Nghost);
    CudaArray<double> ubar = make_CudaArray<double>(g.NR+2*g.Nghost);

    setup_gas(g, Sig_g, u_g, nu, alpha, star);
    setup_dust(g, Sig_d, Sig_g);
    find_Rcav(g, Sig_g, Rcav);
    printf("Rcav = %g\n",Rcav/au);
    Sigdot_w_JO(g, Sigdot_w, 30.3, M_star);

    int boundary = 0;
    int boundaryg = BoundaryFlags::open_R_inner | BoundaryFlags::open_R_outer;

    calculate_ubar(g, Sig_d, Sig_g, ubar, u_g, 0, u_f, rho_s, alpha, a0, star, boundary, boundaryg);

    double ts[20];
    for (int i=0; i<20; i++) {
        ts[i] = 1e6*year/20. * i + 1e6*year/20.;
    }

    double dt_CFL = 0.2*calc_dt(g, nu);
    std::cout << dt_CFL << "\n";
    double t = 0., dt;

    std::ofstream f((dir / "1Drun.dat"));
    f << "#" << g.NR + 2*g.Nghost << "\n" ;
    f << "#R Sig_g Sig_d ubar u_g \n" ;

    for (int i=0; i<g.NR+2*g.Nghost; i++) {
        f << g.Rc(i) << " " << Sig_g[i] << " " << Sig_d[i] << " " << ubar[i] << " " << u_g[i] << "\n";
    }

    f.close();
    int count = 0;
    std::chrono::_V2::system_clock::time_point start,stop;
    start = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds duration;
    double yps;

    for (double ti : ts) {
        while (t<ti) { 
            if (!(count%10000)) {
                std::cout << "t = " << t/year << " years\n";
                std::cout << "dt = " << dt_CFL/year << " years\n";
                stop = std::chrono::high_resolution_clock::now();
                yps = (t/year) / std::chrono::duration_cast<std::chrono::seconds>(stop - start).count();
                std::cout << "Years per second: " << yps << "\n";
            }

            if (count < 100) {dt_CFL = 1e5;}
            dt = std::min(dt_CFL, ti-t);
            if (count > 0) {
                calculate_ubar(g, Sig_d, Sig_g, ubar, u_g, t, u_f, rho_s, alpha, a0, star, boundary, boundaryg);
            }
            update_dust_sigma(g, Sig_d, Sig_g, ubar, nu, dt, boundary);
            update_gas_sources(g, Sig_g, Sigdot_w, dt, boundaryg,1e-10);
            update_gas_sigma(g, Sig_g, dt, nu, boundaryg,1e-10);
            update_gas_vel(g, Sig_g, u_g, alpha, star);
            t += dt;
            dt_CFL = 0.2*calc_dt(g, nu); //compute_CFL(g, ubar, nu, 0.2, 0.1);
            count += 1;
        }
        f.open((dir / "1Drun.dat"), std::ifstream::app);
        for (int i=0; i<g.NR+2*g.Nghost; i++) {
            f << g.Rc(i) << " " << Sig_g[i] << " " << Sig_d[i] << " " << ubar[i] << " " << u_g[i] << "\n";
        }
        f.close();
    }
    
    stop = std::chrono::high_resolution_clock::now();
    std::cout << count << " timesteps\n" ;

    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time taken: " << duration.count()/1.e6 << " seconds" << std::endl;  
    std::cout << "Av time per step: " << duration.count()/(1.e3*count) << " milliseconds" << std::endl;  

    return 0; 
}