#include <iostream>
#include <string>

#include "constants.h"
#include "dustdynamics1D.h"
#include "dustdynamics.h"
#include "gas1d.h"
#include "file_io.h"

double rhos_bench[] = {55.291079, 55.291079, 55.293968, 52.791557, 50.410654, 48.144727, 45.987549, 43.933287, 41.976471, 40.111976, 
                        38.334996, 36.641025, 35.025836, 33.485464, 32.016188, 30.614514, 29.277162, 28.001049, 26.783274, 25.621112, 
                        24.511998, 23.453514, 22.443384, 21.479465, 20.559733, 19.682283, 18.845319, 18.047148, 17.286176, 16.560904, 
                        15.869926, 15.211926, 14.585676, 13.990039, 13.423968, 12.886509, 12.376811, 11.894127, 11.437831, 11.007431, 
                        10.602591, 10.223151, 9.8691613, 9.5409081, 9.2389267, 8.9639842, 8.7169841, 8.4987288, 8.3094477, 8.1479868, 
                        8.0106177, 7.8895885, 7.7716651, 7.6367436, 7.4568338, 7.1970364, 6.8203629, 6.2967686, 5.6148415, 4.7922816, 
                        3.8796769, 2.9531759, 2.0962942, 1.3772787, 0.83192415, 0.45928482, 0.23057223, 0.10480096, 0.042967603, 
                        0.015839891, 0.0052360673, 0.0015482986, 0.00040866367, 9.609116e-05, 2.0091105e-05, 3.7286783e-06, 6.1316586e-07, 
                        8.9189536e-08, 1.1454917e-08, 1.2966278e-09, 1.2909343e-10, 1.1269867e-11, 8.5525427e-13, 5.8219434e-14, 2.1933721e-14, 
                        1.9687363e-14, 1.7619853e-14, 1.5721662e-14, 1.3983483e-14, 1.2396225e-14, 1.0951014e-14, 9.6391876e-15, 8.4523037e-15, 
                        7.3821397e-15, 6.4206999e-15, 5.560223e-15, 4.7931909e-15, 4.1123386e-15, 3.5106648e-15, 2.9814435e-15, 2.5182354e-15, 
                        2.1148984e-15, 2.1148984e-15, 2.1148984e-15};


void set_up(Grid& g, Field3D<Prims1D>& W_d, Field<Prims1D>& W_g, Field<double>& T, Field<double>& cs, 
            CudaArray<double>& nu, Field<double>& alpha2D, Field3D<double>& D, double alpha, Star& star) {

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
    for (int i=0; i<g.NR+2*g.Nghost; i++) {
        for (int k=0; k<W_d.Nd; k++) {
            W_d(i,g.Nghost,k).Sig = dtg*W_g(i,g.Nghost).Sig;
            W_d(i,g.Nghost,k).v_R = W_g(i,g.Nghost).v_R;
            D(i,g.Nghost,k) =  W_g(i,g.Nghost).Sig * nu[i]/Sc;
        }
    }

}


int main() {

    std::cout << "Test 1D advection-diffusion... ";
    std::cout.flush() ;

    Grid::params p;
    p.NR = 100;
    p.Nphi = 1;
    p.Nghost = 2;

    p.Rmin = 1.*au;
    p.Rmax = 100.*au;
    p.R_spacing = RadialSpacing::log;

    p.theta_min = -M_PI / 20.;
    p.theta_max = M_PI / 20.;

    Grid g(p);
    SizeGrid sizes(1e-1, 1, 1, 1);

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

    set_up(g, W_d, W_g, T, cs, nu, alpha2D, D, 1.e-3, star);

    double floor = 1e-15;
    double gas_floor = 1e-20;

    double t = 0, dt;
    const int ntimes = 1;  
    double ts[ntimes] = {1e5*year};

    DustDyn1D dyn(D, cs, star, sizes, nu, 0.4, 0.1, floor, gas_floor);
    int boundary = BoundaryFlags::open_R_inner | BoundaryFlags::open_R_outer;
    dyn.set_boundaries(boundary);

    int count = 0;

    double dt_CFL = 1;

    for (double ti : ts) {

        while (t < ti) {  
            dt = std::min(dt_CFL, ti-t);
            dyn(g, W_d, W_g, dt);
            count += 1;
            t += dt;
            dt_CFL = std::min(dyn.get_CFL_limit(g, W_d, W_g),1.2*dt);

        }
    }
    
    double L2 = 0;
    for (int i=0; i<g.NR+2*g.Nghost; i++) {
        L2 += (std::pow(W_d(i,2,0).Sig-rhos_bench[i], 2.)/(g.NR+2*g.Nghost));
    }
    L2 = std::sqrt(L2);
    if (L2 <= 8.e-4) {printf("Pass.\n");}
    else {printf("\n\tL2 = %g, fail.\n", L2);}

    return 0;
}