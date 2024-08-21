#include <iostream>
#include <string>

#include "constants.h"
#include "dustdynamics1D.h"
#include "dustdynamics.h"
#include "gas1d.h"
#include "file_io.h"

double rhos_bench[] = {52.8023, 52.8023, 52.809, 50.3457, 48.0204, 45.8187, 43.7319, 41.7524, 39.873, 38.0873, 36.3895, 34.7744, 33.237, 31.7729, 30.3781, 29.0489, 27.7817, 26.5734, 25.421, 24.3218, 23.2732, 22.2728, 21.3184, 20.4079, 19.5393, 18.7109, 17.9207, 17.1674, 16.4492, 15.7648, 15.113, 14.4923, 13.9017, 13.3401, 12.8065, 12.3, 11.8199, 11.3654, 10.936, 10.5311, 10.1506, 9.79437, 9.46239, 9.15502, 8.8728, 8.61653, 8.38708, 8.18528, 8.01136, 7.86425, 7.74043, 7.63261, 7.52834, 7.40872, 7.24728, 7.01072, 6.66321, 6.17477, 5.53225, 4.74968, 3.87271, 2.97278, 2.13075, 1.41523, 0.865142, 0.483826, 0.246234, 0.113525, 0.0472304, 0.0176716, 0.00592911, 0.00177929, 0.000476485, 0.000113629, 2.40832e-05, 4.52802e-06, 7.53839e-07, 1.10926e-07, 1.44006e-08, 1.64632e-09, 1.65455e-10, 1.46138e-11, 1.15462e-12, 1.01454e-13, 2.66285e-14, 1.96874e-14, 1.76199e-14, 1.57217e-14, 1.39835e-14, 1.23962e-14, 1.0951e-14, 9.63919e-15, 8.4523e-15, 7.38214e-15, 6.4207e-15, 5.56022e-15, 4.79319e-15, 4.11234e-15, 3.51066e-15, 2.98144e-15, 2.51824e-15, 2.1149e-15, 2.1149e-15, 2.1149e-15};


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
        W_g(i,g.Nghost).v_phi = v_k;
        
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
            W_d(i,g.Nghost,k).v_phi = W_g(i,g.Nghost).v_phi;
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
    calc_v_gas(g, W_g, cs, nu, star.GM, 1e-20);

    double floor = 1e-15;
    double gas_floor = 1e-20;

    double t = 0, dt;
    const int ntimes = 1;  
    double ts[ntimes] = {1e5*year};
    DustDyn1D dyn(D, cs, star, sizes, 2.4, 1.e-3, 0.4, 0.1, floor, gas_floor);  
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
    if (L2 <= 1.e-4) {printf("Pass.\n");}
    else {printf("\n\tL2 = %g, fail.\n", L2);}

    return 0;
}