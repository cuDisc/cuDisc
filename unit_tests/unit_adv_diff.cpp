#include <iostream>
#include <fstream>
#include <cmath>
#include <sstream>
#include <filesystem>

#include "grid.h"
#include "dustdynamics.h"
#include "sources.h"
#include "file_io.h"

double P(double x, double y, double t, double D, double A) {
    return A/(t) * std::exp(-std::pow(x-30.-(5.*(t-0.1)),2.)/(4.*D*t)) * std::exp(-std::pow(y-(2.*(t-0.1)),2.)/(4.*D*t));
}

int main() {

    std::cout << "Test advection-diffusion...\n";

    int Ns[2] = {128,256};
    double L2_bench[2] = {0.002114, 0.0004294};
    double slope_bench = 2.;
    double L2[2] = {0,0};

    for (int i=0; i<2; i++) {

        Grid::params p;
        p.NR = Ns[i];
        p.Nphi = Ns[i];
        p.Nghost = 2;

        p.Rmin = 10.;
        p.Rmax = 50.;
        p.theta_min = -M_PI / 6.;
        p.theta_max = M_PI / 6.;

        p.R_spacing = RadialSpacing::log ;

        Grid g(p); 

        g.set_coord_system(Coords::cart);

        Field3D<Prims> Ws = Field3D<Prims>(g.NR+2*g.Nghost, g.Nphi+2*g.Nghost, 1);
        Field<Prims> Ws_gas = Field<Prims>(g.NR+2*g.Nghost, g.Nphi+2*g.Nghost);
        Field3D<double> D = Field3D<double>(g.NR+2*g.Nghost, g.Nphi+2*g.Nghost, 1);

        Field<double> rho = Field<double>(g.NR+2*g.Nghost, g.Nphi+2*g.Nghost);
        Field<double> vR = Field<double>(g.NR+2*g.Nghost, g.Nphi+2*g.Nghost);
        Field<double> vphi = Field<double>(g.NR+2*g.Nghost, g.Nphi+2*g.Nghost);
        Field<double> vZ = Field<double>(g.NR+2*g.Nghost, g.Nphi+2*g.Nghost);
        Field<double> cs = Field<double>(g.NR+2*g.Nghost, g.Nphi+2*g.Nghost);

        set_all(g, cs, 1e100);

        for (int i=0; i<g.NR+2*g.Nghost; i++) {
            for (int j=0; j<g.Nphi+2*g.Nghost; j++) {
                rho(i,j) = 1e-10 + (1./std::pow(0.1,1)) * std::exp(-std::pow(g.Rc(i)-30.,2.)/(4.*0.1))*std::exp(-std::pow(g.Zc(i,j),2.)/(4.*0.1));
                vR(i,j) = 5.;
                vphi(i,j) = 0.;
                vZ(i,j) = 2.;
            }
        }


        for (int i=0; i<g.NR+2*g.Nghost; i++) {
            for (int j=0; j<g.Nphi+2*g.Nghost; j++) {
                Ws_gas(i,j).rho = 1.;
                Ws_gas(i,j).v_R = 0.;
                Ws_gas(i,j).v_phi = 0.;
                Ws_gas(i,j).v_Z = 0.;
            }
        }

        for (int i=0; i<g.NR+2*g.Nghost; i++) {
            for (int j=0; j<g.Nphi+2*g.Nghost; j++) {
                Ws(i,j,0).rho = rho(i,j);
                Ws(i,j,0).v_R = vR(i,j);
                Ws(i,j,0).v_phi = vphi(i,j);
                Ws(i,j,0).v_Z = vZ(i,j);

                D(i,j,0) = 1.;
            }
        }

        NoSources nosrc;
        DustDynamics dyn(D, cs, nosrc, 0.4, 0.1, 1e-10, 1e-1);

        double dt_cfl = dyn.get_CFL_limit(g, Ws, Ws_gas);

        double t = 0, dt;
        double ts[1] = {0.9};
        

        int boundary = BoundaryFlags::open_R_inner | BoundaryFlags::open_R_outer | BoundaryFlags::open_Z_inner | BoundaryFlags::open_Z_outer;

        dyn.set_boundaries(boundary);
        for (double ti : ts) {
            while (t < ti) {
                dt = std::min(dt_cfl, ti-t);
                dyn(g, Ws, Ws_gas, dt);
                t += dt;
                dt_cfl = dyn.get_CFL_limit(g, Ws, Ws_gas);
            }
        }

        for (int j=g.Nghost; j<g.NR+g.Nghost; j++) {
            for (int k=g.Nghost; k<g.Nphi+g.Nghost; k++) {
                L2[i] += (std::pow(Ws(j,k,0).rho-P(g.Rc(j), g.Zc(j,k), 1.,1.,1.), 2.)/(g.NR*g.Nphi));
            }
        }

        L2[i] = std::sqrt(L2[i]);

    }

    double slope = - std::log(L2[1]/L2[0]) / std::log(Ns[1]/Ns[0]);

    if (L2[0] <= L2_bench[0]) {std::cout << "\nL2_128 = " << L2[0] << ", pass\n";}
    else {std::cout << "\nL2_128 = " << L2[0] << ", fail\n";}
    if (L2[1] <= L2_bench[1]) {std::cout << "L2_256 = " << L2[1] << ", pass\n";;}
    else {std::cout << "L2_256 = " << L2[1] << ", fail\n";}
    if (slope >= slope_bench) {std::cout << "Slope = " << slope << ", pass\n\n";}
    else {std::cout << "Slope = " << slope << ", fail\n\n";}
    return 0;
}