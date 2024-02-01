#include <iostream>
#include <fstream>
#include <cmath>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <filesystem>

#include "grid.h"
#include "field.h"
#include "cuda_array.h"
#include "advection.h"
#include "constants.h"
#include "dustdynamics.h"
#include "sources.h"
#include "file_io.h"

int main() {

    int Ns[3] = {128,256,512};//,1024};

    for (int i=0; i<3; i++) {

        std::filesystem::path dir = std::string("../outputs/adv_diff/run_"+std::to_string(Ns[i]));
        std::filesystem::create_directories(dir);

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
        g.write_grid(dir);

        std::cout << "N = " << g.NR << "\n";

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
        
        write_prims(dir, 0, g, Ws, Ws_gas);

        int boundary = BoundaryFlags::open_R_inner | BoundaryFlags::open_R_outer | BoundaryFlags::open_Z_inner | BoundaryFlags::open_Z_outer;

        dyn.set_boundaries(boundary);
        auto start = std::chrono::high_resolution_clock::now();
        int nsteps=0;
        for (double ti : ts) {

            while (t < ti) {
                dt = std::min(dt_cfl, ti-t);
                dyn(g, Ws, Ws_gas, dt);
                t += dt;
                dt_cfl = dyn.get_CFL_limit(g, Ws, Ws_gas);
                nsteps+=1;
            }

            write_prims(dir, 1, g, Ws, Ws_gas);

        }
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "Time taken by function: " << duration.count()/1.e6 << " seconds" << std::endl;     
        std::cout << "N_steps = " << nsteps << "\n\n" ;
    }

    return 0;
}