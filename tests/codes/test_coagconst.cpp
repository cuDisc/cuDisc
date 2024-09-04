#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "grid.h"
#include "coagulation/coagulation.h"
#include "coagulation/integration.h"
#include "file_io.h"

void setup_IC(Grid &g, SizeGrid& sizes, Field3D<double>& rho) {
    int N = sizes.size() ;

    // An exponential distribution in (N/m) (approx)
    double allm = 0.0 ;
    for (int k = 0; k < N; ++k) {
        double m0 = sizes.edge_mass(k) ;
        double m1 = sizes.edge_mass(k+1) ;
        double m = sizes.centre_mass(k) ;

        for (int i=0; i < g.NR + 2*g.Nghost; i++)
            for (int j=0; j < g.NR + 2*g.Nghost; j++) {
              rho(i,j,k) = m*m * std::exp(-m) * (m1 - m0) ;
              if (i==0 && j==0)
                allm += rho(i,j,k) ;
            }
    }

    // Normalization
    for (int i=0; i < g.NR + 2*g.Nghost; i++)
        for (int j=0; j < g.NR + 2*g.Nghost; j++) 
            for (int k = 0; k < N; ++k)
                rho(i,j,k) /= allm ;
}

void save_grid(Grid& g, Field3D<double>& rho, std::string filename) {
    std::ofstream f(filename) ;

    for (int i=0; i < g.NR + 2*g.Nghost; i++)
        for (int j = 0; j < g.Nphi + 2*g.Nghost; j++) {
            for (int k =0; k < rho.Nd; k++)
                f << " " << rho(i, j, k) ;
            f << "\n" ; 
        }
}

int main() {

    std::filesystem::path path = __FILE__;
    path = (path.parent_path()).parent_path();
    std::filesystem::path dir = path / std::string("outputs/coag_const");
    std::filesystem::create_directories(dir);

    std::cout << "Output directory: " << dir  << "\n";

    // Set up a dummy grid
    Grid::params p ;
    p.Nghost = 0 ;
    p.NR = 4 ;
    p.Rmax = 1. ;
    p.Rmin = 0.1 ;
    p.R_spacing = RadialSpacing::log ;

    p.Nphi = 32 ;
    p.theta_min = 0 ;
    p.theta_max = M_PI / 4 ;

    Grid g(p) ;

    // Setup a size distribution
    double a0 = std::pow(3e-6/(4*M_PI), 1/3.) ;
    double a1 = std::pow(3e+9/(4*M_PI), 1/3.) ;
    SizeGrid sizes(a0, a1, 150) ;

    write_grids(dir, &g, &sizes);

    // Generate the initial conditions
    Field3D<double> rho = create_field3D<double>(g, 150) ;
    setup_IC(g, sizes, rho) ;

    Field<double> rho_g = create_field<double>(g) ;
    set_all(g,rho_g,1);

    // Create the kernel/rates
    BS32Integration<CoagulationRate<ConstantKernel, SimpleErosion>>
        coagulation_integrate(
            create_coagulation_rate(sizes, ConstantKernel(g), SimpleErosion())
        ) ;

    // Run the test

    std::vector<double> t_out = {0., 1., 10., 100., 1000.} ;
    double t = 0;
    int Nout = 0 ;
    double dt = 0;
    for (auto ti : t_out) {
        if (ti > t) 
            coagulation_integrate.integrate(g, rho, rho_g, ti-t, dt) ;
        t = ti ;

        // Save to file:
        std::stringstream fName ;
        fName << "coag_const_"  << Nout++ << ".txt" ;
        save_grid(g, rho, dir / (fName.str())) ;
    }

    return 0 ;
}