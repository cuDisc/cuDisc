#include <iostream>

#include "grid.h"

int main() {

    std::cout << "Test subgrids... ";
    std::cout.flush() ;

    bool pass = 1;

    Grid::params p;

    p.NR = 10;
    p.Nphi = 10;
    p.Nghost = 2;
    p.Rmin = 10;
    p.Rmax = 20.;
    p.theta_min = 0. ;
    p.theta_power = 0.75;
    p.theta_max = 1.;
    p.theta_spacing = ThetaSpacing::power;
    p.R_spacing = RadialSpacing::linear;

    Grid g(p);

    GridManager g_man(g);

    Grid g_in = g_man.add_subgrid(10,14);
    Grid g_out = g_man.add_subgrid(14,20);

    if (g_in.Re(g_in.NR+2*g.Nghost) != g.Re(g_in.NR+2*g.Nghost) && g_out.Re(0) != g_in.Re(g_in.NR+2*g.Nghost-4)) {
        std::cout << "Grid failure: edges incorrect.\n";
        pass = 0;
    }

    Field3D<double> F_main = create_field3D<double>(g,2);
    Field3D<double> F_sub_in = create_field3D<double>(g_in,2);
    Field3D<double> F_sub_out = create_field3D<double>(g_out,2);

    F_main(5,5,1) = 2;
    
    g_man.copy_to_subgrid(g_in, F_main, F_sub_in);

    if (F_main(5,5,1) != F_sub_in(5,5,1)) {
        std::cout << "Copy to subgrid failure.\n";
        pass = 0;
    };

    F_sub_in(5,5,1) = 3;

    g_man.copy_from_subgrid(g_in, F_main, F_sub_in);

    if (F_main(5,5,1) != F_sub_in(5,5,1)) {
        std::cout << "Copy from subgrid failure.\n";
        pass = 0;
    };

    if (pass) {std::cout << "Pass.\n";}
    else {std::cout << "Fail.\n";}

    return 0;
}