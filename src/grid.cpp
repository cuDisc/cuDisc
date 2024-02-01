
#include <cmath>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <filesystem>

#include "grid.h"


double centroid(double R0, double R1) {
    double R02 = R0*R0 ;
    double R12 = R1*R1 ;

    return 0.75*(R12*R12 - R02*R02) / (R12*R1 - R02*R0) ;
}


/* Grid::Grid 
 *
 * Constructs grid factors based upon params 
 */
Grid::Grid(Grid::params p) 
{

    // Step 1: Radial grids:
    NR = p.NR ;
    Nghost = p.Nghost ;

    _Rc = make_CudaArray<double>(NR + 2*Nghost) ;
    _Re = make_CudaArray<double>(NR + 2*Nghost + 1) ;


    switch(p.R_spacing) {
      double dR ;
      case RadialSpacing::linear:
        dR = (p.Rmax - p.Rmin) / NR ;
        _Re[0] = p.Rmin - Nghost * dR ;
        for (int i=0; i < NR + 2*Nghost; i++) {
            _Re[i+1] = p.Rmin + (i+1 - Nghost) * dR ;
            _Rc[i] = centroid(_Re[i],  _Re[i+1]) ;
        }
        break ;
      case RadialSpacing::log:
        dR = std::log(p.Rmax/p.Rmin) / NR ;
        _Re[0] = p.Rmin / std::pow(1+dR, Nghost) ;
        for (int i=0; i < NR + 2*Nghost; i++) {
            _Re[i+1] = _Re[i] * (1 + dR) ;
            _Rc[i] = centroid(_Re[i],  _Re[i+1]) ;
        }
        break ; 
    } ;


    // Step 1: Theta grid:
    Nphi = p.Nphi ;

    _sin_theta_c = make_CudaArray<double>(Nphi + 2*Nghost) ;
    _cos_theta_c = make_CudaArray<double>(Nphi + 2*Nghost) ;
    _tan_theta_c = make_CudaArray<double>(Nphi + 2*Nghost) ;

    _sin_theta_e = make_CudaArray<double>(Nphi + 2*Nghost + 1) ;
    _cos_theta_e = make_CudaArray<double>(Nphi + 2*Nghost + 1) ;
    _tan_theta_e = make_CudaArray<double>(Nphi + 2*Nghost + 1) ;

    switch(p.theta_spacing) {
      double dt, th0, th1;

      case ThetaSpacing::linear:
        dt = (p.theta_max - p.theta_min) / Nphi ;
        th0 = p.theta_min - Nghost * dt ;

        _sin_theta_e[0] = std::sin(th0) ;
        _cos_theta_e[0] = std::cos(th0) ;
        _tan_theta_e[0] = std::tan(th0) ;
        for (int i=0; i < Nphi + 2*Nghost; i++) {
            th1 = p.theta_min + (i+1 - Nghost) * dt ;
            _sin_theta_e[i+1] = std::sin(th1) ;
            _cos_theta_e[i+1] = std::cos(th1) ;
            _tan_theta_e[i+1] = std::tan(th1) ;

            _tan_theta_c[i] = 0.5*(_tan_theta_e[i+1] + _tan_theta_e[i]) ;
            th0 = std::atan(_tan_theta_c[i]) ;

            _sin_theta_c[i] = std::sin(th0) ;
            _cos_theta_c[i] = std::cos(th0) ;

            th0 = th1 ;
        }
        break ;

      case ThetaSpacing::power:
        dt = (std::pow(p.theta_max, p.theta_power) - std::pow(p.theta_min, p.theta_power)) / Nphi ;
        if (std::pow(p.theta_min,p.theta_power) - Nghost * dt < 0.) {
            
            // th0 = p.theta_min*(1. + Nghost) - Nghost*std::pow(std::pow(p.theta_min,p.theta_power) + dt, 1./p.theta_power);

            double thoffset = std::pow(std::pow(p.theta_min,p.theta_power) + Nghost * dt, 1./p.theta_power);

            th0 = p.theta_min - thoffset;

            _sin_theta_e[0] = std::sin(th0) ;
            _cos_theta_e[0] = std::cos(th0) ;
            _tan_theta_e[0] = std::tan(th0) ;

            for (int i=0; i < Nghost; i++) {
                th1 = p.theta_min - std::pow(std::pow(p.theta_min,p.theta_power) + (Nghost-(i+1)) * dt, 1./p.theta_power);
                _sin_theta_e[i+1] = std::sin(th1) ;
                _cos_theta_e[i+1] = std::cos(th1) ;
                _tan_theta_e[i+1] = std::tan(th1) ;
                
                _tan_theta_c[i] = 0.5*(_tan_theta_e[i+1] + _tan_theta_e[i]) ;
                th0 = std::atan(_tan_theta_c[i]) ;

                _sin_theta_c[i] = std::sin(th0) ;
                _cos_theta_c[i] = std::cos(th0) ;

                th0 = th1 ;
            }
            for (int i=Nghost; i < Nphi + 2*Nghost; i++) {
                th1 = std::pow(std::pow(p.theta_min,p.theta_power) + (i+1-Nghost) * dt, 1./p.theta_power) ;
                _sin_theta_e[i+1] = std::sin(th1) ;
                _cos_theta_e[i+1] = std::cos(th1) ;
                _tan_theta_e[i+1] = std::tan(th1) ;

                _tan_theta_c[i] = 0.5*(_tan_theta_e[i+1] + _tan_theta_e[i]) ;
                th0 = std::atan(_tan_theta_c[i]) ;

                _sin_theta_c[i] = std::sin(th0) ;
                _cos_theta_c[i] = std::cos(th0) ;

                th0 = th1 ;
            }

        }
        else {

            th0 = std::pow(std::pow(p.theta_min,p.theta_power) - Nghost * dt, 1./p.theta_power) ;

            _sin_theta_e[0] = std::sin(th0) ;
            _cos_theta_e[0] = std::cos(th0) ;
            _tan_theta_e[0] = std::tan(th0) ;
            for (int i=0; i < Nphi + 2*Nghost; i++) {
                th1 = std::pow(std::pow(p.theta_min,p.theta_power) + (i+1 - Nghost) * dt, 1./p.theta_power) ;
                _sin_theta_e[i+1] = std::sin(th1) ;
                _cos_theta_e[i+1] = std::cos(th1) ;
                _tan_theta_e[i+1] = std::tan(th1) ;

                _tan_theta_c[i] = 0.5*(_tan_theta_e[i+1] + _tan_theta_e[i]) ;
                th0 = std::atan(_tan_theta_c[i]) ;

                _sin_theta_c[i] = std::sin(th0) ;
                _cos_theta_c[i] = std::cos(th0) ;

                th0 = th1 ;
            }
        }
        break ;
      
      case ThetaSpacing::subdiv:
        dt = (p.theta_max + p.theta_subdiv - 2*p.theta_min) / Nphi ;
        th0 = p.theta_min - Nghost * (dt/2.) ;
        int Nsub = 2*(p.theta_subdiv - p.theta_min) / dt ; 

        std::cout << "No. of cells in high-res region: " << Nsub << "\n";

        _sin_theta_e[0] = std::sin(th0) ;
        _cos_theta_e[0] = std::cos(th0) ;
        _tan_theta_e[0] = std::tan(th0) ;
        for (int i=0; i < Nphi + 2*Nghost; i++) {
            if (i < Nsub + Nghost) {
                th1 = p.theta_min + (i+1 - Nghost) * (dt/2.) ;
            }
            else {
                th1 = p.theta_min + Nsub * (dt/2.) + (i+1-Nsub-Nghost) * (dt) ;
            }
            _sin_theta_e[i+1] = std::sin(th1) ;
            _cos_theta_e[i+1] = std::cos(th1) ;
            _tan_theta_e[i+1] = std::tan(th1) ;

            _tan_theta_c[i] = 0.5*(_tan_theta_e[i+1] + _tan_theta_e[i]) ;
            th0 = std::atan(_tan_theta_c[i]) ;

            _sin_theta_c[i] = std::sin(th0) ;
            _cos_theta_c[i] = std::cos(th0) ;

            th0 = th1 ;
        }
        break ;
        
    } 
} 

Grid::Grid(int NR_, int Nphi_, int Nghost_, 
           CudaArray<double> R, CudaArray<double> phi) {

    NR = NR_ ;
    Nphi = Nphi_ ;
    Nghost = Nghost_ ;

    // Set up radial grid
    _Rc = make_CudaArray<double>(NR + 2*Nghost) ;
    _Re = std::move(R) ;

    for (int i=0; i < NR + 2*Nghost; i++) {
        _Rc[i] = centroid(_Re[i], _Re[i+1]) ;
    }

    // Setup azimuthal grid
    _sin_theta_c = make_CudaArray<double>(Nphi + 2*Nghost) ;
    _cos_theta_c = make_CudaArray<double>(Nphi + 2*Nghost) ;
    _tan_theta_c = make_CudaArray<double>(Nphi + 2*Nghost) ;

    _sin_theta_e = make_CudaArray<double>(Nphi + 2*Nghost + 1) ;
    _cos_theta_e = make_CudaArray<double>(Nphi + 2*Nghost + 1) ;
    _tan_theta_e = make_CudaArray<double>(Nphi + 2*Nghost + 1) ;

    _sin_theta_e[0] = std::sin(phi[0]) ;
    _cos_theta_e[0] = std::cos(phi[0]) ;
    _tan_theta_e[0] = std::tan(phi[0]) ;
    for (int i=0; i < Nphi + 2*Nghost; i++) {
        _sin_theta_e[i+1] = std::sin(phi[i+1]) ;
        _cos_theta_e[i+1] = std::cos(phi[i+1]) ;
        _tan_theta_e[i+1] = std::tan(phi[i+1]) ;

        _tan_theta_c[i] = 0.5*(_tan_theta_e[i+1] + _tan_theta_e[i]) ;
        double phic = std::atan(_tan_theta_c[i]) ;

        _sin_theta_c[i] = std::sin(phic) ;
        _cos_theta_c[i] = std::cos(phic) ;
    }

}

OrthGrid::OrthGrid(int _NR, int _NZ, int _Nghost, double Rmin, double Rmax, double Zmin, double Zmax) {

    Nghost = _Nghost;

    NR = _NR;
    NZ = _NZ;

    _Rc = make_CudaArray<double>(NR + 2*Nghost);
    _Re = make_CudaArray<double>(NR + 2*Nghost + 1);

    _Zc = make_CudaArray<double>(NZ + 2*Nghost);
    _Ze = make_CudaArray<double>(NZ + 2*Nghost + 1);

    double dR = (Rmax - Rmin)/NR;
    double dZ = (Zmax - Zmin)/NZ;
    
    _Re[0] = Rmin - Nghost * dR ;
    for (int i=0; i < NR + 2*Nghost; i++) {
        _Re[i+1] = Rmin + (i+1 - Nghost) * dR ;
        _Rc[i] = (_Re[i] + _Re[i+1])/2 ;
    }
    _Ze[0] = Zmin - Nghost * dZ ;
    for (int i=0; i < NZ + 2*Nghost; i++) {
        _Ze[i+1] = Zmin + (i+1 - Nghost) * dZ ;
        _Zc[i] = (_Ze[i] + _Ze[i+1])/2 ;
    }     

}

void Grid::write_grid(std::filesystem::path fname) const {
    
    std::ofstream fout(fname / ("2Dgrid.dat"), std::ios::binary) ;

    if (!fout) {
        throw std::runtime_error("Could not open file " + std::string(fname) + " for writing grid structure") ;
    }
    
    int NR_t = NR+2*Nghost ;
    int NZ_t = Nphi+2*Nghost;
    fout.write((char*) &NR_t, sizeof(int));
    fout.write((char*) &NZ_t, sizeof(int));

    fout.write((char*) &_Re[0], (NR_t+1) * sizeof(double)) ;
    fout.write((char*) &_Rc[0], ( NR_t ) * sizeof(double)) ;

    fout.write((char*) &_tan_theta_e[0], (NZ_t+1) * sizeof(double)) ;
    fout.write((char*) &_tan_theta_c[0], ( NZ_t ) * sizeof(double)) ;

    for (int i=0; i < NR_t; i++) { 
        for (int j = 0; j < NZ_t; j++) {
            double vol = volume(i,j);
            fout.write((char*) &vol, sizeof(double));
        }
    }
}