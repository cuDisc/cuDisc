#ifndef _FILE_IO_H_
#define _FILE_IO_H_

#include <iostream>
#include <limits>
#include <string>
#include <sstream>
#include <fstream>
#include "field.h"
#include "DSHARP_opacs.h"
#include "bins.h"

void write_grid(std::string folder, Grid &g) {

    g.write_grid(folder) ;
}

template<typename in_type>
void read_file(std::filesystem::path folder, in_type in, Field3D<Prims>& wd, Field<Prims>& wg, CudaArray<double>& Sig_g, 
                    Field<double> &T, Field3D<double> &J) {

    std::stringstream in_string ;
    in_string << in ;
    
    std::ifstream f(folder / ("dens_" + in_string.str()  + ".dat"), std::ios::binary);

    int NR, NZ, nspec, nbands;

    f.read((char*) &NR, sizeof(int));
    f.read((char*) &NZ, sizeof(int));
    f.read((char*) &nspec, sizeof(int));
    for (int i=0; i<NR; i++) {
        for (int j=0; j<NZ; j++) {
            for (int k=0; k<4; k++) {
                f.read((char*) &wg(i,j)[k], sizeof(double));
            }
            for (int k=0; k<nspec; k++) {
                for (int l=0; l<4; l++) {
                    f.read((char*) &wd(i,j,k)[l], sizeof(double));
                }
            }
        }
        f.read((char*) &Sig_g[i], sizeof(double));
    }  
    f.close();

    f.open(folder / ("temp_" + in_string.str() + ".dat"), std::ios::binary);
        
    f.read((char*) &NR, sizeof(int));
    f.read((char*) &NZ, sizeof(int));
    f.read((char*) &nbands, sizeof(int));
    for (int i=0; i < NR; i++) { 
        for (int j = 0; j < NZ; j++) {
            f.read((char*) &T(i,j), sizeof(double));
            for (int k=0; k<nbands; k++) {
                f.read((char*) &J(i,j,k), sizeof(double));
            }
        }
    }
    f.close();
}


template<typename out_type>
void write_file(std::filesystem::path folder, out_type out, Grid &g, Field3D<Prims>& wd, Field<Prims>& wg, CudaArray<double>& Sig_g, 
                    Field<double> &T, Field3D<double> &J) {

    std::stringstream out_string ;
    out_string << out ;
    
    std::ofstream f(folder / ("dens_" + out_string.str()  + ".dat"), std::ios::binary);

    int NR = g.NR+2*g.Nghost, NZ = g.Nphi+2*g.Nghost, nspec = wd.Nd, nbands = J.Nd;

    f.write((char*) &NR, sizeof(int));
    f.write((char*) &NZ, sizeof(int));
    f.write((char*) &nspec, sizeof(int));
    for (int i=0; i<g.NR+2*g.Nghost; i++) {
        for (int j=0; j<g.Nphi+2*g.Nghost; j++) {
            for (int k=0; k<4; k++) {
                f.write((char*) &wg(i,j)[k], sizeof(double));
            }
            for (int k=0; k<wd.Nd; k++) {
                for (int l=0; l<4; l++) {
                    f.write((char*) &wd(i,j,k)[l], sizeof(double));
                }
            }
        }
        f.write((char*) &Sig_g[i], sizeof(double));
    }  
    f.close();

    f.open(folder / ("temp_" + out_string.str() + ".dat"), std::ios::binary);
        
    f.write((char*) &NR, sizeof(int));
    f.write((char*) &NZ, sizeof(int));
    f.write((char*) &nbands, sizeof(int));
    for (int i=0; i < g.NR + 2*g.Nghost; i++) { 
        for (int j = 0; j < g.Nphi + 2*g.Nghost; j++) {
            f.write((char*) &T(i,j), sizeof(double));
            for (int k=0; k<J.Nd; k++) {
                f.write((char*) &J(i,j,k), sizeof(double));
            }
        }
    }
    f.close();
}

template<typename out_type>
void write_prims(std::filesystem::path folder, out_type out, Grid &g, Field3D<Prims>& wd, Field<Prims>& wg) {

    std::stringstream out_string ;
    out_string << out ;

    std::ofstream f(folder / ("dens_" + out_string.str() + ".dat"), std::ios::binary);

    int NR = g.NR+2*g.Nghost, NZ = g.Nphi+2*g.Nghost, nspec = wd.Nd;

    double Sig_g = std::numeric_limits<double>::quiet_NaN() ;

    f.write((char*) &NR, sizeof(int));
    f.write((char*) &NZ, sizeof(int));
    f.write((char*) &nspec, sizeof(int));
    for (int i=0; i<g.NR+2*g.Nghost; i++) {
        for (int j=0; j<g.Nphi+2*g.Nghost; j++) {
            for (int k=0; k<4; k++) {
            f.write((char*) &wg(i,j)[k], sizeof(double));
            }
            for (int k=0; k<wd.Nd; k++) {
                for (int l=0; l<4; l++) {
                    f.write((char*) &wd(i,j,k)[l], sizeof(double));
                }
            }
        }
        f.write((char*) &Sig_g, sizeof(double));
    }  
    f.close();
}

template<typename out_type>
void write_temp(std::filesystem::path folder, out_type out, Grid &g, Field<double> &T, Field3D<double> &J) {

    std::stringstream out_string ;
    out_string << out ;

    int NR = g.NR+2*g.Nghost, NZ = g.Nphi+2*g.Nghost, nbands = J.Nd;
    
    std::ofstream f(folder / ("temp_" + out_string.str() + ".dat"), std::ios::binary);
        
    f.write((char*) &NR, sizeof(int));
    f.write((char*) &NZ, sizeof(int));
    f.write((char*) &nbands, sizeof(int));
    for (int i=0; i < g.NR + 2*g.Nghost; i++) { 
        for (int j = 0; j < g.Nphi + 2*g.Nghost; j++) {
            f.write((char*) &T(i,j), sizeof(double));
            for (int k=0; k<J.Nd; k++) {
                f.write((char*) &J(i,j,k), sizeof(double));
            }
        }
    }
    f.close();
}

void write_restart_file(std::string filename, int count, double t, double dt, double t_coag, double t_temp, double dt_coag, double dt_1perc, double t_interp) {

    std::ofstream f(filename, std::ios::binary);

    f.write((char*) &count, sizeof(int));
    f.write((char*) &t, sizeof(double));
    f.write((char*) &dt, sizeof(double));
    f.write((char*) &t_coag, sizeof(double));
    f.write((char*) &t_temp, sizeof(double));
    f.write((char*) &dt_coag, sizeof(double));
    f.write((char*) &dt_1perc, sizeof(double));
    f.write((char*) &t_interp, sizeof(double));

    f.close();
}

void read_restart_file(std::string filename, int& count, double& t, double& dt, double& t_coag, double& t_temp, double& dt_coag, double& dt_1perc, double& t_interp) {

    std::ifstream f(filename, std::ios::binary);

    f.read((char*) &count, sizeof(int));
    f.read((char*) &t, sizeof(double));
    f.read((char*) &dt, sizeof(double));
    f.read((char*) &t_coag, sizeof(double));
    f.read((char*) &t_temp, sizeof(double));
    f.read((char*) &dt_coag, sizeof(double));
    f.read((char*) &dt_1perc, sizeof(double));
    f.read((char*) &t_interp, sizeof(double));

    f.close();
}

void write_grids(std::filesystem::path folder, Grid* g, SizeGrid* s, DSHARP_opacs* o = nullptr, WavelengthBinner* b = nullptr) {

    if ((o != nullptr && b == nullptr) || (o == nullptr && b != nullptr)) {
        throw std::runtime_error("One of bins or opacity grids missing - both are required.\n"); 
    }

    g->write_grid(folder);
    s->write_grid(folder);

    if (o != nullptr) {
        o->write_interp(folder);
    }
    if (b != nullptr) {
        b->write_wle(folder);
    }
}
#endif