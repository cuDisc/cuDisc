#ifndef _FILE_IO_H_
#define _FILE_IO_H_

#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include "field.h"
#include "DSHARP_opacs.h"
#include "bins.h"


void read_data_txt(std::string filename, Field3D<Quants>& qd, Field<Quants>& wg) {

    std::ifstream file;
    
    file.open(filename);
    
    if (!file) {
        std::cout << "file not found error\n" ;
    }

    std::string line;
    
    std::string xstr;
    double x;

    int ps[4];

    getline(file,line);
    line.erase(0,1);
    std::stringstream params(line);
    for (int i=0; i<4; i++) {
        params >> ps[i];
    }
    getline(file, line);

    for (int t=0; t<ps[2]+1; t++) {
        for (int i=0; i<ps[0]; i++) {
            for (int j=0; j<ps[1]; j++) {
                for (int n=0; n<6+ps[3]; n++) {
                    file >> xstr;
                    x = ::atof(xstr.c_str());
                    if (n>1 && n<=5) {
                        wg(i,j)[n-2] = x;
                    }

                    if (n>5) {
                        int k = int((n-6));
                        qd(i,j,k).rho = x;
                        file >> xstr;
                        x = ::atof(xstr.c_str());
                        qd(i,j,k).mom_R = x;
                        file >> xstr;
                        x = ::atof(xstr.c_str());
                        qd(i,j,k).amom_phi = x;
                        file >> xstr;
                        x = ::atof(xstr.c_str());
                        qd(i,j,k).mom_Z = x;
                    }       
                }
            }
        }
    }
}

void read_temp_txt(std::string filename, Field<double>& T, Field3D<double>& J, int nbands, int ntimes) {

    std::ifstream file;
    
    file.open(filename);
    
    if (!file) {
        std::cout << "file not found error\n" ;
    }

    std::string line;
    
    std::string xstr;
    double x;

    int ps[4];

    getline(file,line);
    line.erase(0,1);
    std::stringstream params(line);
    for (int i=0; i<2; i++) {
        params >> ps[i];
    }
    getline(file, line);
    getline(file, line);

    for (int t=0; t<ntimes; t++) {
        for (int i=0; i<ps[0]; i++) {
            for (int j=0; j<ps[1]; j++) {
                for (int n=0; n<5+nbands; n++) {
                    file >> xstr;
                    x = ::atof(xstr.c_str());
                    if (n==4) {
                       T(i,j) = x;
                    }

                    if (n>4) {
                        J(i,j,n-5) = x;
                    }       
                }
            }
        }
    }
}

void read_temp(std::string filename, Field<double>& T, Field3D<double>& J, int nbands, int ntimes) {

    std::ifstream f_tempin(filename, std::ios::binary);

    if (!f_tempin) {
        std::cout << "file not found error\n" ;
    }

    int NR, NZ, nb;
    f_tempin.read((char*) &NR, sizeof(int));
    f_tempin.read((char*) &NZ, sizeof(int));
    f_tempin.read((char*) &nb, sizeof(int));

    for (int t=0; t<ntimes; t++) {
        for (int i=0; i < NR; i++) { 
            for (int j = 0; j < NZ; j++) {
                f_tempin.read((char*) &T(i,j), sizeof(double));
                for (int n=0; n<nbands; n++) {
                    f_tempin.read((char*) &J(i,j,n), sizeof(double));
                }    
            }
        }
    }

    f_tempin.close();
}

void read_density(std::string filename, Field3D<Quants>& qd, Field<Quants>& wg, int ntimes) {

    std::ifstream f_in(filename, std::ios::binary);

    if (!f_in) {
        std::cout << "file not found error\n" ;
    }

    int NR, NZ, Nt, Nq;
    f_in.read((char*) &NR, sizeof(int));
    f_in.read((char*) &NZ, sizeof(int));
    f_in.read((char*) &Nt, sizeof(int));
    f_in.read((char*) &Nq, sizeof(int));

    for (int t=0; t<ntimes; t++) {
        for (int i=0; i < NR; i++) { 
            for (int j = 0; j < NZ; j++) {
                for (int k=0; k<4; k++) {
                    f_in.read((char*) &wg(i,j)[k], sizeof(double));
                }
                for (int k=0; k<Nq; k++) {
                    for (int l=0; l<4; l++) {
                        f_in.read((char*) &qd(i,j,k)[l], sizeof(double));
                    }
                }    
            }
        }
    }

    f_in.close();
}
void read_density(std::string filename, Field3D<Prims>& qd, Field<Prims>& wg, CudaArray<double>& Sig_g, int ntimes) {

    std::ifstream f_in(filename, std::ios::binary);

    if (!f_in) {
        std::cout << "file not found error\n" ;
    }

    int NR, NZ, Nt, Nq;
    f_in.read((char*) &NR, sizeof(int));
    f_in.read((char*) &NZ, sizeof(int));
    f_in.read((char*) &Nt, sizeof(int));
    f_in.read((char*) &Nq, sizeof(int));

    for (int t=0; t<ntimes; t++) {
        for (int i=0; i < NR; i++) { 
            for (int j = 0; j < NZ; j++) {
                for (int k=0; k<4; k++) {
                    f_in.read((char*) &wg(i,j)[k], sizeof(double));
                }
                for (int k=0; k<Nq; k++) {
                    for (int l=0; l<4; l++) {
                        f_in.read((char*) &qd(i,j,k)[l], sizeof(double));
                    }
                }    
            }
            f_in.read((char*) &Sig_g[i], sizeof(double));
        }
    }

    f_in.close();
}


void write_grid(std::string folder, Grid &g) {

    g.write_grid(folder) ;
}



void write_init_dens(std::string filename, Grid &g, Field3D<Quants>& qd, Field<Quants>& wg, int ntimes, int nspec) {

    std::ofstream f(filename, std::ios::binary);

    int NR = g.NR+2*g.Nghost, NZ = g.Nphi+2*g.Nghost;

    f.write((char*) &NR, sizeof(int));
    f.write((char*) &NZ, sizeof(int));
    f.write((char*) &ntimes, sizeof(int));
    f.write((char*) &nspec, sizeof(int));
    for (int i=0; i < g.NR + 2*g.Nghost; i++) { 
        for (int j = 0; j < g.Nphi + 2*g.Nghost; j++) {
            for (int k=0; k<4; k++) {
                f.write((char*) &wg(i,j)[k], sizeof(double));
            }
            for (int k=0; k<nspec; k++) {
                for (int l=0; l<4; l++) {
                    f.write((char*) &qd(i,j,k)[l], sizeof(double));
                }
            }
        }
    }
    f.close();
}
void write_init_dens(std::string filename, Grid &g, Field3D<Quants>& qd, Field<Quants>& wg, CudaArray<double>& Sig_g, int ntimes, int nspec) {

    std::ofstream f(filename, std::ios::binary);

    int NR = g.NR+2*g.Nghost, NZ = g.Nphi+2*g.Nghost;

    f.write((char*) &NR, sizeof(int));
    f.write((char*) &NZ, sizeof(int));
    f.write((char*) &ntimes, sizeof(int));
    f.write((char*) &nspec, sizeof(int));
    for (int i=0; i < g.NR + 2*g.Nghost; i++) { 
        for (int j = 0; j < g.Nphi + 2*g.Nghost; j++) {
            for (int k=0; k<4; k++) {
                f.write((char*) &wg(i,j)[k], sizeof(double));
            }
            for (int k=0; k<nspec; k++) {
                for (int l=0; l<4; l++) {
                    f.write((char*) &qd(i,j,k)[l], sizeof(double));
                }
            }
        }
        f.write((char*) &Sig_g[i], sizeof(double));
    }
    f.close();
}

void write_init_dens(std::string filename, Grid &g, Field3D<Prims>& qd, Field<Prims>& wg, CudaArray<double>& Sig_g, int ntimes, int nspec) {

    std::ofstream f(filename, std::ios::binary);

    int NR = g.NR+2*g.Nghost, NZ = g.Nphi+2*g.Nghost;

    f.write((char*) &NR, sizeof(int));
    f.write((char*) &NZ, sizeof(int));
    f.write((char*) &ntimes, sizeof(int));
    f.write((char*) &nspec, sizeof(int));
    for (int i=0; i < g.NR + 2*g.Nghost; i++) { 
        for (int j = 0; j < g.Nphi + 2*g.Nghost; j++) {
            for (int k=0; k<4; k++) {
                f.write((char*) &wg(i,j)[k], sizeof(double));
            }
            for (int k=0; k<nspec; k++) {
                for (int l=0; l<4; l++) {
                    f.write((char*) &qd(i,j,k)[l], sizeof(double));
                }
            }
        }
        f.write((char*) &Sig_g[i], sizeof(double));
    }
    f.close();
}

void write_init_temp(std::string filename, Grid &g, Field<double> &T, Field3D<double> &J) {

    std::ofstream f_temp(filename, std::ios::binary);

    int NR = g.NR+2*g.Nghost, NZ = g.Nphi+2*g.Nghost, Nbands = J.Nd;
    f_temp.write((char*) &NR, sizeof(int));
    f_temp.write((char*) &NZ, sizeof(int));
    f_temp.write((char*) &Nbands, sizeof(int));
    for (int i=0; i < g.NR + 2*g.Nghost; i++) { 
        for (int j = 0; j < g.Nphi + 2*g.Nghost; j++) {
            f_temp.write((char*) &T(i,j), sizeof(double));
            for (int k=0; k<J.Nd; k++) {
                f_temp.write((char*) &J(i,j,k), sizeof(double));
            }
        }
    }
    f_temp.close();
}

void append_temp(std::string filename, Grid &g, Field<double> &T, Field3D<double> &J) {
        
        std::ofstream f_temp(filename, std::ios::binary | std::fstream::app);

        for (int i=0; i < g.NR + 2*g.Nghost; i++) { 
            for (int j = 0; j < g.Nphi + 2*g.Nghost; j++) {
                f_temp.write((char*) &T(i,j), sizeof(double));
                for (int k=0; k<J.Nd; k++) {
                    f_temp.write((char*) &J(i,j,k), sizeof(double));
                }
            }
        }
        f_temp.close();
}

void append_dens(std::string filename, Grid &g, Field3D<Prims>& qd, Field<Prims>& wg) {

        std::ofstream f(filename, std::ios::binary | std::fstream::app);

        for (int i=0; i<g.NR+2*g.Nghost; i++) {
            for (int j=0; j<g.Nphi+2*g.Nghost; j++) {
                for (int k=0; k<4; k++) {
                f.write((char*) &wg(i,j)[k], sizeof(double));
                }
                for (int k=0; k<qd.Nd; k++) {
                    for (int l=0; l<4; l++) {
                        f.write((char*) &qd(i,j,k)[l], sizeof(double));
                    }
                }
            }
        }  
        f.close();
}
void append_dens(std::string filename, Grid &g, Field3D<Prims>& qd, Field<Prims>& wg, CudaArray<double>& Sig_g) {

        std::ofstream f(filename, std::ios::binary | std::fstream::app);

        for (int i=0; i<g.NR+2*g.Nghost; i++) {
            for (int j=0; j<g.Nphi+2*g.Nghost; j++) {
                for (int k=0; k<4; k++) {
                f.write((char*) &wg(i,j)[k], sizeof(double));
                }
                for (int k=0; k<qd.Nd; k++) {
                    for (int l=0; l<4; l++) {
                        f.write((char*) &qd(i,j,k)[l], sizeof(double));
                    }
                }
            }
            f.write((char*) &Sig_g[i], sizeof(double));
        }  
        f.close();
}

template<typename out_type>
void write_file(std::filesystem::path folder, out_type out, Grid &g, Field3D<Prims>& qd, Field<Prims>& wg, CudaArray<double>& Sig_g, 
                    Field<double> &T, Field3D<double> &J) {

    std::stringstream out_string ;
    out_string << out ;
    
    std::ofstream f(folder / ("dens_" + out_string.str()  + ".dat"), std::ios::binary);

    int NR = g.NR+2*g.Nghost, NZ = g.Nphi+2*g.Nghost, nspec = qd.Nd, nbands = J.Nd;

    f.write((char*) &NR, sizeof(int));
    f.write((char*) &NZ, sizeof(int));
    f.write((char*) &nspec, sizeof(int));
    for (int i=0; i<g.NR+2*g.Nghost; i++) {
        for (int j=0; j<g.Nphi+2*g.Nghost; j++) {
            for (int k=0; k<4; k++) {
            f.write((char*) &wg(i,j)[k], sizeof(double));
            }
            for (int k=0; k<qd.Nd; k++) {
                for (int l=0; l<4; l++) {
                    f.write((char*) &qd(i,j,k)[l], sizeof(double));
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

    std::ofstream f(folder / ("prims_" + out_string.str() + ".dat"), std::ios::binary);

    int NR = g.NR+2*g.Nghost, NZ = g.Nphi+2*g.Nghost, nspec = wd.Nd;

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

void read_restart_quants(std::string folder, Grid &, Field3D<Prims>& qd, Field<Prims>& wg, CudaArray<double>& Sig_g, 
                    Field<double> &T, Field3D<double> &J) {

    std::ifstream f(folder + "/dens_restart.dat", std::ios::binary);

    int NR, NZ, Nq, Nbands;
    f.read((char*) &NR, sizeof(int));
    f.read((char*) &NZ, sizeof(int));
    f.read((char*) &Nq, sizeof(int));

    for (int i=0; i < NR; i++) { 
        for (int j = 0; j < NZ; j++) {
            for (int k=0; k<4; k++) {
                f.read((char*) &wg(i,j)[k], sizeof(double));
            }
            for (int k=0; k<Nq; k++) {
                for (int l=0; l<4; l++) {
                    f.read((char*) &qd(i,j,k)[l], sizeof(double));
                }
            }    
        }
        f.read((char*) &Sig_g[i], sizeof(double));
    }
    
    f.close();

    f.open(folder + "/temp_restart.dat", std::ios::binary);
        
    f.read((char*) &NR, sizeof(int));
    f.read((char*) &NZ, sizeof(int));
    f.read((char*) &Nbands, sizeof(int));

    for (int i=0; i < NR; i++) { 
        for (int j = 0; j < NZ; j++) {
            f.read((char*) &T(i,j), sizeof(double));
            for (int n=0; n<Nbands; n++) {
                f.read((char*) &J(i,j,n), sizeof(double));
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