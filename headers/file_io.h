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
#include "dustdynamics1D.h"

void write_grid(std::string folder, Grid &g) {

    g.write_grid(folder) ;
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



/* write_prims
 * 
 * Store the gas and dust primitive variables (density, velocities), including the gas
 * surface density.
 * 
 *  Data is written to the file "folder/dens_out.dat" and readable by fileIO.py
 *  Arguments:
 *      folder : Folder to store data in
 *      out : Label string for snapshot (typically a number or restart).
 *      g : Grid object for simulation
 *      wd : Dust primitive data
 *      wg : Gas primitive data
 *      Sig_g : Gas surface density
*/
template<typename out_type>
void write_prims(std::filesystem::path folder, out_type out, Grid &g, Field3D<Prims>& wd, Field<Prims>& wg, 
                 CudaArray<double>& Sig_g) {

    std::stringstream out_string ;
    out_string << out ;

    std::ofstream f(folder / ("dens_" + out_string.str() + ".dat"), std::ios::binary);

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
       f.write((char*) &Sig_g[i], sizeof(double));
    }  
    f.close();
}

/* write_prims
 * 
 * Store the gas and dust primitive variables (density, velocities), without the gas
 * surface density. The surface density data on disk will be replaced by NaNs. 
 * 
 *  Data is written to the file "folder/dens_out.dat" and readable by fileIO.py
 *  Arguments:
 *      folder : Folder to store data in
 *      out : Label string for snapshot (typically a number or restart).
 *      g : Grid object for simulation
 *      wd : Dust primitive data
 *      wg : Gas primitive data
*/
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
void write_prims1D(std::filesystem::path folder, out_type out, Grid &g, Field3D<Prims1D>& wd, Field<Prims1D>& wg) {

    std::stringstream out_string ;
    out_string << out ;

    std::ofstream f(folder / ("dens1D_" + out_string.str() + ".dat"), std::ios::binary);

    int NR = g.NR+2*g.Nghost, nspec = wd.Nd;

    f.write((char*) &NR, sizeof(int));
    f.write((char*) &nspec, sizeof(int));
    for (int i=0; i<g.NR+2*g.Nghost; i++) {
        for (int k=0; k<2; k++) {
            f.write((char*) &wg(i,g.Nghost)[k], sizeof(double));
        }
        for (int k=0; k<wd.Nd; k++) {
            for (int l=0; l<2; l++) {
                f.write((char*) &wd(i,g.Nghost,k)[l], sizeof(double));
            }
        }
    }  
    f.close();
}

/* read_prims
 * 
 * Read the gas and dust primitive variables (density, velocities), without the gas
 * surface density. 
 * 
 *  Data is read from the file "folder/dens_out.dat".
 *  Arguments:
 *      folder : Folder to store data in
 *      out : Label string for snapshot (typically a number or restart).
 *      g : Grid object for simulation
 *      wd : Dust primitive data
 *      wg : Gas primitive data
*/
template<typename snap_type>
void read_prims(std::filesystem::path folder, snap_type snap, Field3D<Prims>& wd, Field<Prims>& wg, 
                 CudaArray<double>& Sig_g) {
    
    std::stringstream snap_string ;
    snap_string << snap ;

    std::ifstream f(folder / ("dens_" + snap_string.str() + ".dat"), std::ios::binary);

    int NR, NZ, Nq;
    f.read((char*) &NR, sizeof(int));
    f.read((char*) &NZ, sizeof(int));
    f.read((char*) &Nq, sizeof(int));

    if (wd.Nd != Nq) {
        std::stringstream ss ;
        ss << "Number of dust species on file (" << Nq << ") does not match the"
           << "number in the Field3D wd (" << wd.Nd << ")";
        throw std::invalid_argument(ss.str()) ;
    }

    for (int i=0; i < NR; i++) { 
        for (int j = 0; j < NZ; j++) {
            for (int k=0; k<4; k++) {
                f.read((char*) &wg(i,j)[k], sizeof(double));
            }
            for (int k=0; k<Nq; k++) {
                for (int l=0; l<4; l++) {
                    f.read((char*) &wd(i,j,k)[l], sizeof(double));
                }
            }    
        }
        f.read((char*) &Sig_g[i], sizeof(double));
    }
    
    f.close();
}

template<typename out_type>
void read_prims1D(std::filesystem::path folder, out_type out, Field3D<Prims1D>& wd, Field<Prims1D>& wg) {

    std::stringstream out_string ;
    out_string << out ;

    std::ifstream f(folder / ("dens1D_" + out_string.str() + ".dat"), std::ios::binary);

    int NR, nspec;

    f.read((char*) &NR, sizeof(int));
    f.read((char*) &nspec, sizeof(int));
    for (int i=0; i<NR; i++) {
        for (int k=0; k<2; k++) {
            f.read((char*) &wg(i,2)[k], sizeof(double));
        }
        for (int k=0; k<nspec; k++) {
            for (int l=0; l<2; l++) {
                f.read((char*) &wd(i,2,k)[l], sizeof(double));
            }
        }
    }  
    f.close();
}


/* write_temp
 * 
 * Store the temperature and radiation field information.
 * 
 *  Data is written to the file "folder/temp_out.dat" and readable by fileIO.py
 *  Arguments:
 *      folder : Folder to store data in
 *      out : Label string for snapshot (typically a number or restart).
 *      g : Grid object for simulation
 *      T : Temperature data
 *      J : Radiation field information
*/
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

/* write_temp
 * 
 * Store the temperature and radiation field information.
 * 
 *  Data is written to the file "folder/temp_out.dat" and readable by fileIO.py
 *  Arguments:
 *      folder : Folder to store data in
 *      out : Label string for snapshot (typically a number or restart).
 *      g : Grid object for simulation
 *      T : Temperature data
 *      J : Radiation field information
*/
template<typename out_type>
void write_temp(std::filesystem::path folder, out_type out, Grid &g, Field<double> &T, Field<double> &J) {

    std::stringstream out_string ;
    out_string << out ;

    int NR = g.NR+2*g.Nghost, NZ = g.Nphi+2*g.Nghost, nbands = 1;
    
    std::ofstream f(folder / ("temp_" + out_string.str() + ".dat"), std::ios::binary);
    
    f.write((char*) &NR, sizeof(int));
    f.write((char*) &NZ, sizeof(int));
    f.write((char*) &nbands, sizeof(int));
    for (int i=0; i < g.NR + 2*g.Nghost; i++) { 
        for (int j = 0; j < g.Nphi + 2*g.Nghost; j++) {
            f.write((char*) &T(i,j), sizeof(double));
            f.write((char*) &J(i,j), sizeof(double));
        }
    }
    f.close();
}

/* write_temp
 * 
 * Store the temperature information without the radiation field.
 * 
 *  Data is written to the file "folder/temp_out.dat" and readable by fileIO.py
 *  Arguments:
 *      folder : Folder to store data in
 *      out : Label string for snapshot (typically a number or restart).
 *      g : Grid object for simulation
 *      T : Temperature data
*/
template<typename out_type>
void write_temp(std::filesystem::path folder, out_type out, Grid &g, Field<double> &T) {

    std::stringstream out_string ;
    out_string << out ;

    int NR = g.NR+2*g.Nghost, NZ = g.Nphi+2*g.Nghost, nbands = 0 ;
    
    std::ofstream f(folder / ("temp_" + out_string.str() + ".dat"), std::ios::binary);
        
    f.write((char*) &NR, sizeof(int));
    f.write((char*) &NZ, sizeof(int));
    f.write((char*) &nbands, sizeof(int));
    for (int i=0; i < g.NR + 2*g.Nghost; i++) { 
        for (int j = 0; j < g.Nphi + 2*g.Nghost; j++) {
            f.write((char*) &T(i,j), sizeof(double));
        }
    }
    f.close();
}

/* read_temp
 * 
 * Read the temperature and radiation field information.
 * 
 *  Data is read from the file "folder/temp_out.dat".
 *  Arguments:
 *      folder : Folder to store data in
 *      out : Label string for snapshot (typically a number or restart).
 *      g : Grid object for simulation
 *      T : Temperature data
 *      J : Radiation field information
*/
template<typename snap_type>
void read_temp(std::filesystem::path folder, snap_type snap, Field<double> &T, Field3D<double> &J) {

    std::stringstream snap_string ;
    snap_string << snap ;

    int NR, NZ, Nbands ;
    
    std::ifstream f(folder / ("temp_" + snap_string.str() + ".dat"), std::ios::binary);
                
    f.read((char*) &NR, sizeof(int));
    f.read((char*) &NZ, sizeof(int));
    f.read((char*) &Nbands, sizeof(int));

    if (Nbands == 0) {
        std::cerr << "Warning: Nbands == 0, no radiation field information"
                  << " will be read" << std::endl ;
    }

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

/* read_temp
 * 
 * Read the temperature information without the radiation field.
 * 
 *  Data is read from the file "folder/temp_out.dat".
 *  Arguments:
 *      folder : Folder to store data in
 *      out : Label string for snapshot (typically a number or restart).
 *      g : Grid object for simulation
 *      T : Temperature data
*/
template<typename snap_type>
void read_temp(std::filesystem::path folder, snap_type snap, Field<double> &T) {

    std::stringstream snap_string ;
    snap_string << snap ;

    int NR, NZ, Nbands ;
    
    std::ifstream f(folder / ("temp_" + snap_string.str() + ".dat"), std::ios::binary);
                
    f.read((char*) &NR, sizeof(int));
    f.read((char*) &NZ, sizeof(int));
    f.read((char*) &Nbands, sizeof(int));

    if (Nbands > 0) {
        std::cerr << "Warning: Nbands > 0. The radiation field will be discarded."
                   << std::endl ;
    }

    double J ;
    for (int i=0; i < NR; i++) { 
        for (int j = 0; j < NZ; j++) {
            f.read((char*) &T(i,j), sizeof(double));
            for (int n=0; n<Nbands; n++) {
                f.read((char*) &J, sizeof(double));
            }    
        }
    }

    f.close();
}

/* write_file
 * 
 * Store the primitive data, temperature and radiation field information.
 * 
 *  Data is written to the files "folder/prim_out.dat" and "folder/temp_out.dat".
 *  The data is readable by fileIO.py
 * 
 *  See write_prims and write_temp for information about the arguments
*/
template<typename out_type>
void write_file(std::filesystem::path folder, out_type out, Grid &g, Field3D<Prims>& wd, Field<Prims>& wg,
                CudaArray<double>& Sig_g,  Field<double> &T, Field3D<double> &J) {

    // Get the output name once, to avoid any possible issues.
    std::stringstream out_ss ;
    out_ss << out ;    
    std::string out_str = out_ss.str() ;

    // Write the primtive data
    write_prims(folder, out_str, g, wd, wg, Sig_g) ;

    // Write temperature / radiation field
    write_temp(folder, out_str, g, T, J) ;
}

template<typename snap_type>
void read_file(std::string folder, snap_type snap, Field3D<Prims>& wd, Field<Prims>& wg,
               CudaArray<double>& Sig_g, Field<double> &T, Field3D<double> &J) {

    // Get the snapshot name once, to avoid any possible issues.
    std::stringstream snap_ss ;
    snap_ss << snap ;    
    std::string snap_str = snap_ss.str() ;

    // Read the primtive data
    read_prims(folder, snap_str, wd, wg, Sig_g) ;

    // Write temperature / radiation field
    read_temp(folder, snap_str,T, J) ;

}

/*
 * Copy file with name filename to filename.bak, if it exists
 *
 */

void backup_file(std::filesystem::path filename) {

    namespace fs = std::filesystem ;

    if (fs::exists(filename)) {

        fs::path backup = filename; 
        backup.replace_extension(".bak") ;
        
        if (fs::exists(backup)) {
            fs::remove(backup) ;
        }

        fs::rename(filename, backup) ;
    }
}

/* write_restart_quants / read_restart_quants 
 * 
 * Same as read_file/write_file but specificically for restart files. 

*/
void write_restart_quants(std::filesystem::path folder, Grid &g, Field3D<Prims>& wd, Field<Prims>& wg, CudaArray<double>& Sig_g, 
                          Field<double> &T, Field3D<double> &J) {
    
    // Backup existing files.
    backup_file(folder / ("dens_restart.dat")) ;
    backup_file(folder / ("temp_restart.dat")) ;

    write_file(folder, "restart", g, wd, wg, Sig_g, T, J) ;             
}
void read_restart_quants(std::filesystem::path folder, Field3D<Prims>& wd, Field<Prims>& wg, CudaArray<double>& Sig_g, 
                         Field<double> &T, Field3D<double> &J) {

    read_file(folder, "restart", wd, wg, Sig_g, T, J) ;       
}

void write_restart_prims(std::filesystem::path folder, Grid &g, Field3D<Prims>& wd, Field<Prims>& wg, CudaArray<double>& Sig_g) {
    
    // Backup existing files.
    backup_file(folder / ("dens_restart.dat")) ;

    write_prims(folder, "restart", g, wd, wg, Sig_g) ;             
}
void read_restart_prims(std::filesystem::path folder, Field3D<Prims>& wd, Field<Prims>& wg, CudaArray<double>& Sig_g) {

    read_prims(folder, "restart", wd, wg, Sig_g) ;       
}

void write_restart_file(std::string filename, int count, double t, double dt, double t_coag, double t_temp, double dt_coag, double dt_1perc, double t_interp) {

    backup_file(filename) ;
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

void write_restart_file(std::string filename, int count, double t, double dt, double t_i, double dt_i, double t_coag_i, double dt_coag_i, double t_coag_o, double dt_coag_o, double t_temp, double dt_1perc) {

    std::ofstream f(filename, std::ios::binary);

    f.write((char*) &count, sizeof(int));
    f.write((char*) &t, sizeof(double));
    f.write((char*) &dt, sizeof(double));
    f.write((char*) &t_i, sizeof(double));
    f.write((char*) &dt_i, sizeof(double));
    f.write((char*) &t_coag_i, sizeof(double));
    f.write((char*) &dt_coag_i, sizeof(double));
    f.write((char*) &t_coag_o, sizeof(double));
    f.write((char*) &dt_coag_o, sizeof(double));
    f.write((char*) &t_temp, sizeof(double));
    f.write((char*) &dt_1perc, sizeof(double));

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

void read_restart_file(std::string filename, int& count, double& t, double& dt, double& t_i, double& dt_i, double& t_coag_i, double& dt_coag_i, double& t_coag_o, double& dt_coag_o, double& t_temp, double& dt_1perc) {

    std::ifstream f(filename, std::ios::binary);

    f.read((char*) &count, sizeof(int));
    f.read((char*) &t, sizeof(double));
    f.read((char*) &dt, sizeof(double));
    f.read((char*) &t_i, sizeof(double));
    f.read((char*) &dt_i, sizeof(double));
    f.read((char*) &t_coag_i, sizeof(double));
    f.read((char*) &dt_coag_i, sizeof(double));
    f.read((char*) &t_coag_o, sizeof(double));
    f.read((char*) &dt_coag_o, sizeof(double));
    f.read((char*) &t_temp, sizeof(double));
    f.read((char*) &dt_1perc, sizeof(double));

    f.close();
}


#endif