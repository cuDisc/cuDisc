
#include <cmath>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <filesystem>

#include "dustdynamics.h"
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
    index = p.index ;

    _Rc = make_CudaArray<double>(NR + 2*Nghost) ;
    _Re = make_CudaArray<double>(NR + 2*Nghost + 1) ;
    _Ar = make_CudaArray<double>(NR + 2*Nghost + 1) ;
    _Az = make_CudaArray<double>(NR + 2*Nghost + 1) ;
    _V  = make_CudaArray<double>(NR + 2*Nghost + 1) ;


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
      case RadialSpacing::power:
        dR = (std::pow(p.Rmax,p.R_power) - std::pow(p.Rmin,p.R_power)) / NR ;
        _Re[0] = std::pow(std::pow(p.Rmin,p.R_power) - Nghost*dR, 1./p.R_power)  ;
        if (std::isnan(_Re[0])) {throw std::runtime_error("Inner radial edge < 0: either increase NR, decrease exponent, or increase Rmin") ;}
        for (int i=0; i < NR + 2*Nghost; i++) {
            _Re[i+1] = std::pow(std::pow(p.Rmin,p.R_power) + (i+1 - Nghost)*dR, 1./p.R_power);
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

    set_coord_system(coord_system) ;
} 

Grid::Grid(int NR_, int Nphi_, int Nghost_, 
           CudaArray<double> R, CudaArray<double> phi, int index_) {

    NR = NR_ ;
    Nphi = Nphi_ ;
    Nghost = Nghost_ ;
    index = index_;

    // Set up radial grid
    _Rc = make_CudaArray<double>(NR + 2*Nghost) ;
    _Re = std::move(R) ;
    _Ar = make_CudaArray<double>(NR + 2*Nghost + 1) ;
    _Az = make_CudaArray<double>(NR + 2*Nghost + 1) ;
    _V  = make_CudaArray<double>(NR + 2*Nghost + 1) ;

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

    set_coord_system(coord_system) ;
}

void Grid::set_coord_system(Coords system) {
    coord_system = system;
    
    switch(coord_system) {
    case Coords::cart:
        for (int i=0; i < NR + 2*Nghost+1; i++) {
            _Ar[i] = _Re[i] ;
            _Az[i] = _Re[i] ;
            _V[i]  = _Re[i]*_Re[i]/2. ;
        }
        break ;
      case Coords::cyl:
        for (int i=0; i < NR + 2*Nghost+1; i++) {
            _Ar[i] = _Re[i]*_Re[i] ;
            _Az[i] = _Re[i]*_Re[i]/2. ;
            _V[i]  = _Re[i]*_Re[i]*_Re[i]/3. ;
        }
        break ;
      default: __builtin_unreachable() ;
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
    
    std::ofstream fout;
    if (index == -1) {
        fout.open(fname / ("2Dgrid.dat"), std::ios::binary) ;
    }
    else {
        fout.open(fname / ("2Dgrid_sub"+std::to_string(index+1)+".dat"), std::ios::binary) ;
    }
    
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

// Sub-grid functions

Grid GridManager::add_subgrid(double R_in, double R_out) {

    int sg_idx = in_idx.size();

    if (R_in > R_out) {
        throw std::runtime_error("Subgrid R_in greater than R_out; check values!");
    }

    for (int i=0; i<g.NR+2*g.Nghost+1; i++) {
        if (g.Re(i) > R_in) {
            
            if (i <= g.Nghost) {
                in_idx.push_back(0);
                break;
            } // Check whether we are nearer Re(i) or Re(i-1) and use the closest
            else if ((g.Re(i)-R_in) < (R_in-g.Re(i-1))) {
                in_idx.push_back(i - g.Nghost);
                break;
            }
            else {
                in_idx.push_back(i-1 - g.Nghost);
                break;
            }
        }
        if (i==g.NR+g.Nghost) {throw std::runtime_error("Inner radius of subgrid at outer edge of main grid; check R_in for subgrid!");}
    }

    if (R_out < g.Re(g.Nghost))
        throw std::runtime_error("Outer edge of sub-grid is less than inner edge of main grid!") ;

    for (int i=in_idx[sg_idx]; i<g.NR+2*g.Nghost+1; i++) {
        if (i >= g.NR+g.Nghost) {
            out_idx.push_back(g.NR+2*g.Nghost);
            break;
        }
        if (g.Re(i) > R_out) {
             // Check whether we are nearer Re(i) or Re(i-1) and use the closest
            if ((g.Re(i)-R_out) < (R_out-g.Re(i-1))) {
                out_idx.push_back(i + g.Nghost);
                break;
            }
            else {
                out_idx.push_back(i-1 + g.Nghost);
                break;
            }
        }
    }

    CudaArray<double> Re_sub = make_CudaArray<double>(out_idx[sg_idx]-in_idx[sg_idx]+1);
    CudaArray<double> phie_sub = make_CudaArray<double>(g.Nphi+2*g.Nghost+1);

    for (int i=0; i<out_idx[sg_idx]-in_idx[sg_idx]+1; i++) {
        Re_sub[i] = g.Re(i+in_idx[sg_idx]);
    }
    for (int i=0; i<g.Nphi+2*g.Nghost+1; i++) {
        phie_sub[i] = std::asin(g.sin_th(i));
    }

    Grid subg = Grid(out_idx[sg_idx]-in_idx[sg_idx] - 2*g.Nghost, g.Nphi, g.Nghost, std::move(Re_sub), std::move(phie_sub), sg_idx);

    subgrids.push_back(subg);

    return subg;
}

Grid GridManager::add_1Dsubgrid(double R_in, double R_out) {

    int sg_idx = in_idx.size();

    if (R_in > R_out) {
        throw std::runtime_error("Subgrid R_in greater than R_out; check values!");
    }

    for (int i=0; i<g.NR+2*g.Nghost+1; i++) {
        if (g.Re(i) > R_in) {
            
            if (i <= g.Nghost) {
                in_idx.push_back(0);
                break;
            } // Check whether we are nearer Re(i) or Re(i-1) and use the closest
            else if ((g.Re(i)-R_in) < (R_in-g.Re(i-1))) {
                in_idx.push_back(i - g.Nghost);
                break;
            }
            else {
                in_idx.push_back(i-1 - g.Nghost);
                break;
            }
        }
        if (i==g.NR+g.Nghost) {throw std::runtime_error("Inner radius of subgrid at outer edge of main grid; check R_in for subgrid!");}
    }

    if (R_out < g.Re(g.Nghost))
        throw std::runtime_error("Outer edge of sub-grid is less than inner edge of main grid!") ;

    for (int i=in_idx[sg_idx]; i<g.NR+2*g.Nghost+1; i++) {
        if (i >= g.NR+g.Nghost) {
            out_idx.push_back(g.NR+2*g.Nghost);
            break;
        }
        if (g.Re(i) > R_out) {
             // Check whether we are nearer Re(i) or Re(i-1) and use the closest
            if ((g.Re(i)-R_out) < (R_out-g.Re(i-1))) {
                out_idx.push_back(i + g.Nghost);
                break;
            }
            else {
                out_idx.push_back(i-1 + g.Nghost);
                break;
            }
        }
    }

    CudaArray<double> Re_sub = make_CudaArray<double>(out_idx[sg_idx]-in_idx[sg_idx]+1);
    CudaArray<double> phie_sub = make_CudaArray<double>(1+2*g.Nghost+1);

    for (int i=0; i<out_idx[sg_idx]-in_idx[sg_idx]+1; i++) {
        Re_sub[i] = g.Re(i+in_idx[sg_idx]);
    }
    for (int i=0; i<1+2*g.Nghost+1; i++) {
        phie_sub[i] = -0.1 + 4.e-2*i;
    }

    Grid subg = Grid(out_idx[sg_idx]-in_idx[sg_idx] - 2*g.Nghost, 1, g.Nghost, std::move(Re_sub), std::move(phie_sub), sg_idx);

    subgrids.push_back(subg);

    return subg;
}


template<typename T>
__global__
void _copy_to_subgrid(GridRef g_sub, int idx_in, int idx_out, Field3DConstRef<T> F_main, Field3DRef<T> F_sub) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    // printf("%d\n",g_sub.NR);

    for (int i=iidx; i<g_sub.NR+2*g_sub.Nghost; i+=istride) {
        for (int j=jidx; j<g_sub.Nphi+2*g_sub.Nghost; j+=jstride) { 
            for (int k=0; k<F_main.Nd; k++) { 
                F_sub(i,j,k) = F_main(idx_in+i,j,k);
            }
        }
    }
}

template<typename T>
void GridManager::copy_to_subgrid(Grid& g_sub, const Field<T>& F_main, Field<T>& F_sub) {

    int sg_idx = g_sub.index;

    if (g_sub.Re(0) != subgrids[sg_idx].Re(0) || g_sub.Re(g_sub.NR+2*g_sub.Nghost) != subgrids[sg_idx].Re(subgrids[sg_idx].NR+2*subgrids[sg_idx].Nghost)) {
        throw std::runtime_error("Incorrect subgrid passed to copier.");
    }

    dim3 threads(32,32,1);
    dim3 blocks((g_sub.NR + 2*g_sub.Nghost+31)/32,(g_sub.Nphi + 2*g_sub.Nghost+31)/32, 1) ;

    _copy_to_subgrid<<<blocks,threads>>>(g_sub, in_idx[sg_idx], out_idx[sg_idx], Field3DConstRef<T>(F_main), Field3DRef<T>(F_sub));
    check_CUDA_errors("_copy_from_subgrid");
}

template<typename T>
void GridManager::copy_to_subgrid(Grid& g_sub, const Field3D<T>& F_main, Field3D<T>& F_sub) {

    int sg_idx = g_sub.index;

    if (g_sub.Re(0) != subgrids[sg_idx].Re(0) || g_sub.Re(g_sub.NR+2*g_sub.Nghost) != subgrids[sg_idx].Re(subgrids[sg_idx].NR+2*subgrids[sg_idx].Nghost)) {
        throw std::runtime_error("Incorrect subgrid passed to copier.");
    }

    dim3 threads(32,32,1);
    dim3 blocks((g_sub.NR + 2*g_sub.Nghost+31)/32,(g_sub.Nphi + 2*g_sub.Nghost+31)/32, 1) ;

    _copy_to_subgrid<<<blocks,threads>>>(g_sub, in_idx[sg_idx], out_idx[sg_idx], Field3DConstRef<T>(F_main), Field3DRef<T>(F_sub));
    check_CUDA_errors("_copy_from_subgrid");
}

template<typename T>
__global__
void _copy_from_subgrid(GridRef g_main, GridRef g_sub, int idx_in, int idx_out, Field3DRef<T> F_main, Field3DConstRef<T> F_sub) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx; i<g_sub.NR+2*g_sub.Nghost; i+=istride) {

        if (i<g_sub.Nghost && idx_in != 0) {
            continue;
        }
        else if (i>g_sub.NR+g_sub.Nghost-1 && idx_out != g_main.NR+2*g_main.Nghost) {
            continue;
        }
        else {  
            for (int j=jidx; j<g_sub.Nphi+2*g_sub.Nghost; j+=jstride) { 
                for (int k=0; k<F_main.Nd; k++) { 
                    F_main(idx_in+i,j,k) = F_sub(i,j,k);
                }
            }
        }
    }
}

template<typename T>
void GridManager::copy_from_subgrid(Grid& g_sub, Field<T>& F_main, const Field<T>& F_sub) {

    int sg_idx = g_sub.index;

    if (g_sub.Re(0) != subgrids[sg_idx].Re(0) || g_sub.Re(g_sub.NR+2*g_sub.Nghost) != subgrids[sg_idx].Re(subgrids[sg_idx].NR+2*subgrids[sg_idx].Nghost)) {
        throw std::runtime_error("Incorrect subgrid passed to copier.");
    }

    dim3 threads(32,32,1);
    dim3 blocks((g_sub.NR + 2*g_sub.Nghost+31)/32,(g_sub.Nphi + 2*g_sub.Nghost+31)/32, 1) ;

    _copy_from_subgrid<<<blocks,threads>>>(g, g_sub, in_idx[sg_idx], out_idx[sg_idx], Field3DRef<T>(F_main), Field3DConstRef<T>(F_sub));
    check_CUDA_errors("_copy_from_subgrid");
}

template<typename T>
void GridManager::copy_from_subgrid(Grid& g_sub, Field3D<T>& F_main, const Field3D<T>& F_sub) {

    int sg_idx = g_sub.index;

    if (g_sub.Re(0) != subgrids[sg_idx].Re(0) || g_sub.Re(g_sub.NR+2*g_sub.Nghost) != subgrids[sg_idx].Re(subgrids[sg_idx].NR+2*subgrids[sg_idx].Nghost)) {
        throw std::runtime_error("Incorrect subgrid passed to copier.");
    }

    dim3 threads(32,32,1);
    dim3 blocks((g_sub.NR + 2*g_sub.Nghost+31)/32,(g_sub.Nphi + 2*g_sub.Nghost+31)/32, 1) ;

    _copy_from_subgrid<<<blocks,threads>>>(g, g_sub, in_idx[sg_idx], out_idx[sg_idx], Field3DRef<T>(F_main), Field3DConstRef<T>(F_sub));
    check_CUDA_errors("_copy_from_subgrid");
}

template<typename T>
void GridManager::copy_to_subgrid(Grid& g_sub, const CudaArray<T>& F_main, CudaArray<T>& F_sub) { 
    
    int sg_idx = g_sub.index;

    if (g_sub.Re(0) != subgrids[sg_idx].Re(0) || g_sub.Re(g_sub.NR+2*g_sub.Nghost) != subgrids[sg_idx].Re(subgrids[sg_idx].NR+2*subgrids[sg_idx].Nghost)) {
        throw std::runtime_error("Incorrect subgrid passed to copier.");
    }

    for (int i=0; i<g_sub.NR+2*g_sub.Nghost; i++) {
        F_sub[i] = F_main[in_idx[sg_idx]+i]; 
    }
}

template<typename T>
void GridManager::copy_from_subgrid(Grid& g_sub, CudaArray<T>& F_main, const CudaArray<T>& F_sub) { 
    
    int sg_idx = g_sub.index;

    if (g_sub.Re(0) != subgrids[sg_idx].Re(0) || g_sub.Re(g_sub.NR+2*g_sub.Nghost) != subgrids[sg_idx].Re(subgrids[sg_idx].NR+2*subgrids[sg_idx].Nghost)) {
        throw std::runtime_error("Incorrect subgrid passed to copier.");
    }

    for (int i=0; i<g_sub.NR+2*g_sub.Nghost; i++) {

        if (i<g_sub.Nghost && in_idx[sg_idx] != 0) {
            continue;
        }
        else if (i>g_sub.NR+g_sub.Nghost-1 && out_idx[sg_idx] != g.NR+2*g.Nghost) {
            continue;
        }
        else {  
            F_main[in_idx[sg_idx]+i] = F_sub[i];
        }
    }
}


template void GridManager::copy_to_subgrid<double>(Grid& g_sub, const Field<double>& F_main, Field<double>& F_sub);
template void GridManager::copy_from_subgrid<double>(Grid& g_sub, Field<double>& F_main, const Field<double>& F_sub);

template void GridManager::copy_to_subgrid<int>(Grid& g_sub, const Field<int>& F_main, Field<int>& F_sub);
template void GridManager::copy_from_subgrid<int>(Grid& g_sub, Field<int>& F_main, const Field<int>& F_sub);

template void GridManager::copy_to_subgrid<Prims>(Grid& g_sub, const Field<Prims>& F_main, Field<Prims>& F_sub);
template void GridManager::copy_from_subgrid<Prims>(Grid& g_sub, Field<Prims>& F_main, const Field<Prims>& F_sub);

template void GridManager::copy_to_subgrid<double>(Grid& g_sub, const Field3D<double>& F_main, Field3D<double>& F_sub);
template void GridManager::copy_from_subgrid<double>(Grid& g_sub, Field3D<double>& F_main, const Field3D<double>& F_sub);

template void GridManager::copy_to_subgrid<int>(Grid& g_sub, const Field3D<int>& F_main, Field3D<int>& F_sub);
template void GridManager::copy_from_subgrid<int>(Grid& g_sub, Field3D<int>& F_main, const Field3D<int>& F_sub);

template void GridManager::copy_to_subgrid<Prims>(Grid& g_sub, const Field3D<Prims>& F_main, Field3D<Prims>& F_sub);
template void GridManager::copy_from_subgrid<Prims>(Grid& g_sub, Field3D<Prims>& F_main, const Field3D<Prims>& F_sub);

template void GridManager::copy_to_subgrid<double>(Grid& g_sub, const CudaArray<double>& F_main, CudaArray<double>& F_sub);
template void GridManager::copy_from_subgrid<double>(Grid& g_sub, CudaArray<double>& F_main, const CudaArray<double>& F_sub);

template void GridManager::copy_to_subgrid<int>(Grid& g_sub, const CudaArray<int>& F_main, CudaArray<int>& F_sub);
template void GridManager::copy_from_subgrid<int>(Grid& g_sub, CudaArray<int>& F_main, const CudaArray<int>& F_sub);
