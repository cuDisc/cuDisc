
#ifndef _CUDISC_GRID_H_
#define _CUDISC_GRID_H_

#include "cuda_array.h"
#include "field.h"
#include <filesystem>
#include <vector>

// Variables for controlling the grid spacing in the radial / theta directions
enum class RadialSpacing {
    linear, log, power 
} ;

enum class ThetaSpacing {
    linear, subdiv, power
} ;

// Specify whether a field is staggered
enum class Staggering {
    none, R, phi, both 
} ;

// Specify grid coordinate system

enum class Coords {
    cart, cyl
} ;

struct vec { 
    double R, Z;

    __device__ __host__ vec& operator+=(const vec& o) {
        R += o.R ; Z += o.Z ;
        return *this ;
    }

    __device__ __host__ vec& operator-=(const vec& o) {
        R -= o.R ; Z -= o.Z ;
        return *this ;
    }

    __device__ __host__ vec& operator*=(double s) {
        R *= s ; Z *= s ;
        return *this ;
    }

    __device__ __host__ vec& operator/=(double s) {
        R /= s ; Z /= s;
        return *this ;
    }
} ;
inline __device__ __host__ vec operator+(const vec& a, const vec& b) {
    vec res = a ;
    res += b ;
    return res; 
}
inline __device__ __host__ vec operator-(const vec& a, const vec& b) {
    vec res = a ;
    res -= b ;
    return res; 
}
inline __device__ __host__ vec operator*(const vec& a, double s) {
    vec res = a ;
    res *= s ;
    return res; 
}
inline __device__ __host__ vec operator/(const vec& a, double s) {
    vec res = a ;
    res /= s ;
    return res;
}
inline __device__ __host__ double dot(const vec& a, const vec& b) {
    return a.R*b.R + a.Z*b.Z ;
}


class GridRef ;

/* class Grid
 *
 *   Handles the geometry factors for the mesh grid. Provides distances, face 
 *   areas and normals, and cell volumes for each grid cell.
 *   
 *   Cells are indexed as follows:
 * 
 *   -----*------------*------------*------------*----- Edge j+2
 *        |            |            |            |
 *        | (i-1, j+1) | ( i , j+1) | (i+1, j+1) |
 *        |            |            |            |  
 *   -----*------------*------------*------------*----- Edge j+1
 *        |            |            |            |
 *        | (i-1,  j ) | ( i ,  j ) | (i+1,  j ) |
 *        |            |            |            | 
 *   -----*------------*------------*------------*----- Edge j
 *        |            |            |            |
 *        | (i-1, j-1) | ( i , j-1) | (i+1, j-1) |
 *        |            |            |            |
 *   -----*------------*------------*------------*----- Edge j-1
 *        |            |            |            |
 *     Edge i-1     Edge i       Edge i+1     Edge i+2     
 * 
 *   The definitions of the face areas / volumes are:
 *     - area_R : the face area between cells (i-1, j) and (i, j)
 *     - area_Z : the face area between cells (i, j-1) and (i, j)
 *     - face_normal_R : normal to the face between cells (i-1, j) and (i, j)
 *                       pointing into cell (i,j)
 *     - face_normal_Z : normal to the face between cells (i, j-1) and (i, j)
 *                       pointing into cell (i,j)
 *     - volume : volume of cell (i,j).
 *  
 *   Similarly the cell centres are given by Rc and Zc (cylindical 
 *   co-ordinates), while the centre of the faces are given by Re and Ze.
 * 
 *  Notes:
 *  - The grid geometry is a quadrilateral grid in cylindrical radius (R) and 
 *    polar angle (theta), not R-Z which is commonly used. The grid is therefore
 *    not an orthogonal grid. 
 * 
 * - The definition used here has theta = 0 referring to the disc mid-plane. 
 *   
 */
class Grid {
  public:

    // Initialization parameters for Grid Objects
    struct params {
        int NR, Nphi, Nghost = 2 ;
        double Rmin, Rmax ;
        double theta_min, theta_max ;
        RadialSpacing R_spacing = RadialSpacing::linear ;
        ThetaSpacing theta_spacing = ThetaSpacing::linear ;
        double theta_subdiv = 0.;
        double theta_power = 0.;
        double R_power = 0.;
        int index = -1;
    } ;

    int NR, Nphi, Nghost, index;
    Coords coord_system = Coords::cyl;

    Grid(Grid::params) ;
    Grid(int NR, int Nphi, int Nghost, 
         CudaArray<double> R, CudaArray<double> phi, int index=-1);

    void set_coord_system(Coords system) {
        coord_system = system;
    }

    double area_R(int i, int j) const { 
        switch(coord_system) {
            case Coords::cart:{
                return Re(i)*(_tan_theta_e[j+1] - _tan_theta_e[j]);
            }
            case Coords::cyl: {
                return Re(i) * Re(i) * (_tan_theta_e[j+1] - _tan_theta_e[j]) ;
            }
            default: __builtin_unreachable();
        }
    }

    double area_Z(int i, int j) const { 
        switch(coord_system) {
            case Coords::cart: {
                return (Re(i+1) - Re(i))/_cos_theta_e[j];
            }
            case Coords::cyl: {
                return 0.5 * (Re(i+1) * Re(i+1) - Re(i) * Re(i)) / _cos_theta_e[j] ;
            }
            default: __builtin_unreachable();
        }
    }
    
    double volume(int i, int j) const {
        switch(coord_system) {
            case Coords::cart: {
                return 0.5*(Re(i+1)*Re(i+1) - Re(i)*Re(i)) * (_tan_theta_e[j+1] - _tan_theta_e[j]); 
            }
            case Coords::cyl: {
                double dV = (Re(i+1)*Re(i+1)*Re(i+1) - Re(i)*Re(i)*Re(i)) / 3 ;
                return dV * (_tan_theta_e[j+1] - _tan_theta_e[j]) ;
            }    
            default: __builtin_unreachable();
        }
    }

    // Locations in cylindrical co-ordinates
    double  Rc(int i) const { return _Rc[i] ; }
    double  Re(int i) const { return _Re[i] ; }
    double dRc(int i) const { return Rc(i+1) - Rc(i) ; }
    double dRe(int i) const { return Re(i+1) - Re(i) ; }

    double  Zc(int i, int j) const { return Rc(i) * _tan_theta_c[j] ; }
    double  Ze(int i, int j) const { return Rc(i) * _tan_theta_e[j] ; }
    double dZc(int i, int j) const { return Zc(i, j+1) - Zc(i, j) ; }
    double dZe(int i, int j) const { return Ze(i, j+1) - Ze(i, j) ; }

    vec face_normal_R(int, int) const {
        return {1, 0} ;
    }
    vec face_normal_Z(int, int j) const {
        return { -_sin_theta_e[j], _cos_theta_e[j] } ;
    }

    // Locations in spherical co-ordinates
    double  rc(int i, int j) const { return Rc(i) / _cos_theta_c[j] ; }
    double  re(int i, int j) const { return Re(i) / _cos_theta_c[j] ; }
    double drc(int i, int j) const { return rc(i+1,j) - rc(i,j) ; }
    double dre(int i, int j) const { return re(i+1,j) - re(i,j) ; }

    double sin_th(int j) const { return _sin_theta_e[j] ;}
    double cos_th(int j) const { return _cos_theta_e[j] ;} 
    double sin_th_c(int j) const { return _sin_theta_c[j] ;}
    double cos_th_c(int j) const { return _cos_theta_c[j] ;} 
    double dsin_th(int j) const { return _sin_theta_e[j+1] - _sin_theta_e[j] ;}

    void write_grid(std::filesystem::path) const ;

  private:
    CudaArray<double> _Re, _Rc ;
    CudaArray<double> _sin_theta_e, _sin_theta_c ;
    CudaArray<double> _cos_theta_e, _cos_theta_c ;
    CudaArray<double> _tan_theta_e, _tan_theta_c ;

    friend class GridRef ;
} ;


/* class GridRef
 *
 * Reference type for Grid class.
 *  
 * This class exists to enable copying Grid objects to the GPU. Since objects
 * can't be passed by reference to __global__ functions we need a special class
 * to handle this. Note that passing by value is impossible because CudaArrays
 * are non-copyable.
 * 
 * Note:
 *  - Grid objects are logically const so we only have one reference type.
 * 
 */
class GridRef {
  public:
    int NR, Nphi, Nghost, index ;
    Coords coord_system;

    GridRef(const Grid& g)
     : NR(g.NR), Nphi(g.Nphi), Nghost(g.Nghost), index(g.index), coord_system(g.coord_system),
       _Re(g._Re.get()), _Rc(g._Rc.get()), 
       _sin_theta_e(g._sin_theta_e.get()), _sin_theta_c(g._sin_theta_c.get()),
       _cos_theta_e(g._cos_theta_e.get()), _cos_theta_c(g._cos_theta_c.get()),
       _tan_theta_e(g._tan_theta_e.get()), _tan_theta_c(g._tan_theta_c.get())
    { } ;

    // __host__ __device__ 
    // double area_R(int i, int j) const { 
    //     //return Re(i)*(_tan_theta_e[j+1] - _tan_theta_e[j]); // cart coords
    //     return Re(i) * Re(i) * (_tan_theta_e[j+1] - _tan_theta_e[j]) ;
    // }
    // __host__ __device__ 
    // double area_Z(int i, int j) const { 
    //     //return (Re(i+1) - Re(i))/_cos_theta_e[j]; // cart coords
    //     return 0.5 * (Re(i+1) * Re(i+1) - Re(i) * Re(i)) / _cos_theta_e[j] ;
    // }
    
    // __host__ __device__ 
    // double volume(int i, int j) const {
    //     //return 0.5*(Re(i+1)*Re(i+1) - Re(i)*Re(i)) * (_tan_theta_e[j+1] - _tan_theta_e[j]); // cart coords
    //     double dV = (Re(i+1)*Re(i+1)*Re(i+1) - Re(i)*Re(i)*Re(i)) / 3 ;
    //     return dV * (_tan_theta_e[j+1] - _tan_theta_e[j]) ;
    // }

    __host__ __device__ 
    double area_R(int i, int j) const { 
        switch(coord_system) {
            case Coords::cart: {
                return Re(i)*(_tan_theta_e[j+1] - _tan_theta_e[j]);
            }
            case Coords::cyl: {
                return Re(i) * Re(i) * (_tan_theta_e[j+1] - _tan_theta_e[j]) ;
            } 
            default: __builtin_unreachable();
        }
    }
    __host__ __device__ 
    double area_Z(int i, int j) const { 
        switch(coord_system) {
            case Coords::cart: {
                return (Re(i+1) - Re(i))/_cos_theta_e[j];
            }    
            case Coords::cyl: {
                return 0.5 * (Re(i+1) * Re(i+1) - Re(i) * Re(i)) / _cos_theta_e[j] ;
            }  
            default: __builtin_unreachable();
        }
    }
    __host__ __device__ 
    double volume(int i, int j) const {
        switch(coord_system) {
            case Coords::cart: {
                return 0.5*(Re(i+1)*Re(i+1) - Re(i)*Re(i)) * (_tan_theta_e[j+1] - _tan_theta_e[j]);
            }    
            case Coords::cyl: {
                double dV = (Re(i+1)*Re(i+1)*Re(i+1) - Re(i)*Re(i)*Re(i)) / 3 ;
                return dV * (_tan_theta_e[j+1] - _tan_theta_e[j]) ;
            }
            default: __builtin_unreachable();
        }
    }


    __host__ __device__ 
    double  Rc(int i) const { return _Rc[i] ; }
    __host__ __device__ 
    double  Re(int i) const { return _Re[i] ; }
    __host__ __device__ 
    double dRc(int i) const { return Rc(i+1) - Rc(i) ; }
    __host__ __device__ 
    double dRe(int i) const { return Re(i+1) - Re(i) ; }

    __host__ __device__ 
    double  Zc(int i, int j) const { return Rc(i) * _tan_theta_c[j] ; }
    __host__ __device__ 
    double  Ze(int i, int j) const { return Rc(i) * _tan_theta_e[j] ; }
    __host__ __device__ 
    double dZc(int i, int j) const { return Zc(i, j+1) - Zc(i, j) ; }
    __host__ __device__ 
    double dZe(int i, int j) const { return Ze(i, j+1) - Ze(i, j) ; }

    __host__ __device__ 
    vec face_normal_R(int, int) const {
        return {1, 0} ;
    }
    __host__ __device__ 
    vec face_normal_Z(int, int j) const {
        return { -_sin_theta_e[j], _cos_theta_e[j] } ;
    }

    // Locations in spherical co-ordinates
    __host__ __device__ 
    double  rc(int i, int j) const { return Rc(i) / _cos_theta_c[j] ; }
    __host__ __device__ 
    double  re(int i, int j) const { return Re(i) / _cos_theta_c[j] ; }
    __host__ __device__ 
    double drc(int i, int j) const { return rc(i+1,j) - rc(i,j) ; }
    __host__ __device__ 
    double dre(int i, int j) const { return re(i+1,j) - re(i,j) ; }

    __host__ __device__
    double sin_th(int j) const { return _sin_theta_e[j] ;}
    __host__ __device__
    double cos_th(int j) const { return _cos_theta_e[j] ;} 
    __host__ __device__
    double sin_th_c(int j) const { return _sin_theta_c[j] ;}
    __host__ __device__
    double cos_th_c(int j) const { return _cos_theta_c[j] ;} 
    __host__ __device__
    double dsin_th(int j) const { return _sin_theta_e[j+1] - _sin_theta_e[j] ;}

  private:
    const double *_Re, *_Rc ;
    const double *_sin_theta_e, *_sin_theta_c ;
    const double *_cos_theta_e, *_cos_theta_c ;
    const double *_tan_theta_e, *_tan_theta_c ;

} ;

/* create_field
 *
 *  Create a Field object (data storage) for a given grid
 */
template<class T>
Field<T> create_field(const Grid& g, Staggering s=Staggering::none) {
    
    int NR = g.NR + 2*g.Nghost ;
    int Nphi = g.Nphi + 2*g.Nghost ;
    if (s == Staggering::R or s == Staggering::both) 
        NR += 1 ;
    if (s == Staggering::phi or s == Staggering::both) 
        Nphi += 1 ;
     
    return Field<T>(NR, Nphi) ;
}


/* create_field3D
 *
 *  Create a Field3D object (data storage) for a given grid
 */
template<class T>
Field3D<T> create_field3D(const Grid& g, int n, Staggering s=Staggering::none) {
    
    int NR = g.NR + 2*g.Nghost ;
    int Nphi = g.Nphi + 2*g.Nghost ;
    if (s == Staggering::R or s == Staggering::both) 
        NR += 1 ;
    if (s == Staggering::phi or s == Staggering::both) 
        Nphi += 1 ;
    
    return Field3D<T>(NR, Nphi, n) ;
}

class OrthGrid {

    public:

        int NR, NZ, Nghost;

        OrthGrid(int _NR, int _NZ, int _Nghost, double Rmin, double Rmax, double Zmin, double Zmax) ;

        double area_R(int i, int j) const { 
            return (Ze(i,j+1) - Ze(i,j));//*Re(i);
        }
        double area_Z(int i, int) const { 
            return Re(i+1) - Re(i);//0.5 * (Re(i+1) * Re(i+1) - Re(i) * Re(i)) ;//
            
        }
        double volume(int i, int j) const {
            return (Ze(i,j+1) - Ze(i,j)) *  (Re(i+1) - Re(i));
            //return 0.5 * (Re(i+1) * Re(i+1) - Re(i) * Re(i)) * (Ze(i,j+1) - Ze(i,j));
        
        }     


        double  Rc(int i) const { return _Rc[i] ; }
        double  Re(int i) const { return _Re[i] ; }
        double dRc(int i) const { return Rc(i+1) - Rc(i) ; }
        double dRe(int i) const { return Re(i+1) - Re(i) ; }

        double  Zc(int, int j) const { return _Zc[j] ; }
        double  Ze(int, int j) const { return _Ze[j] ; }
        double dZc(int i, int j) const { return Zc(i,j+1) - Zc(i,j) ; }
        double dZe(int i, int j) const { return Ze(i,j+1) - Ze(i,j) ; }
        
        vec face_normal_R(int, int) const {
            return {1, 0} ;
        }
        vec face_normal_Z(int, int) const {
            return {0, 1} ;
        }
    private:

        CudaArray<double> _Re, _Rc ;
        CudaArray<double> _Ze, _Zc ;

} ;


class GridManager {
    public:
        GridManager(Grid& g_main) : g(g_main) {}

        Grid add_subgrid(double R_in, double R_out);

        template<typename T>
        void copy_to_subgrid(Grid& g_sub, const Field<T>& F_main, Field<T>& F_sub) ;

        template<typename T>
        void copy_from_subgrid(Grid& g_sub, Field<T>& F_main, const Field<T>& F_sub) ;
        
        template<typename T>
        void copy_to_subgrid(Grid& g_sub, const Field3D<T>& F_main, Field3D<T>& F_sub) ;

        template<typename T>
        void copy_from_subgrid(Grid& g_sub, Field3D<T>& F_main, const Field3D<T>& F_sub) ;

    private:
        std::vector<int> in_idx;
        std::vector<int> out_idx;
        GridRef g;
        std::vector<GridRef> subgrids;
} ;



#endif//_CUDISC_GRID_H_