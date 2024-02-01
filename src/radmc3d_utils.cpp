
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "constants.h"
#include "radmc3d_utils.h"



template<typename T0>
void _parse_multiple(std::stringstream& ss, T0& v0) {
    ss >> v0 ;
}

template<typename T0, typename... Ts>
void _parse_multiple(std::stringstream& ss, T0& v0, Ts&... vs) {
    ss >> v0 ;
    _parse_multiple(ss, vs...) ;
}

template<typename... T>
void parse_items(std::string& item, T&... vals) {
    std::stringstream iss(item) ;
    _parse_multiple(iss, vals...) ;
}

template<typename T>
T parse_item(std::string& item) {
    T val ;
    parse_items(item, val) ;
    return val ;
}


namespace Radmc3d {

DiscModel::DiscModel(std::string folder)
  : _folder(folder) 
{
    _parse_amr_grid() ;
}

void DiscModel::_parse_amr_grid(std::string filename) {

    std::ifstream amr_file(_folder + "/" + filename) ;
    std::string line ;

    // Check the format
    getline(amr_file, line) ;
    int format = parse_item<int>(line) ;
    if (format != 1)
        throw std::invalid_argument("AMR file must have format=1") ;

    // Check the grid is not an AMR grid (not supported)
    getline(amr_file, line) ;
    int style = parse_item<int>(line) ;
    if (style != 0)
        throw std::invalid_argument("Only non-refined grids are supported (style=0)") ;

    // Read the co-ordinate system
    getline(amr_file, line) ;
    int coord = parse_item<int>(line) ;
    if (coord < 100 or coord >= 200)
        throw std::invalid_argument("Only spherical grids are supported") ;

    // Read the gridinfo parameter
    getline(amr_file, line) ;
    int gridinfo = parse_item<int>(line) ;
    if (gridinfo != 0)
        throw std::invalid_argument("Only minimal grid information is supported (gridinfo=0)") ;

    // Work out which dimensions are present
    getline(amr_file, line) ;
    int inc_x, inc_y, inc_z ;
    parse_items(line, inc_x, inc_y, inc_z) ;

    getline(amr_file, line) ;
    int nx, ny, nz ;
    parse_items(line, nx, ny, nz) ;
    if (inc_z == 1 and nz > 0)
        throw std::invalid_argument("Only 2D grids are supported") ;
    
    // Setup grid-space
    _radius.resize(nx+1) ;
    _colatitude.resize(ny+1) ;

    for (int i=0; i <= nx; i++) {
        getline(amr_file, line) ;
        _radius[i] = parse_item<double>(line) ;
    }
    for (int i=0; i <= ny; i++) {
        getline(amr_file, line) ;
        _colatitude[i] = parse_item<double>(line) ;
    }

}

Grid DiscModel::create_grid(int nghost_y) const {

    // Setup radial grid
    CudaArray<double> R = make_CudaArray<double>(_radius.size()) ;
    for(int i=0; i <= nr(); i++)
        R[i] = _radius[i] ;

    // Setup latitude grid
    CudaArray<double> lat = make_CudaArray<double>(_colatitude.size()+2*nghost_y) ;
    for(int i=0; i <= nt(); i++)
        lat[i+nghost_y] = M_PI/2 - _colatitude[nt()-i] ;

    // Fill ghosts
    for (int i=0; i < nghost_y; i++) {
        lat[i] = 2*lat[nghost_y] - lat[2*nghost_y-i] ;
        lat[nt() + nghost_y + i+1] = 2*lat[nt() + nghost_y] - lat[nt() + nghost_y-1-i] ;
    }

    return Grid(nr()-2*nghost_y, nt(), nghost_y, std::move(R), std::move(lat)) ;
}

Field<double> DiscModel::read_quantity(std::string filename, const Grid& g) const {

    std::ifstream amr_file(_folder + "/" + filename) ;
    std::string line ;

    // Read the header information
     // Check the format
    getline(amr_file, line) ;
    int format = parse_item<int>(line) ;
    if (format != 1)
        throw std::invalid_argument("read_quantity: File must have format=1") ;

    // Check the grid is not an AMR grid (not supported)
    getline(amr_file, line) ;
    int num_cells = parse_item<int>(line) ;
    if (num_cells != nr()*nt())
        throw std::invalid_argument("read_quantity: File has wrong number of cells");

    // Read the co-ordinate system
    getline(amr_file, line) ;
    int num_spec = parse_item<int>(line) ;
    if (num_spec != 1)
        throw std::invalid_argument("Only 1 species is supported") ;

    // Load the data
    std::vector<double> data(num_cells) ;

    for(int i=0; i < num_cells; i++) {
        getline(amr_file, line) ;
        data[i] = parse_item<double>(line) ;
    }

    // Create the Field:
    Field<double> field = create_field<double>(g)  ;

    // Interpolate the data to the new grid
    for (int j=0; j < nt(); j++) {
        for (int i=0; i < nr(); i++) {
            double r = g.rc(i, j+g.Nghost) ;
            field[field.index(i, j+g.Nghost)] = interp_qty(data, r, nt() - (j+1)) ;
        }
    }

    // Fill the ghosts
    for (int i=0; i < nr(); i++)
        for (int j=0; j < g.Nghost; j++) {
            field[field.index(i,j)] = field[field.index(i,2*g.Nghost - (j+1))] ;
            field[field.index(i,g.Nphi+g.Nghost+j)] = field[field.index(i,g.Nphi+g.Nghost-1)] ;
        }
    
    return field ;
        
}

double DiscModel::interp_qty(const std::vector<double>& qty, double r, int j) const {

    // Get the appropriate cell index
    int i = 0;
    while (r > r_c(i) && i < nr()) 
        i++ ;

    if (i == 0)
        return qty[j*nr() + i] ;
    else if (i == nr())
        return qty[j*nr() + nr()-1] ;
    else {
        double f = (r - r_c(i-1))/(r_c(i) - r_c(i-1)) ;
        return  qty[j*nr() + i]*f +  qty[j*nr() + i-1]*(1-f) ;
    }
}

DustOpacity::DustOpacity(std::string folder, std::string filename)
  : _folder(folder), _file(filename) 
{
      _parse_dust_opacity() ;
}

void DustOpacity::_parse_dust_opacity() {

    std::ifstream opac_file(_folder + "/" + _file) ;
    std::string line ;

    // Check the format
    getline(opac_file, line) ;
    while (line[0] == '#')
        getline(opac_file, line) ;
    
    int format = parse_item<int>(line) ;
    if (format < 1 || format > 3)
        throw std::invalid_argument("DustOpacity:Only 1 <= format <= 3 is supported");

    // Get the number of bins
    getline(opac_file, line) ;
    int nwle = parse_item<int>(line) ;

    _wle.resize(nwle) ;
    _kabs.resize(nwle) ;
    if (format > 1) _ksca.resize(nwle) ;
    if (format > 2) _gsca.resize(nwle) ;

    // Dummy line
    getline(opac_file, line) ;

    for (int i=0; i < nwle; i++) {
        
        getline(opac_file, line) ;
        switch (format) {
          case 1:
            parse_items(line, _wle[i], _kabs[i]) ;
            break;
          case 2:
            parse_items(line, _wle[i], _kabs[i], _ksca[i]) ;
            break;
          case 3:
            parse_items(line, _wle[i], _kabs[i], _ksca[i], _gsca[i]) ;
            break;
        }

    }
}

double DustOpacity::_interp(double l, const std::vector<double>& qty, bool extrap) const {
    int il = 0;
    int iu = num_wle()-1 ;

    // Out of range
    if (l < _wle.front()) 
        return qty.front() ;

    if (l > _wle.back()) {
        if (not extrap) return qty.back() ;

        double q = 
            std::log(qty[iu-1]/qty[iu-2]) / 
            std::log(_wle[iu-1]/_wle[iu-2]) ;
        
        return qty.back() * std::pow(l/_wle.back(), q) ;
    }

    // Bracket wavelength
    while (iu > il+1) {
        int im = (il+iu)/2 ;
        double lm = _wle[im] ;

        if (l > lm)
            il = im ;
        else 
            iu = im ;
    }
    assert(il+1 == iu) ;

    double f = (l - _wle[il])/(_wle[iu] - _wle[il]) ;

    return qty[iu]*f + (1-f) * qty[il] ;
}


Star load_stellar_properties(std::string folder, std::string filename) {

    std::ifstream star_file(folder + "/" + filename) ;
    std::string line ;

    // Check the format
    getline(star_file, line) ;
    int format = parse_item<int>(line) ;

    if (format != 2)
        throw std::invalid_argument("load_stellar_properties: format=2 required") ;

    getline(star_file, line) ;    
    int nstar, nwav ;
    parse_items(line, nstar, nwav) ;

    if (nstar != 1)
         throw std::invalid_argument("load_stellar_properties: Only one star is supported") ;

    // Empty line
    //getline(star_file, line) ;    

    // Stellar properties
    getline(star_file, line) ;    
    double mass, radius, x,y,z ;
    parse_items(line, radius, mass, x,y,z) ;

    // Fluxes
    CudaArray<double> wle  = make_CudaArray<double>(nwav) ;
    CudaArray<double> flux = make_CudaArray<double>(nwav) ;

    // Empty line
    getline(star_file, line) ;    

    for (int i=0; i < nwav; i++) {
        getline(star_file, line) ;
        wle[i] = parse_item<double>(line) ;
    }

    // Empty line
    getline(star_file, line) ;    

    for (int i=0; i < nwav; i++) {
        getline(star_file, line) ;
        flux[i] = parse_item<double>(line); 
    }

    // Handle case of Black-body star:
    if (flux[0] < 0) {
        double L = 4*M_PI*radius*radius*sigma_SB*std::pow(flux[0], 4) ;

        Star star(GMsun*mass/Msun, L, -flux[0]) ;
        star.set_wavelengths(nwav, wle) ;
        star.set_blackbody_fluxes() ;

        return star ;
    }

    // Empty line
    getline(star_file, line) ;    

    // Integrate flux over bands
    CudaArray<double> Lband = make_CudaArray<double>(nwav) ;
    for (int i=0; i < nwav; i++) {
        double fl, fr, wl, wr ;
       
        fl = flux[std::max(i-1,0)] ;
        wl = wle[std::max(i-1,0)]/wle[i] ;

        fr = flux[std::min(i+1,nwav-1)] ;
        wr = wle[std::min(i+1,nwav-1)]/wle[i] ;

        double q = std::log(fr/fl) / std::log(wr/wl) ;

        if (i==0)
            wl = 0 ;
        else
            wl = std::sqrt(wle[i-1]/wle[i]) ;
        
        if (i==nwav-1)
            wr = 1e10*wle[nwav-1] ;
        else 
            wr = std::sqrt(wle[i+1]/wle[i]) ;

        Lband[i] = flux[i] * (std::pow(wr,q+1) - std::pow(wl,q+1))/(q+1) 
            * 4 * M_PI * 3.08572e18 * 3.08572e18 ;
    }

    return Star(GMsun*mass/Msun, radius, nwav, wle, Lband) ; 
}


} // namespace Radmc3d