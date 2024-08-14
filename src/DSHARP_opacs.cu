#include "dustdynamics.h"
#include "cuda_runtime.h"
#include "DSHARP_opacs.h"
#include <filesystem>

__global__ void _calc_rho_kappa(GridRef g, Field3DConstRef<Prims> qd, FieldConstRef<Prims> wg, DSHARP_opacsRef opacs,
                         double kgas_abs, double kgas_sca, Field3DRef<double> rhokabs, Field3DRef<double> rhoksca) {

    int k = threadIdx.x + blockIdx.x*blockDim.x ;
    int j = threadIdx.y + blockIdx.y*blockDim.y ;
    int i = threadIdx.z + blockIdx.z*blockDim.z ;

    if (k < opacs.n_lam && j < g.Nphi + 2*g.Nghost && i < g.NR+2*g.Nghost) {

        double rhok_dust_abs = 0;
        double rhok_dust_sca = 0;

        for (int l=0; l<opacs.n_a; l++) { 
            rhok_dust_abs += qd(i,j,l).rho*opacs.k_abs(l,k);
            rhok_dust_sca += qd(i,j,l).rho*opacs.k_sca(l,k);
        }

        rhokabs(i,j,k) = wg(i,j).rho*kgas_abs + rhok_dust_abs;
        rhoksca(i,j,k) = wg(i,j).rho*kgas_sca + rhok_dust_sca;
    }
} 

__global__ void _calc_rho_kappa(GridRef g, Field3DConstRef<Prims> qd, FieldConstRef<Prims> wg, 
                                DSHARP_opacsRef opacs, Field3DRef<double> rhokabs, Field3DRef<double> rhoksca) {

    int k = threadIdx.x + blockIdx.x*blockDim.x ;
    int j = threadIdx.y + blockIdx.y*blockDim.y ;
    int i = threadIdx.z + blockIdx.z*blockDim.z ;

    if (k < opacs.n_lam && j < g.Nphi + 2*g.Nghost && i < g.NR+2*g.Nghost) {

        double rhok_dust_abs = 0;
        double rhok_dust_sca = 0;

        for (int l=0; l<opacs.n_a; l++) { 
            rhok_dust_abs += qd(i,j,l).rho*opacs.k_abs(l,k);
            rhok_dust_sca += qd(i,j,l).rho*opacs.k_sca(l,k);
        }

        rhokabs(i,j,k) = wg(i,j).rho*opacs.k_abs_g(k) + rhok_dust_abs;
        rhoksca(i,j,k) = wg(i,j).rho*opacs.k_sca_g(k) + rhok_dust_sca;
    }
} 

__global__ void _calc_rho_kappa_ice(GridRef g, Field3DConstRef<Prims> qd, FieldConstRef<Prims> wg, 
                                DSHARP_opacsRef opacs, Field3DRef<double> rhokabs, Field3DRef<double> rhoksca, MoleculeRef mol) {

    int k = threadIdx.x + blockIdx.x*blockDim.x ;
    int j = threadIdx.y + blockIdx.y*blockDim.y ;
    int i = threadIdx.z + blockIdx.z*blockDim.z ;

    if (k < opacs.n_lam && j < g.Nphi + 2*g.Nghost && i < g.NR+2*g.Nghost) {

        double rhok_dust_abs = 0;
        double rhok_dust_sca = 0;

        for (int l=0; l<opacs.n_a; l++) { 
            rhok_dust_abs += (qd(i,j,l).rho + mol.ice(i,j,l))*opacs.k_abs(l,k);
            rhok_dust_sca += (qd(i,j,l).rho + mol.ice(i,j,l))*opacs.k_sca(l,k);
        }

        rhokabs(i,j,k) = wg(i,j).rho*opacs.k_abs_g(k) + rhok_dust_abs;
        rhoksca(i,j,k) = wg(i,j).rho*opacs.k_sca_g(k) + rhok_dust_sca;
    }
} 

__global__ void _calc_grain_rho_kappa(GridRef g, Field3DConstRef<Prims> qd, DSHARP_opacsRef opacs,
                                        Field3DRef<double> rhokabs, Field3DRef<double> rhoksca) {

    int k = threadIdx.x + blockIdx.x*blockDim.x ;
    int j = threadIdx.y + blockIdx.y*blockDim.y ;
    int i = threadIdx.z + blockIdx.z*blockDim.z ;

    if (k < opacs.n_a && j < g.Nphi + 2*g.Nghost && i < g.NR+2*g.Nghost) {

        double rhok_dust_abs = 0;
        double rhok_dust_sca = 0;

        for (int l=0; l<opacs.n_lam; l++) { 
            rhok_dust_abs += qd(i,j,k).rho*opacs.k_abs(k,l);
            rhok_dust_sca += qd(i,j,k).rho*opacs.k_sca(k,l);
        }

        rhokabs(i,j,k) = rhok_dust_abs;
        rhoksca(i,j,k) = rhok_dust_sca;
    }
} 

void calculate_total_rhokappa(Grid& g, Field3D<Prims>& qd, Field<Prims>& wg, DSHARP_opacs& opacs,
                                    Field3D<double>& rhokappa_abs, Field3D<double>& rhokappa_sca, 
                                    double kgas_abs, double kgas_sca) {

    int nk = 1 ;
    while (nk < opacs.n_lam && nk < 32)
        nk *= 2 ;
    int nj = 1024 / nk ;

    dim3 threads(nk, nj, 1) ;
    dim3 blocks((opacs.n_lam +  nk-1)/nk, 
                (g.Nphi +  2*g.Nghost + nj-1)/nj, 
                 g.NR +  2*g.Nghost) ;

    _calc_rho_kappa<<<blocks,threads>>>(g, qd, wg, opacs, kgas_abs, kgas_sca, rhokappa_abs, rhokappa_sca);
    check_CUDA_errors("_calc_rho_kappa") ;
}

void calculate_total_rhokappa(Grid& g, Field3D<Prims>& qd, Field<Prims>& wg, DSHARP_opacs& opacs,
                                    Field3D<double>& rhokappa_abs, Field3D<double>& rhokappa_sca) {

    int nk = 1 ;
    while (nk < opacs.n_lam && nk < 32)
        nk *= 2 ;
    int nj = 1024 / nk ;

    dim3 threads(nk, nj, 1) ;
    dim3 blocks((opacs.n_lam +  nk-1)/nk, 
                (g.Nphi +  2*g.Nghost + nj-1)/nj, 
                 g.NR +  2*g.Nghost) ;
    
    _calc_rho_kappa<<<blocks,threads>>>(g, qd, wg, opacs, rhokappa_abs, rhokappa_sca);
    check_CUDA_errors("_calc_rho_kappa") ;
}

void calculate_total_rhokappa(Grid& g, Field3D<Prims>& qd, Field<Prims>& wg, DSHARP_opacs& opacs,
                                    Field3D<double>& rhokappa_abs, Field3D<double>& rhokappa_sca, Molecule& mol) {

    int nk = 1 ;
    while (nk < opacs.n_lam && nk < 32)
        nk *= 2 ;
    int nj = 1024 / nk ;

    dim3 threads(nk, nj, 1) ;
    dim3 blocks((opacs.n_lam +  nk-1)/nk, 
                (g.Nphi +  2*g.Nghost + nj-1)/nj, 
                 g.NR +  2*g.Nghost) ;
    
    _calc_rho_kappa_ice<<<blocks,threads>>>(g, qd, wg, opacs, rhokappa_abs, rhokappa_sca, mol);
    check_CUDA_errors("_calc_rho_kappa_ice") ;
}

void calculate_grain_rhokappa(Grid& g, Field3D<Prims>& qd, DSHARP_opacs& opacs,
                                    Field3D<double>& rhokappa_abs_grain, Field3D<double>& rhokappa_sca_grain) {

    int nk = 1 ;
    while (nk < opacs.n_lam && nk < 32)
        nk *= 2 ;
    int nj = 1024 / nk ;

    dim3 threads(nk, nj, 1) ;
    dim3 blocks((opacs.n_lam +  nk-1)/nk, 
                (g.Nphi +  2*g.Nghost + nj-1)/nj, 
                 g.NR +  2*g.Nghost) ;

    _calc_grain_rho_kappa<<<blocks,threads>>>(g, qd, opacs, rhokappa_abs_grain, rhokappa_sca_grain);
    check_CUDA_errors("_calc_rho_kappa") ;
}

void DSHARP_opacs::interpolate_opacs(const DSHARP_opacs& opac_in) {

    // Step 1:
    //  Interpolate existing wavelength grid to desired grain sizes
    
    std::vector<double> a_grid(opac_in.n_a), lam_grid(opac_in.n_lam); 
    std::vector<double> k_abs_grid(std::max(opac_in.n_a, opac_in.n_lam)); 
    std::vector<double> k_sca_grid(std::max(opac_in.n_a, opac_in.n_lam)); 

    // Set up temporary grid for each of the desired grain sizes
    std::vector<std::vector<double>> logk_abs(n_a);
    std::vector<std::vector<double>> logk_sca(n_a);

    for (int i=0; i < n_a; i++) {
        logk_abs[i] = std::vector<double>(opac_in.n_lam);
        logk_sca[i] = std::vector<double>(opac_in.n_lam);
    }


    // Interpolate to the grain sizes
    for (int j=0; j<opac_in.n_lam; j++) {
        for (int i=0; i<opac_in.n_a; i++) {
            a_grid[i] = std::log10(opac_in.a(i));
            k_abs_grid[i] = std::log10(opac_in.k_abs(i,j));
            k_sca_grid[i] = std::log10(opac_in.k_sca(i,j));
        }
                        
        lam_grid[j] = std::log10(opac_in.lam(j)) ;
        PchipInterpolator<2> interp(a_grid,k_abs_grid,k_sca_grid);
        for (int k=0; k < n_a; k++) {
            auto k_interp = interp(std::log10(a(k))) ;
            logk_abs[k][j] = k_interp[0] ;
            logk_sca[k][j] = k_interp[1] ;
        }
    }
        
    // Step 2:
    //  Interpolate to the desired wavelengths
    for (int i=0; i < n_a; i++) {
        PchipInterpolator<2> interp(lam_grid,logk_abs[i],logk_sca[i]);
        for (int j=0; j<n_lam; j++) {
            auto k_interp = interp(std::log10(lam(j))) ;
            
            k_abs_ptr[i*n_lam + j] = std::pow(10., k_interp[0]);
            k_sca_ptr[i*n_lam + j] = std::pow(10., k_interp[1]);
        }
    }
}


void DSHARP_opacs::write_interp(std::filesystem::path folder) const {

    std::ofstream f(folder / ("interp_opacs.dat"), std::ios::binary);

    f.write((char*) &n_a, sizeof(int));
    f.write((char*) &n_lam, sizeof(int));
    for (int i=0; i < n_a; i++) { 
        for (int j = 0; j < n_lam; j++) {
            double kappaabs = k_abs(i,j), kappasca = k_sca(i,j);
            f.write((char*) &kappaabs, sizeof(double));
            f.write((char*) &kappasca, sizeof(double));
        }
    }
    f.close();

}
