

#include <iostream>
#include <cuda_runtime.h>
#include <vector>

#include "constants.h"
#include "planck.h"
#include "opacity.h"
#include "reductions.h"
#include "stellar_irradiation.h"
#include "utils.h"
#include "timing.h"
#include "dustdynamics.h"
#include "DSHARP_opacs.h"
#include "interpolate.h"

__global__ void _cell_optical_depth_tab(GridRef g, 
                                    int num_wavelengths, 
                                    Field3DConstRef<double> rhokappa,
                                    Field3DRef<double> tau) {

    int k = threadIdx.x + blockIdx.x*blockDim.x ;
    int j = threadIdx.y + blockIdx.y*blockDim.y ;
    int i = threadIdx.z + blockIdx.z*blockDim.z ;
    
    if (k < num_wavelengths && j < g.Nphi + 2*g.Nghost && i < g.NR+2*g.Nghost) {
        tau(i,j,k) = g.dre(i, j) * rhokappa(i,j,k) ;
    }
} 


__global__ void _cell_optical_depth_tab_with_scattering(GridRef g, 
                                                        int num_wavelengths, 
                                                        Field3DConstRef<double> rhok_abs,
                                                        Field3DConstRef<double> rhok_sca,
                                                        Field3DRef<double> tau) {

    int k = threadIdx.x + blockIdx.x*blockDim.x ;
    int j = threadIdx.y + blockIdx.y*blockDim.y ;
    int i = threadIdx.z + blockIdx.z*blockDim.z ;

    if (k < num_wavelengths && j < g.Nphi + 2*g.Nghost && i < g.NR+2*g.Nghost) {
        tau(i,j,k) = g.dre(i,j) * (rhok_abs(i,j,k) + rhok_sca(i,j,k)) ;
    }
} 

__global__ void _volumetric_heating(GridRef g, 
                                    int num_wavelengths, const double* Lband,
                                    Field3DConstRef<double> tau,
                                    FieldRef<double> heating) {

    int j = threadIdx.x + blockIdx.x*blockDim.x ;
    int i = threadIdx.y + blockIdx.y*blockDim.y ;

    if (i < g.NR + 2*g.Nghost && j < g.Nphi + 2*g.Nghost) {

        double heat = 0 ;
        double tau0 = 0, dtau ;
        for (int k=0; k < num_wavelengths; k++) {
            if (i > 0) tau0 = tau[tau.index(i-1,j,k)] ;
            dtau = tau0 - tau[tau.index(i,j,k)] ;

            heat -= Lband[k] * exp(-tau0) * expm1(dtau) ;
        }
        heat *= g.dsin_th(j) / (4 * M_PI * g.volume(i,j)) ;

        heating(i,j) = heat ;
    }
} 



__global__ void _volumetric_heating_with_scattering(GridRef g, 
                                                    int num_wavelengths, const double* Lband,
                                                    Field3DConstRef<double> tau,
                                                    Field3DConstRef<double> kappa_abs,
                                                    Field3DConstRef<double> kappa_sca,
                                                    FieldRef<double> heating,
                                                    Field3DRef<double> scattering) {

    int j = threadIdx.x + blockIdx.x*blockDim.x ;
    int i = threadIdx.y + blockIdx.y*blockDim.y ;

    if (i < g.NR + 2*g.Nghost && j < g.Nphi + 2*g.Nghost) {

        double heat = 0 ;
        double tau0 = 0, dtau, albedo, term ;
        double inv_area = g.dsin_th(j) / (4 * M_PI * g.volume(i,j)) ;
        for (int k=0; k < num_wavelengths; k++) {
            if (i > 0) tau0 = tau(i-1,j,k) ;
            dtau = tau0 - tau(i,j,k) ;

            albedo = kappa_sca(i,j,k) + kappa_abs(i,j,k) ;
            if (albedo > 0) albedo = kappa_sca(i,j,k) / albedo ;

            term = - inv_area * Lband[k] * exp(-tau0) * expm1(dtau) ;

            heat += term * (1 - albedo) ;
            scattering(i,j,k) = term * albedo ;
        }
        heating(i,j) = heat ;
    }
} 

__global__ void _volumetric_heating_with_scattering(GridRef g, 
                                                    int num_wavelengths, const double* Lband,
                                                    Field3DConstRef<double> tau, FieldConstRef<double> tau_inner,
                                                    Field3DConstRef<double> kappa_abs,
                                                    Field3DConstRef<double> kappa_sca,
                                                    FieldRef<double> heating,
                                                    Field3DRef<double> scattering) {

    int j = threadIdx.x + blockIdx.x*blockDim.x ;
    int i = threadIdx.y + blockIdx.y*blockDim.y ;

    if (i < g.NR + 2*g.Nghost && j < g.Nphi + 2*g.Nghost) {

        double heat = 0 ;
        double tau0 = 0, dtau, albedo, term;
        double inv_area = g.dsin_th(j) / (4 * M_PI * g.volume(i,j)) ;
        for (int k=0; k < num_wavelengths; k++) {
            if (i > 0) tau0 = tau(i-1,j,k) ;
            dtau = tau0 - tau(i,j,k) ;

            albedo = kappa_sca(i,j,k) + kappa_abs(i,j,k) ;
            if (albedo > 0) albedo = kappa_sca(i,j,k) / albedo ;

            term = - inv_area * Lband[k] * exp(-(tau0+tau_inner(j,k))) * expm1(dtau) ;

            // if (i==0 && k==20) {printf("%d %g\n", j,term);}

            heat += term * (1 - albedo) ;
            scattering(i,j,k) = term * albedo ;
        }
        heating(i,j) = heat ;
    }
} 

__global__ void _pressure_force_with_scattering(GridRef g, 
                                                    int num_wavelengths, const double* Lband,
                                                    Field3DConstRef<double> tau,
                                                    Field3DRef<double> f_pressure, Field3DConstRef<Prims> qd, DSHARP_opacsRef opacs) {

    int j = threadIdx.x + blockIdx.x*blockDim.x ;
    int i = threadIdx.y + blockIdx.y*blockDim.y ;
    int k = threadIdx.z + blockIdx.z*blockDim.z ;

    if (i < g.NR + 2*g.Nghost && j < g.Nphi + 2*g.Nghost && k < f_pressure.Nd) {

        double f = 0 ;
        double tau0 = 0;
        for (int l=0; l < num_wavelengths; l++) {

            if (i > 0) tau0 = tau(i-1,j,l) ;

            f += (Lband[l]/(4*M_PI*g.rc(i,j)*g.rc(i,j))) * exp(-tau0) * (qd(i,j,k).rho * (opacs.k_abs(k,l)+opacs.k_sca(k,l))) / c_light;
        }
        f_pressure(i,j,k) = f ;
        
    }
} 
__global__ void _pressure_force_with_scattering(GridRef g, 
                                                    int num_wavelengths, const double* Lband,
                                                    Field3DConstRef<double> tau, FieldConstRef<double> tau_inner,
                                                    Field3DRef<double> f_pressure, Field3DConstRef<Prims> qd, DSHARP_opacsRef opacs) {

    int j = threadIdx.x + blockIdx.x*blockDim.x ;
    int i = threadIdx.y + blockIdx.y*blockDim.y ;
    int k = threadIdx.z + blockIdx.z*blockDim.z ;

    if (i < g.NR + 2*g.Nghost && j < g.Nphi + 2*g.Nghost && k < f_pressure.Nd) {

        double f = 0 ;
        double tau0 = 0;
        for (int l=0; l < num_wavelengths; l++) {

            if (i > 0) tau0 = tau(i-1,j,l) ;

            f += (Lband[l]/(4*M_PI*g.rc(i,j)*g.rc(i,j))) * exp(-(tau0+tau_inner(j,l))) * (qd(i,j,k).rho * (opacs.k_abs(k,l)+opacs.k_sca(k,l))) / c_light;
        }
        f_pressure(i,j,k) = f ;
        
    }
} 



void _compute_stellar_heating_from_tau(const Star& star, const Grid& g, 
                                       const Field3D<double>& tau, Field<double>& heating) {

    // Step 3: Compute volumetric heating rates
    dim3 threads2(32, 32) ;
    dim3 blocks2((g.Nphi + 2*g.Nghost + 31)/32, (g.NR + 2*g.Nghost+31)/32) ;

    _volumetric_heating<<<blocks2, threads2>>>(g, star.num_wle, star.Lband.get(),
                                               tau, heating) ;
    check_CUDA_errors("_volumetric_heating") ;
                                    
}

void _compute_stellar_heating_with_scattering_from_tau(const Star& star, const Grid& g, 
                                                       const Field3D<double>& tau, 
                                                       const Field3D<double>& kappa_abs,
                                                       const Field3D<double>& kappa_sca,
                                                       Field<double>& heating,
                                                       Field3D<double>& scattering) {

    // Step 3: Compute volumetric heating rates
    dim3 threads2(32, 32) ;
    dim3 blocks2((g.Nphi + 2*g.Nghost + 31)/32, (g.NR + 2*g.Nghost+31)/32) ;

    _volumetric_heating_with_scattering<<<blocks2, threads2>>>
        (g, star.num_wle, star.Lband.get(), tau, kappa_abs, kappa_sca, heating, scattering) ;
    check_CUDA_errors("_volumetric_heating_with_scattering") ;
}

void _compute_stellar_heating_with_scattering_from_tau(const Star& star, const Grid& g, 
                                                       const Field3D<double>& tau, const Field<double>& tau_inner, 
                                                       const Field3D<double>& kappa_abs,
                                                       const Field3D<double>& kappa_sca,
                                                       Field<double>& heating,
                                                       Field3D<double>& scattering) {

    // Step 3: Compute volumetric heating rates
    dim3 threads2(32, 32) ;
    dim3 blocks2((g.Nphi + 2*g.Nghost + 31)/32, (g.NR + 2*g.Nghost+31)/32) ;

    _volumetric_heating_with_scattering<<<blocks2, threads2>>>
        (g, star.num_wle, star.Lband.get(), tau, tau_inner, kappa_abs, kappa_sca, heating, scattering) ;
    check_CUDA_errors("_volumetric_heating_with_scattering") ;
}

void _compute_stellar_pressure_with_scattering_from_tau(const Star& star, const Grid& g, const Field3D<double>& tau, 
                                                Field3D<double>& f_pressure, const Field3D<Prims>& qd, DSHARP_opacs& opacs) {
    dim3 threads2(32, 32, 1) ;
    dim3 blocks2((g.Nphi + 2*g.Nghost + 31)/32, (g.NR + 2*g.Nghost+31)/32, f_pressure.Nd) ;

    _pressure_force_with_scattering<<<blocks2, threads2>>>
        (g, star.num_wle, star.Lband.get(), tau, f_pressure, qd, opacs) ;
}

void _compute_stellar_pressure_with_scattering_from_tau(const Star& star, const Grid& g, const Field3D<double>& tau, const Field<double>& tau_inner, 
                                                Field3D<double>& f_pressure, const Field3D<Prims>& qd, DSHARP_opacs& opacs) {
    dim3 threads2(32, 32, 1) ;
    dim3 blocks2((g.Nphi + 2*g.Nghost + 31)/32, (g.NR + 2*g.Nghost+31)/32, f_pressure.Nd) ;

    _pressure_force_with_scattering<<<blocks2, threads2>>>
        (g, star.num_wle, star.Lband.get(), tau, tau_inner, f_pressure, qd, opacs) ;
}


void compute_stellar_heating(const Star& star, const Grid& g, 
                             const Field3D<double>& rhokappa, 
                             Field<double>& heating) {

    CodeTiming::BlockTimer timing_block = 
        timer->StartNewTimer("compute_stellar_heating") ;
    
    // Step 0: Decomposition for gpu
    int nk = 1 ;
    while (nk < star.num_wle && nk < 32)
        nk *= 2 ;
    int nj = 1024 / nk ;

    dim3 threads(nk, nj, 1) ;
    dim3 blocks((star.num_wle +  nk-1)/nk, 
                (g.Nphi +  2*g.Nghost + nj-1)/nj, 
                 g.NR +  2*g.Nghost) ;

                               
    // Step 1: compute optical depths                                
    Field3D<double> tau = create_field3D<double>(g, star.num_wle) ;
    _cell_optical_depth_tab<<<blocks,threads>>>(g, star.num_wle, 
                                                rhokappa, tau) ;
 
    check_CUDA_errors("_cell_optical_depth") ;
    Reduction::scan_R_sum(g, tau) ;

    // Step 2: compute heating using optical depths
    _compute_stellar_heating_from_tau(star, g, tau, heating) ;
}


void compute_stellar_heating_with_scattering(const Star& star, const Grid& g, 
                                             const Field3D<double>& rhok_abs,
                                             const Field3D<double>& rhok_sca, 
                                             Field<double>& heating,
                                             Field3D<double>& scattering) {

    CodeTiming::BlockTimer timing_block = 
        timer->StartNewTimer("compute_stellar_heating_with_scattering") ;

    // Step 0: Decomposition for gpu
    int nk = 1 ;
    while (nk < star.num_wle && nk < 32)
    nk *= 2 ;
    int nj = 1024 / nk ;

    dim3 threads(nk, nj, 1) ;
    dim3 blocks((star.num_wle +  nk-1)/nk, 
    (g.Nphi +  2*g.Nghost + nj-1)/nj, 
    g.NR +  2*g.Nghost) ;

          
    // Step 1: compute optical depths                                
    Field3D<double> tau = create_field3D<double>(g, star.num_wle) ;
    _cell_optical_depth_tab_with_scattering<<<blocks,threads>>>
        (g, star.num_wle, rhok_abs, rhok_sca, tau) ;

    check_CUDA_errors("_cell_optical_depth") ;
    Reduction::scan_R_sum(g, tau) ;

    // Step 2: compute heating using optical depths
    _compute_stellar_heating_with_scattering_from_tau
        (star, g, tau, rhok_abs, rhok_sca, heating, scattering) ;
}

void compute_radiation_pressure_with_scattering(const Star& star, const Grid& g, 
                                             const Field3D<double>& rhok_abs, const Field3D<double>& rhok_sca, 
                                             Field3D<double>& f_pressure, const Field3D<Prims>& qd, DSHARP_opacs& opacs) {

    CodeTiming::BlockTimer timing_block = 
        timer->StartNewTimer("compute_radiation_pressure_with_scattering") ;

    // Step 0: Decomposition for gpu
    int nk = 1 ;
    while (nk < star.num_wle && nk < 32)
    nk *= 2 ;
    int nj = 1024 / nk ;

    dim3 threads(nk, nj, 1) ;
    dim3 blocks((star.num_wle +  nk-1)/nk, 
    (g.Nphi +  2*g.Nghost + nj-1)/nj, 
    g.NR +  2*g.Nghost) ;

          
    // Step 1: compute optical depths                                
    Field3D<double> tau = create_field3D<double>(g, star.num_wle) ;
    _cell_optical_depth_tab_with_scattering<<<blocks,threads>>>
        (g, star.num_wle, rhok_abs, rhok_sca, tau) ;

    check_CUDA_errors("_cell_optical_depth") ;
    Reduction::scan_R_sum(g, tau) ;

    _compute_stellar_pressure_with_scattering_from_tau
        (star, g, tau, f_pressure, qd, opacs);
    
}

void _interpolate_tau_inner(Field3D<double>& tau_inner, Field<double>& tau_inner_interp, double* ts, double t, int NZ, int Nt) {

    // int i = threadIdx.x + blockIdx.x*blockDim.x ;
    // int j = threadIdx.y + blockIdx.y*blockDim.y ;

    // if (i < NZ && j < opacs.n_lam) {

    // std::ofstream f("./iscripts/interpedtau.txt",std::fstream::app);

    for (int i=0; i< NZ; i++) {
        for (int j=0; j<tau_inner.Nd; j++) {

            std::vector<double> tvec(Nt+2), tauvec(Nt+2);

            tvec[0] = -year;
            tauvec[0] = log(tau_inner(0,i,j));
            tvec[Nt+1] = ts[Nt-1]+year;
            tauvec[Nt+1] = log(tau_inner(Nt-1,i,j));

            for (int k=1; k<Nt+1; k++) {
                tvec[k] = ts[k-1];
                tauvec[k] = log(tau_inner(k-1,i,j));
                
            }
            
            PchipInterpolator<1> interp(tvec, tauvec);
            tau_inner_interp(i,j) = exp(interp(t));
            // f << tau_inner_interp(i,j) << "\n";
            // if (tau_inner_interp(i,j) > 100) {
            //     tau_inner_interp(i,j) = 100;
            // }
        }
    }
    // f.close();
}

__global__ void _add_tau_inner(FieldRef<double> tau_inner_interp, Field3DRef<double> tau, int n_lam, int NZ) {

    int i = threadIdx.x + blockIdx.x*blockDim.x ;
    int j = threadIdx.y + blockIdx.y*blockDim.y ;

    if (i < NZ && j < n_lam) {
        tau(0,i,j) += tau_inner_interp(i,j);
    }
}

void compute_stellar_heating_with_scattering_with_inner_disc(const Star& star, const Grid& g, 
                                             const Field3D<double>& rhok_abs,
                                             const Field3D<double>& rhok_sca, 
                                             Field<double>& heating,
                                             Field3D<double>& scattering,
                                             Field3D<double>& tau_inner, double t, CudaArray<double>& ts, int NZ, int Nt) {

    CodeTiming::BlockTimer timing_block = 
        timer->StartNewTimer("compute_stellar_heating_with_scattering") ;

    // Step 0: Decomposition for gpu
    int nk = 1 ;
    while (nk < star.num_wle && nk < 32)
    nk *= 2 ;
    int nj = 1024 / nk ;

    dim3 threads(nk, nj, 1) ;
    dim3 blocks((star.num_wle +  nk-1)/nk, 
    (g.Nphi +  2*g.Nghost + nj-1)/nj, 
    g.NR +  2*g.Nghost) ;

          
    // Step 1: compute optical depths                                
    Field3D<double> tau = create_field3D<double>(g, star.num_wle) ;
    Field<double> tau_inner_interp = Field<double>(NZ, star.num_wle) ;
    _cell_optical_depth_tab_with_scattering<<<blocks,threads>>>
        (g, star.num_wle, rhok_abs, rhok_sca, tau) ;

    check_CUDA_errors("_cell_optical_depth") ;

    _interpolate_tau_inner(tau_inner, tau_inner_interp, ts.get(), t, NZ, Nt);

    Reduction::scan_R_sum(g, tau) ;

    // Step 2: compute heating using optical depths
    _compute_stellar_heating_with_scattering_from_tau
        (star, g, tau, tau_inner_interp, rhok_abs, rhok_sca, heating, scattering) ;
}

void compute_radiation_pressure_with_scattering_with_inner_disc(const Star& star, const Grid& g, 
                                             const Field3D<double>& rhok_abs, const Field3D<double>& rhok_sca, 
                                             Field3D<double>& f_pressure, const Field3D<Prims>& qd, DSHARP_opacs& opacs, 
                                             Field<double>& tau_inner, int NZ) {

    CodeTiming::BlockTimer timing_block = 
        timer->StartNewTimer("compute_radiation_pressure_with_scattering") ;

    // Step 0: Decomposition for gpu
    int nk = 1 ;
    while (nk < star.num_wle && nk < 32)
    nk *= 2 ;
    int nj = 1024 / nk ;

    dim3 threads(nk, nj, 1) ;
    dim3 blocks((star.num_wle +  nk-1)/nk, 
    (g.Nphi +  2*g.Nghost + nj-1)/nj, 
    g.NR +  2*g.Nghost) ;

          
    // Step 1: compute optical depths                                
    Field3D<double> tau = create_field3D<double>(g, star.num_wle) ;
    _cell_optical_depth_tab_with_scattering<<<blocks,threads>>>
        (g, star.num_wle, rhok_abs, rhok_sca, tau) ;

    check_CUDA_errors("_cell_optical_depth") ;

    Reduction::scan_R_sum(g, tau) ;

    _compute_stellar_pressure_with_scattering_from_tau
        (star, g, tau, tau_inner, f_pressure, qd, opacs);
    
}

__global__ void add_viscous_heating_device(double GM, GridRef g, double alpha,
                                           FieldConstRef<double> rho, FieldConstRef<double> cs2,
                                           FieldRef<double> heating) {

    int i = threadIdx.y + blockIdx.y * blockDim.y ;
    int j = threadIdx.x + blockIdx.x * blockDim.x ;

    if (i < g.NR + 2*g.Nghost && j < g.Nphi + 2*g.Nghost) {

        double r3 = g.rc(i,j)*g.rc(i,j)*g.rc(i,j) ;
        double omega = sqrt(GM / r3) ;

        heating[heating.index(i,j)] += 
            2.25 * alpha * rho[rho.index(i,j)] * cs2[cs2.index(i,j)] * omega ;
    }
}



void add_viscous_heating(const Star& star, const Grid &grid, double alpha, 
                         const Field<double>&rho, const Field<double>&cs2, 
                         Field<double>& heating) {
    CodeTiming::BlockTimer timing_block = 
        timer->StartNewTimer("add_viscous_heating") ;

    dim3 threads(32,32) ;
    dim3 blocks((grid.Nphi+2*grid.Nghost+31)/32, (grid.NR + 2*grid.Nghost+31)/32) ;

    add_viscous_heating_device<<<blocks, threads>>>(star.GM, grid, alpha, rho, cs2, heating) ;
    check_CUDA_errors("add_viscous_heating_device") ;
}

__global__ void _rho_from_wg(GridRef g, FieldRef<double> rho, FieldConstRef<Prims> w_g) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) { 
            rho(i,j) = w_g(i,j).rho;
        }
    }

}


void add_viscous_heating(const Star& star, const Grid &grid, double alpha, 
                         const Field<Prims>& w_g, const Field<double>&cs2, 
                         Field<double>& heating) {
                
    Field<double> rho = create_field<double>(grid);
    
    dim3 threadsrho(32,32,1);
    dim3 blocksrho((grid.NR+2*grid.Nghost+31)/32, (grid.Nphi+2*grid.Nghost+31)/32 );

    _rho_from_wg<<<blocksrho,threadsrho>>>(grid, rho, w_g); 
    
    CodeTiming::BlockTimer timing_block = 
        timer->StartNewTimer("add_viscous_heating") ;

    dim3 threads(32,32) ;
    dim3 blocks((grid.Nphi+2*grid.Nghost+31)/32, (grid.NR + 2*grid.Nghost+31)/32) ;

    add_viscous_heating_device<<<blocks, threads>>>(star.GM, grid, alpha, rho, cs2, heating) ;
    check_CUDA_errors("add_viscous_heating_device") ;
}


__global__ void add_viscous_heating_device(double GM, GridRef g,
                                           FieldConstRef<Prims> wg, double* nu,
                                           FieldRef<double> heating) {

    int i = threadIdx.y + blockIdx.y * blockDim.y ;
    int j = threadIdx.x + blockIdx.x * blockDim.x ;

    if (i < g.NR + 2*g.Nghost && j < g.Nphi + 2*g.Nghost) {

        double r3 = g.rc(i,j)*g.rc(i,j)*g.rc(i,j) ;
        double omega2 = GM / r3 ;

        heating[heating.index(i,j)] += 
            2.25 * nu[i] * wg(i,j).rho * omega2 ;
    }
}


void add_viscous_heating(const Star& star, const Grid &grid,
                         const Field<Prims>& w_g, const CudaArray<double>& nu, 
                         Field<double>& heating) {
                

    CodeTiming::BlockTimer timing_block = 
        timer->StartNewTimer("add_viscous_heating") ;

    dim3 threads(32,32) ;
    dim3 blocks((grid.Nphi+2*grid.Nghost+31)/32, (grid.NR + 2*grid.Nghost+31)/32) ;

    add_viscous_heating_device<<<blocks, threads>>>(star.GM, grid, w_g, nu.get(), heating) ;
    check_CUDA_errors("add_viscous_heating_device") ;
}

__global__ void add_viscous_heating_device(double GM, GridRef g,
                                           double* Sig, double* nu,
                                           FieldRef<double> heating) {

    int i = threadIdx.x ;

    if (i < g.NR + 2*g.Nghost) {

        double r3 = g.rc(i,g.Nghost)*g.rc(i,g.Nghost)*g.rc(i,g.Nghost) ;
        double omega2 = GM / r3 ;

        heating[heating.index(i,g.Nghost)] += 
            2.25 * nu[i] * Sig[i] * omega2 / g.dZe(i, g.Nghost);
    }
}


void add_viscous_heating(const Star& star, const Grid &grid,
                         const CudaArray<double>& Sig, const CudaArray<double>& nu, 
                         Field<double>& heating) {
                
    // dim3 threads(1024,1,1);
    // dim3 blocks((grid.NR+2*grid.Nghost+1023)/1024);

    CodeTiming::BlockTimer timing_block = 
        timer->StartNewTimer("add_viscous_heating") ;

    add_viscous_heating_device<<<1, 1024>>>(star.GM, grid, Sig.get(), nu.get(), heating) ;
    check_CUDA_errors("add_viscous_heating_device") ;
}
