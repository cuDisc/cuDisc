#include "dustdynamics.h"
#include "cuda_runtime.h"
#include "cuzzi_opacs.h"
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>

std::string opacity_dir = OPAC_DIR;

enum InterpType {
    loglog = 1 << 0,
    loglin = 1 << 1
};

void interpolate_optconsts(CudaArray<RefIndx>& opt, int interp_type, int n_lam, double* lam,
                            std::vector<double> lam_inp, std::vector<double> n_inp, std::vector<double> k_inp) {

    PchipInterpolator<2> interp(lam_inp, n_inp, k_inp);

    int n_lam_inp = lam_inp.size();

    if (interp_type & InterpType::loglog) {
        for (int i=0; i<n_lam; i++) {
            if (std::log10(lam[i]) < lam_inp[0]) {
                opt[i].n = std::pow(10.,n_inp[0] + (std::log10(lam[i])-lam_inp[0]) * (n_inp[1]-n_inp[0]) / (lam_inp[1]-lam_inp[0]));
                opt[i].k = std::pow(10.,k_inp[0] + (std::log10(lam[i])-lam_inp[0]) * (k_inp[1]-k_inp[0]) / (lam_inp[1]-lam_inp[0]));
            }
            else if (std::log10(lam[i]) > lam_inp[n_lam_inp-1]) {
                opt[i].n = std::pow(10.,n_inp[n_lam_inp-1] + (std::log10(lam[i])-lam_inp[n_lam_inp-1]) * (n_inp[n_lam_inp-1]-n_inp[n_lam_inp-2]) / (lam_inp[n_lam_inp-1]-lam_inp[n_lam_inp-2]));
                opt[i].k = std::pow(10.,k_inp[n_lam_inp-1] + (std::log10(lam[i])-lam_inp[n_lam_inp-1]) * (k_inp[n_lam_inp-1]-k_inp[n_lam_inp-2]) / (lam_inp[n_lam_inp-1]-lam_inp[n_lam_inp-2]));
            }
            else {
                auto interp_res = interp(std::log10(lam[i]));
                opt[i].n = std::pow(10.,interp_res[0]);
                opt[i].k = std::pow(10.,interp_res[1]);
            }
        }
    }
    else {
        for (int i=0; i<n_lam; i++) {
            if (std::log10(lam[i]) < lam_inp[0]) {
                opt[i].n = n_inp[0] + (std::log10(lam[i])-lam_inp[0]) * (n_inp[1]-n_inp[0]) / (lam_inp[1]-lam_inp[0]);
                opt[i].k = std::pow(10.,k_inp[0] + (std::log10(lam[i])-lam_inp[0]) * (k_inp[1]-k_inp[0]) / (lam_inp[1]-lam_inp[0]));
            }
            else if (std::log10(lam[i]) > lam_inp[n_lam_inp-1]) {
                opt[i].n = n_inp[n_lam_inp-1] + (std::log10(lam[i])-lam_inp[n_lam_inp-1]) * ((n_inp[n_lam_inp-1])-(n_inp[n_lam_inp-2])) / (lam_inp[n_lam_inp-1]-lam_inp[n_lam_inp-2]);
                opt[i].k = std::pow(10.,k_inp[n_lam_inp-1] + (std::log10(lam[i])-lam_inp[n_lam_inp-1]) * ((k_inp[n_lam_inp-1])-(k_inp[n_lam_inp-2])) / (lam_inp[n_lam_inp-1]-lam_inp[n_lam_inp-2]));
            }
            else {
                auto interp_res = interp(std::log10(lam[i]));
                opt[i].n = interp_res[0];
                opt[i].k = std::pow(10.,interp_res[1]);
            }
        }
    }
}

CudaArray<RefIndx> load_pollack(int n_lam, double* lam, std::string comptype) {

    int n_lam_inp = 100;
    std::string filename = opacity_dir + "/optical_constants/pollack1994/P94-" + comptype + ".lnk";
    std::ifstream opt(filename);

    if (!opt) {
        throw std::runtime_error("Could not find/open opacity file " + filename) ;
    } 
    
    std::vector<double> lam_inp(n_lam_inp), n_inp(n_lam_inp), k_inp(n_lam_inp);
    CudaArray<RefIndx> optconst = make_CudaArray<RefIndx>(n_lam);

    double x;

    if (comptype == "iron" || comptype == "troilite") {
        for (int i=0; i<n_lam_inp; i++) {
            opt >> x; 
            lam_inp[i] = std::log10(x*1.e4);
            opt >> x; 
            n_inp[i] = std::log10(x);
            opt >> x;
            k_inp[i] = std::log10(x);
        }
        interpolate_optconsts(optconst, InterpType::loglog, n_lam, lam, lam_inp, n_inp, k_inp);
    }
    else {
        for (int i=0; i<n_lam_inp; i++) {
            opt >> x; 
            lam_inp[i] = std::log10(x*1.e4);
            opt >> n_inp[i];
            opt >> x;
            k_inp[i] = std::log10(x);
        }
        interpolate_optconsts(optconst, InterpType::loglin, n_lam, lam, lam_inp, n_inp, k_inp);
    }
    return optconst;
}

CudaArray<RefIndx> load_henning(int n_lam, double* lam, std::string comptype) {

    int n_lam_inp = 113;
    std::string filename = opacity_dir + "/optical_constants/henning/new/" + comptype + ".lnk";
    std::ifstream opt(filename);

    if (!opt) {
        throw std::runtime_error("Could not find/open opacity file " + filename) ;
    } 
    
    std::vector<double> lam_inp(n_lam_inp), n_inp(n_lam_inp), k_inp(n_lam_inp);
    CudaArray<RefIndx> optconst = make_CudaArray<RefIndx>(n_lam);

    double x;

    if (comptype == "ironk" || comptype == "troilitek") {
        for (int i=0; i<n_lam_inp; i++) {
            opt >> x; 
            lam_inp[i] = std::log10(x);
            opt >> x; 
            n_inp[i] = std::log10(x);
            opt >> x;
            k_inp[i] = std::log10(x);
        }
        interpolate_optconsts(optconst, InterpType::loglog, n_lam, lam, lam_inp, n_inp, k_inp);
    }
    else {
        for (int i=0; i<n_lam_inp; i++) {
            opt >> x; 
            lam_inp[i] = std::log10(x);
            opt >> n_inp[i];
            opt >> x;
            k_inp[i] = std::log10(x);
        }
        interpolate_optconsts(optconst, InterpType::loglin, n_lam, lam, lam_inp, n_inp, k_inp);
    }
    return optconst;
}

CudaArray<RefIndx> load_warren(int n_lam, double* lam) {

    int n_lam_inp = 486;
    std::string filename = opacity_dir + "/optical_constants/warren/IOP_2008_ASCIItable.dat";
    std::ifstream opt(filename);

    if (!opt) {
        throw std::runtime_error("Could not find/open opacity file " + filename) ;
    } 
    
    std::vector<double> lam_inp(n_lam_inp), n_inp(n_lam_inp), k_inp(n_lam_inp);
    CudaArray<RefIndx> optconst = make_CudaArray<RefIndx>(n_lam);

    double x;

    for (int i=0; i<n_lam_inp; i++) {
        opt >> x; 
        lam_inp[i] = std::log10(x);
        opt >> n_inp[i];
        opt >> x;
        k_inp[i] = std::log10(x);
    }
    interpolate_optconsts(optconst, InterpType::loglin, n_lam, lam, lam_inp, n_inp, k_inp);
    
    return optconst;
}

CudaArray<RefIndx> load_gavdush(int n_lam, double* lam) {

    // int n_lam_inp = 2146;
    int n_lam_inp = 2344;
    // std::string filename = opacity_dir + "/optical_constants/gavdush/eps_CO.dat";
    std::string filename = opacity_dir + "/optical_constants/gavdush/CO.lnk";
    std::ifstream opt(filename);

    if (!opt) {
        throw std::runtime_error("Could not find/open opacity file " + filename) ;
    } 
    // for (int i=0; i<2; i++){
    //     opt.ignore(std::numeric_limits<std::streamsize>::max(), opt.widen('\n'));
    // }
    
    std::vector<double> lam_inp(n_lam_inp), n_inp(n_lam_inp), k_inp(n_lam_inp);
    CudaArray<RefIndx> optconst = make_CudaArray<RefIndx>(n_lam);

    double x,y;

    for (int i=0; i<n_lam_inp; i++) {
        opt >> x; 
        lam_inp[n_lam_inp-1-i] = std::log10(x);
        opt >> x;
        opt >> y;
        n_inp[n_lam_inp-1-i] = x;//std::sqrt((std::sqrt(x*x+y*y)+x)/2.);
        k_inp[n_lam_inp-1-i] = std::log10(y);//std::log10(std::sqrt((std::sqrt(x*x+y*y)-x)/2.));
    }
    interpolate_optconsts(optconst, InterpType::loglin, n_lam, lam, lam_inp, n_inp, k_inp);
    
    return optconst;
}

CudaArray<RefIndx> load_draine(int n_lam, double* lam, std::string comptype) {

    int n_lam_inp;
    std::string filename;
    if (comptype == "astrosilicates") {
        n_lam_inp = 837;
        filename = opacity_dir + "/optical_constants/draine/callindex.out_silD03";
    }
    else if (comptype == "gpa001" || comptype == "gpa01") {
        n_lam_inp = 386;
        if (comptype == "gpa001") 
            filename = opacity_dir + "/optical_constants/draine/callindex.out_CpaD03_0.01";
        else 
            filename = opacity_dir + "/optical_constants/draine/callindex.out_CpaD03_0.1";
    }
    else {
        n_lam_inp = 383;
        if (comptype == "gpe001") 
            filename = opacity_dir + "/optical_constants/draine/callindex.out_CpeD03_0.01";
        else 
            filename = opacity_dir + "/optical_constants/draine/callindex.out_CpeD03_0.1";
    }
    std::ifstream opt(filename);
    for (int i=0; i<5; i++){
        opt.ignore(std::numeric_limits<std::streamsize>::max(), opt.widen('\n'));
    }

    if (!opt) {
        throw std::runtime_error("Could not find/open opacity file " + filename) ;
    } 
    
    std::vector<double> lam_inp(n_lam_inp), n_inp(n_lam_inp), k_inp(n_lam_inp);
    CudaArray<RefIndx> optconst = make_CudaArray<RefIndx>(n_lam);

    double x;

    for (int i=0; i<n_lam_inp; i++) {
        opt >> x;  
        lam_inp[n_lam_inp-1-i] = std::log10(x);
        opt >> x;  
        opt >> x;  
        opt >> x;  
        n_inp[n_lam_inp-1-i] = x+1.;
        opt >> x;
        k_inp[n_lam_inp-1-i] = std::log10(x);
    }
    interpolate_optconsts(optconst, InterpType::loglin, n_lam, lam, lam_inp, n_inp, k_inp);
    
    return optconst;
}


CuzziComp::CuzziComp(int n_lam, double* lam) {
    std::cout << "Loading Cuzzi composition\n";
    ice.dens = 0.9, ice.mf = 0.399, ice.opt = load_pollack(n_lam, lam, "waterice");
    sil.dens = 3.4, sil.mf = 0.241, sil.opt = load_pollack(n_lam, lam, "orthopyroxene");
    FeS.dens = 4.8, FeS.mf = 0.055, FeS.opt = load_pollack(n_lam, lam, "troilite");
    Fe.dens = 7.8, Fe.mf = 0.009, Fe.opt = load_pollack(n_lam, lam, "iron");
    org.dens = 1.5, org.mf = 0.296, org.opt = load_pollack(n_lam, lam, "organics");
}

DSHARPComp::DSHARPComp(int n_lam, double* lam) {
    std::cout << "Loading DSHARP composition\n";
    ice.dens = 0.92, ice.mf = 0.2, ice.opt = load_warren(n_lam, lam);
    sil.dens = 3.3, sil.mf = 0.329, sil.opt = load_draine(n_lam, lam, "astrosilicates");
    FeS.dens = 4.83, FeS.mf = 0.074336, FeS.opt = load_henning(n_lam, lam, "troilitek");
    org.dens = 1.5, org.mf = 0.396648, org.opt = load_henning(n_lam, lam, "organicsk");
}

DSHARPwCOComp::DSHARPwCOComp(int n_lam, double* lam) {
    std::cout << "Loading DSHARP w/ CO composition\n";
    ice.dens = 0.92, ice.mf = 0.2, ice.opt = load_warren(n_lam, lam);
    sil.dens = 3.3, sil.mf = 0.329, sil.opt = load_draine(n_lam, lam, "astrosilicates");
    FeS.dens = 4.83, FeS.mf = 0.074336, FeS.opt = load_henning(n_lam, lam, "troilitek");
    org.dens = 1.5, org.mf = 0.396648, org.opt = load_henning(n_lam, lam, "organicsk");
    CO.dens = 0.89, CO.mf = 0., CO.opt = load_gavdush(n_lam, lam);
}

template<class CompMix>
CuzziOpacs<CompMix>::CuzziOpacs(int _n_a, int _n_lam, double lam_min, double lam_max) {

    n_a = _n_a;
    n_lam = _n_lam;
    
    a_ptr = make_CudaArray<double>(n_a);
    lam_ptr = make_CudaArray<double>(n_lam);
    double dloglam = (std::log10(lam_max) - std::log10(lam_min)) / static_cast<double>(n_lam-1);
    for (int i=0; i<n_lam; i++) {
        lam_ptr[i] = std::pow(10., std::log10(lam_min) + dloglam*i); 
    }
    k_abs_ptr = make_CudaArray<double>(n_a*n_lam);
    k_sca_ptr = make_CudaArray<double>(n_a*n_lam);
    k_abs_g_ptr = make_CudaArray<double>(n_lam);
    k_sca_g_ptr = make_CudaArray<double>(n_lam);

    // Load optical constants

    comp = CompMix(n_lam, lam_ptr.get());
}

template<class CompMix>
void CuzziOpacs<CompMix>::calc_opacs(SizeGrid& sizes, double por) {

    CudaArray<double> vfs = make_CudaArray<double>(comp.n_comp);

    // Calc optical constants via Garnett EMT

    double rho_1 = 0;
    for (int i=0; i<comp.n_comp; i++) {rho_1 += comp[i].mf/comp[i].dens;}
    double rho_av = 1./rho_1;
    std::cout << "Bulk density: " << rho_av << " gcm-3";

    for (int i=0; i<comp.n_comp; i++) {
        vfs[i] = (1.-por)*rho_av*comp[i].mf/comp[i].dens;
    }

    for (int i=0; i<n_lam; i++) {
        CudaArray<double> sigs = make_CudaArray<double>(comp.n_comp);
        CudaArray<double> gams = make_CudaArray<double>(comp.n_comp);
        RefIndx eps, n;
        double vfsig=0, vf2sig2gam2=0, vfgam=0;

        for (int j=0; j<comp.n_comp; j++) {
            double nr2 = comp[j].opt[i].n*comp[j].opt[i].n;
            double ni2 = comp[j].opt[i].k*comp[j].opt[i].k;
            sigs[j] = ((nr2-ni2-1.)*(nr2-ni2+2.)+4.*nr2*ni2)/((nr2-ni2+2.)*(nr2-ni2+2.)+4.*nr2*ni2);
            gams[j] = 2.*comp[j].opt[i].n*comp[j].opt[i].k/((nr2-ni2+2.)*(nr2-ni2+2.)+4.*nr2*ni2);

            vfsig += vfs[j]*sigs[j];
            vfgam += vfs[j]*gams[j];
        }

        for (int j=0; j<comp.n_comp; j++) {
            for (int k=0; k<comp.n_comp; k++) {
                vf2sig2gam2 += vfs[j]*vfs[k]*(sigs[j]*sigs[k]+9.*gams[j]*gams[k]);
            }
        }
        double D = 1.-2.*vfsig+vf2sig2gam2;
        eps.n = (1.+vfsig-2.*vf2sig2gam2)/D;
        eps.k = 9.*vfgam/D;

        double eps_quad = std::sqrt(eps.n*eps.n+eps.k*eps.k)/2.;
        n.n = std::sqrt(eps_quad+eps.n/2.);
        n.k = std::sqrt(eps_quad-eps.n/2.);

        // std::cout << n.n << ", ";

        for (int k=0; k<sizes.size(); k++) {
            double x = 2.*M_PI*sizes.centre_size(k)/(lam(i)/1.e4);

            double Q_a = std::min(1.,12.*x*eps.k/((eps.n+2.)*(eps.n+2.)+eps.k*eps.k));
            double Q_s, g;
            
            if (x < 1.3) {
                Q_s = 8./3.*std::pow(x,4.)*((eps.n-1.)*(eps.n-1.)+eps.k*eps.k)/((eps.n+2.)*(eps.n+2.)+eps.k*eps.k);
            }
            else {
                Q_s = 2.*x*x*(n.n-1.)*(n.n-1.)*(1.+(n.k/(n.n-1.))*(n.k/(n.n-1.)));
            }

            if (n.k < 1.) {
                if (x < 3.) { g = 0.7*(x/3.)*(x/3.); }
                else { g = 0.7; }
            }
            else {
                if (x < 3.) { g = -0.2; }
                else { g = 0.5; } 
            }

            Q_s = std::min(Q_s*(1.-g),1.);

            k_abs_ptr[k*n_lam + i] = M_PI*sizes.centre_size(k)*sizes.centre_size(k)*Q_a/sizes.centre_mass(k);
            k_sca_ptr[k*n_lam + i] = M_PI*sizes.centre_size(k)*sizes.centre_size(k)*Q_s/sizes.centre_mass(k);
        }
    }
}

// template<typename CompMix>
__global__ void _calc_rho_kappa_vol(GridRef g, Field3DConstRef<Prims> qd, FieldConstRef<Prims> wg, SizeGridIceRef sizes, double por,
                                DSHARPwCOCompRef comps, int n_comp, double* lam, int n_lam, Field3DRef<double> rhokabs, Field3DRef<double> rhoksca, MoleculeRef mol) {

    int k = threadIdx.x + blockIdx.x*blockDim.x ;
    int j = threadIdx.y + blockIdx.y*blockDim.y ;
    int i = threadIdx.z + blockIdx.z*blockDim.z ;

    if (k < n_lam && j < g.Nphi + 2*g.Nghost && i < g.NR+2*g.Nghost) {
        // if (i==10 && j==10 && k==10) {printf("enter kern\n");}

        double rhok_dust_abs = 0;
        double rhok_dust_sca = 0;
        double k_abs_g=0;
        double k_sca_g=0;
        
        for (int n = 0; n < qd.Nd; n++) {

            double eps1, eps2, n1, n2;

            double vol_mf = mol.ice(i, j, n) / (qd(i, j, n).rho + mol.ice(i, j, n));

            double rho_1 = 0;
            for (int l = 0; l < n_comp - 1; l++) {
                rho_1 += (comps[l].mf * (1. - vol_mf)) / comps[l].dens;
            }
            rho_1 += vol_mf / comps[n_comp - 1].dens;
            double rho_av = 1. / rho_1;

            double vfsig = 0, vf2sig2gam2 = 0, vfgam = 0;
            double n_temp, k_temp, vf_l, vf_m, den, nr2, ni2, sig_l, gam_l, sig_m, gam_m;

            // Aggregate values without storing them in arrays
            // if (i==10 && j==10 && k==10 && n==0) {printf("start loop\n");}
            for (int l = 0; l < n_comp; l++) {
                n_temp = comps[l].opt[k].n;
                k_temp = comps[l].opt[k].k;
                vf_l = (l < n_comp - 1) ? 
                  (1. - por) * rho_av * (comps[l].mf * (1. - vol_mf)) / comps[l].dens :
                  (1. - por) * vol_mf / comps[n_comp - 1].dens;

                nr2 = n_temp * n_temp;
                ni2 = k_temp * k_temp;
                den = ((nr2 - ni2 + 2.) * (nr2 - ni2 + 2.) + 4. * nr2 * ni2);
                sig_l = ((nr2 - ni2 - 1.) * (nr2 - ni2 + 2.) + 4. * nr2 * ni2) / den;
                gam_l = 2. * n_temp * k_temp / den;

                vfsig += vf_l * sig_l;
                vfgam += vf_l * gam_l;

                // Inner loop to calculate vf2sig2gam2
                for (int m = 0; m < n_comp; m++) {
                    n_temp = comps[m].opt[k].n;
                    k_temp = comps[m].opt[k].k;
                    vf_m = (m < n_comp - 1) ? 
                      (1. - por) * rho_av * (comps[m].mf * (1. - vol_mf)) / comps[m].dens :
                      (1. - por) * vol_mf / comps[n_comp - 1].dens;

                    nr2 = n_temp * n_temp;
                    ni2 = k_temp * k_temp;
                    den = ((nr2 - ni2 + 2.) * (nr2 - ni2 + 2.) + 4. * nr2 * ni2);
                    sig_m = ((nr2 - ni2 - 1.) * (nr2 - ni2 + 2.) + 4. * nr2 * ni2) / den;
                    gam_m = 2. * n_temp * k_temp / den;

                    vf2sig2gam2 += vf_l * vf_m * (sig_l * sig_m + 9. * gam_l * gam_m);
                }
            }

            // if (i==10 && j==10 && k==10 && n==0) {printf("after loop\n");}

            double x = 1.-2.*vfsig+vf2sig2gam2;
            eps1 = (1.+vfsig-2.*vf2sig2gam2)/x;
            eps2 = 9.*vfgam/x;

            x = sqrt(eps1*eps1+eps2*eps2)/2.;
            n1 = sqrt(x+eps1/2.);
            n2 = sqrt(x-eps1/2.);

            x = 2.*M_PI*sizes.ice(i,j,n).a/(lam[k]/1.e4);

            double Q_a = min(1.,12.*x*eps2/((eps1+2.)*(eps1+2.)+eps2*eps2));
            double Q_s, g_asym;
            
            if (x < 1.3) {
                Q_s = 8./3.*pow(x,4.)*((eps1-1.)*(eps1-1.)+eps2*eps2)/((eps1+2.)*(eps1+2.)+eps2*eps2);
            }
            else {
                Q_s = 2.*x*x*(n1-1.)*(n1-1.)*(1.+(n2/(n1-1.))*(n2/(n1-1.)));
            }

            if (n2 < 1.) {
                if (x < 3.) { g_asym = 0.7*(x/3.)*(x/3.); }
                else { g_asym = 0.7; }
            }
            else {
                if (x < 3.) { g_asym = -0.2; }
                else { g_asym = 0.5; } 
            }

            Q_s = min(Q_s*(1.-g_asym),1.);

            // x = M_PI*sizes.ice(i,j,n).a*sizes.ice(i,j,n).a/(4.188790205*);
            x = 0.75/(sizes.ice(i,j,n).a*sizes.ice(i,j,n).rho);

            if (n==0) {
                k_abs_g = x*Q_a * 1e-10;
                k_sca_g = x*Q_s * 1e-10;
            }
            if (i==50 && j==2 && k==50 && n==0) {
                printf("%g, ", x*Q_a);
            }

            rhok_dust_abs += (qd(i,j,n).rho + mol.ice(i,j,n))*x*Q_a;
            rhok_dust_sca += (qd(i,j,n).rho + mol.ice(i,j,n))*x*Q_s;
        }

        rhokabs(i,j,k) = wg(i,j).rho*k_abs_g + rhok_dust_abs;
        rhoksca(i,j,k) = wg(i,j).rho*k_sca_g + rhok_dust_sca;
    }
} 
__global__ void _calc_rho_kappa_vol(GridRef g, GridRef g_in, Field3DConstRef<double> rho_d, FieldConstRef<Prims> wg, SizeGridIceRef sizes, double por, double* k_abs, double* k_sca,
                                DSHARPwCOCompRef comps, int n_comp, double* lam, int n_lam, Field3DRef<double> rhokabs, Field3DRef<double> rhoksca, MoleculeRef mol) {

    int k = threadIdx.x + blockIdx.x*blockDim.x ;
    int j = threadIdx.y + blockIdx.y*blockDim.y ;
    int i = threadIdx.z + blockIdx.z*blockDim.z ;

    if (k < n_lam && j < g.Nphi + 2*g.Nghost && i < g.NR+2*g.Nghost) {
        // if (i==10 && j==10 && k==10) {printf("enter kern\n");}

        double rhok_dust_abs = 0;
        double rhok_dust_sca = 0;
        double k_abs_g=0;
        double k_sca_g=0;
        
        for (int n = 0; n < rho_d.Nd; n++) {

            if (i > g_in.NR+g_in.Nghost-1) {

                double eps1, eps2, n1, n2;
    
                double vol_mf = mol.ice(i-g_in.NR, j, n) / rho_d(i, j, n);
                
                double rho_1 = 0;
                for (int l = 0; l < n_comp - 1; l++) {
                    rho_1 += (comps[l].mf * (1. - vol_mf)) / comps[l].dens;
                }
                rho_1 += vol_mf / comps[n_comp - 1].dens;
                double rho_av = 1. / rho_1;

                double vfsig = 0, vf2sig2gam2 = 0, vfgam = 0;
                double n_temp, k_temp, vf_l, vf_m, den, nr2, ni2, sig_l, gam_l, sig_m, gam_m;

                // Aggregate values without storing them in arrays
                // if (i==10 && j==10 && k==10 && n==0) {printf("start loop\n");}
                for (int l = 0; l < n_comp; l++) {
                    n_temp = comps[l].opt[k].n;
                    k_temp = comps[l].opt[k].k;
                    vf_l = (l < n_comp - 1) ? 
                    (1. - por) * rho_av * (comps[l].mf * (1. - vol_mf)) / comps[l].dens :
                    (1. - por) * vol_mf / comps[n_comp - 1].dens;

                    nr2 = n_temp * n_temp;
                    ni2 = k_temp * k_temp;
                    den = ((nr2 - ni2 + 2.) * (nr2 - ni2 + 2.) + 4. * nr2 * ni2);
                    sig_l = ((nr2 - ni2 - 1.) * (nr2 - ni2 + 2.) + 4. * nr2 * ni2) / den;
                    gam_l = 2. * n_temp * k_temp / den;

                    vfsig += vf_l * sig_l;
                    vfgam += vf_l * gam_l;

                    // Inner loop to calculate vf2sig2gam2
                    for (int m = 0; m < n_comp; m++) {
                        n_temp = comps[m].opt[k].n;
                        k_temp = comps[m].opt[k].k;
                        vf_m = (m < n_comp - 1) ? 
                        (1. - por) * rho_av * (comps[m].mf * (1. - vol_mf)) / comps[m].dens :
                        (1. - por) * vol_mf / comps[n_comp - 1].dens;

                        nr2 = n_temp * n_temp;
                        ni2 = k_temp * k_temp;
                        den = ((nr2 - ni2 + 2.) * (nr2 - ni2 + 2.) + 4. * nr2 * ni2);
                        sig_m = ((nr2 - ni2 - 1.) * (nr2 - ni2 + 2.) + 4. * nr2 * ni2) / den;
                        gam_m = 2. * n_temp * k_temp / den;

                        vf2sig2gam2 += vf_l * vf_m * (sig_l * sig_m + 9. * gam_l * gam_m);
                    }
                }

                // if (i==10 && j==10 && k==10 && n==0) {printf("after loop\n");}

                double x = 1.-2.*vfsig+vf2sig2gam2;
                eps1 = (1.+vfsig-2.*vf2sig2gam2)/x;
                eps2 = 9.*vfgam/x;

                x = sqrt(eps1*eps1+eps2*eps2)/2.;
                n1 = sqrt(x+eps1/2.);
                n2 = sqrt(x-eps1/2.);

                x = 2.*M_PI*sizes.ice(i-g_in.NR,j,n).a/(lam[k]/1.e4);

                double Q_a = min(1.,12.*x*eps2/((eps1+2.)*(eps1+2.)+eps2*eps2));
                double Q_s, g_asym;
                
                if (x < 1.3) {
                    Q_s = 8./3.*pow(x,4.)*((eps1-1.)*(eps1-1.)+eps2*eps2)/((eps1+2.)*(eps1+2.)+eps2*eps2);
                }
                else {
                    Q_s = 2.*x*x*(n1-1.)*(n1-1.)*(1.+(n2/(n1-1.))*(n2/(n1-1.)));
                }

                if (n2 < 1.) {
                    if (x < 3.) { g_asym = 0.7*(x/3.)*(x/3.); }
                    else { g_asym = 0.7; }
                }
                else {
                    if (x < 3.) { g_asym = -0.2; }
                    else { g_asym = 0.5; } 
                }

                Q_s = min(Q_s*(1.-g_asym),1.);

                // x = M_PI*sizes.ice(i,j,n).a*sizes.ice(i,j,n).a/(4.188790205*);
                x = 0.75/(sizes.ice(i-g_in.NR,j,n).a*sizes.ice(i-g_in.NR,j,n).rho);

                if (n==0) {
                    k_abs_g = x*Q_a * 1e-12;
                    k_sca_g = x*Q_s * 1e-12;
                }
                // if (i==50 && j==2 && k==50 && n==0) {
                //     printf("%g, ", x*Q_a);
                // }

                rhok_dust_abs += rho_d(i,j,n)*x*Q_a;
                // if (i==601 && j==82 && k==0) {printf("%g\n",sizes.ice(i-g_in.NR,j,n).a);}

                rhok_dust_sca += rho_d(i,j,n)*x*Q_s;
            }

            else {

                if (n==0) {
                    k_abs_g = k_abs[n*n_lam + k] * 1e-12;
                    k_sca_g = k_sca[n*n_lam + k] * 1e-12;
                }

                rhok_dust_abs += rho_d(i,j,n)*k_abs[n*n_lam + k];
                rhok_dust_sca += rho_d(i,j,n)*k_sca[n*n_lam + k];

            }
        }

        rhokabs(i,j,k) = wg(i,j).rho*k_abs_g + rhok_dust_abs;
        rhoksca(i,j,k) = wg(i,j).rho*k_sca_g + rhok_dust_sca;
    }
} 

__global__ void _calc_rho_tot_vol(GridRef g, Field3DConstRef<Prims> wd, FieldConstRef<Prims> wg, MoleculeRef mol, FieldRef<double> rho_tot) {

    int j = threadIdx.y + blockIdx.y*blockDim.y ;
    int i = threadIdx.z + blockIdx.z*blockDim.z ;

    if (j < g.Nphi + 2*g.Nghost && i < g.NR+2*g.Nghost) {

        double rho_tot_temp = 0.;
        for (int k=0; k<wd.Nd; k++) {
            rho_tot_temp += wd(i,j,k).rho + mol.ice(i,j,k);
        }    
        rho_tot(i,j) = wg(i,j).rho + rho_tot_temp + mol.vap(i,j);

    }
} 

// template<typename T>
void calculate_total_rhokappa(Grid& g, SizeGridIce& sizes, Field3D<Prims>& qd, Field<Prims>& wg, Field<double>& rho_tot, CuzziOpacs<DSHARPwCOComp>& opacs,
                                    Field3D<double>& rhokappa_abs, Field3D<double>& rhokappa_sca, Molecule& mol) {

    int nk = 1 ;
    while (nk < opacs.n_lam && nk < 32)
        nk *= 2 ;
    int nj = 512 / nk ;

    dim3 threads(nk, nj, 1) ;
    dim3 blocks((opacs.n_lam +  nk-1)/nk, 
                (g.Nphi +  2*g.Nghost + nj-1)/nj, 
                 g.NR +  2*g.Nghost) ;

    dim3 threads2D(1,32,32);
    dim3 blocks2D(1,(g.Nphi+ 2*g.Nghost +31)/32, (g.NR+2*g.Nghost+31)/32);

    // CudaArray<Comp> comps = make_CudaArray<Comp>(opacs.comp.n_comp);
    // for (int i=0; i<opacs.comp.n_comp; i++) {
    //     comps[i].dens = opacs.comp[i].dens;
    //     comps[i].mf = opacs.comp[i].mf;
    //     comps[i].opt = opacs.comp[i].opt;
    // }

    _calc_rho_kappa_vol<<<blocks,threads>>>(g, qd, wg, sizes, opacs.por, opacs.comp, opacs.comp.n_comp, opacs.lam(), opacs.n_lam, rhokappa_abs, rhokappa_sca, mol);
    check_CUDA_errors("_calc_rho_kappa_vol") ;

    _calc_rho_tot_vol<<<blocks2D,threads2D>>>(g, qd, wg, mol, rho_tot);
    check_CUDA_errors("_calc_rho_tot_vol") ;
}

void calculate_total_rhokappa(Grid& g, Grid& g_in, SizeGridIce& sizes, Field3D<double>& rho_d, Field<Prims>& wg, Field<double>& rho_tot, CuzziOpacs<DSHARPwCOComp>& opacs,
                                    Field3D<double>& rhokappa_abs, Field3D<double>& rhokappa_sca, Molecule& mol) {

    int nk = 1 ;
    while (nk < opacs.n_lam && nk < 32)
        nk *= 2 ;
    int nj = 512 / nk ;

    dim3 threads(nk, nj, 1) ;
    dim3 blocks((opacs.n_lam +  nk-1)/nk, 
                (g.Nphi +  2*g.Nghost + nj-1)/nj, 
                 g.NR +  2*g.Nghost) ;

    _calc_rho_kappa_vol<<<blocks,threads>>>(g, g_in, rho_d, wg, sizes, opacs.por, opacs.k_abs(), opacs.k_sca(), opacs.comp, opacs.comp.n_comp, opacs.lam(), opacs.n_lam, rhokappa_abs, rhokappa_sca, mol);
    check_CUDA_errors("_calc_rho_kappa_vol") ;

}










// template<typename CompMixRef>
// __global__
// void _calc_opacs(GridRef g, CompMixRef comp, int n_lam, double* lam, Field3DRef<double> vfs, int n_spec, 
//                     double* a_c, double* m_c, double* k_abs_ptr, double* k_sca_ptr) {
 
//     int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
//     int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
//     int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
//     int istride = gridDim.x * blockDim.x ;
//     int jstride = gridDim.y * blockDim.y ;
//     int kstride = gridDim.z * blockDim.z ;

//     for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
//         for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) {   
//             for (int k=kidx; k<n_lam; k+=kstride) {

//                 CudaArray<double> sigs = make_CudaArray<double>(comp.n_comp);
//                 CudaArray<double> gams = make_CudaArray<double>(comp.n_comp);
//                 RefIndx eps, n;
//                 double vfsig=0, vf2sig2gam2=0, vfgam=0;

//                 for (int l=0; l<comp.n_comp; l++) {
//                     double nr2 = comp[l].opt[k].n*comp[l].opt[k].n;
//                     double ni2 = comp[l].opt[k].k*comp[l].opt[k].k;
//                     sigs[l] = ((nr2-ni2-1.)*(nr2-ni2+2.)+4.*nr2*ni2)/((nr2-ni2+2.)*(nr2-ni2+2.)+4.*nr2*ni2);
//                     gams[l] = 2.*comp[l].opt[k].n*comp[l].opt[k].k/((nr2-ni2+2.)*(nr2-ni2+2.)+4.*nr2*ni2);

//                     vfsig += vfs(i,j,l)*sigs[l];
//                     vfgam += vfs(i,j,l)*gams[l];
//                 }

//                 for (int l=0; l<comp.n_comp; l++) {
//                     for (int m=0; m<comp.n_comp; m++) {
//                         vf2sig2gam2 += vfs(i,j,l)*vfs(i,j,m)*(sigs[l]*sigs[m]+9.*gams[l]*gams[m]);
//                     }
//                 }
//                 double D = 1.-2.*vfsig+vf2sig2gam2;
//                 eps.n = (1.+vfsig-2.*vf2sig2gam2)/D;
//                 eps.k = 9.*vfgam/D;

//                 double eps_quad = std::sqrt(eps.n*eps.n+eps.k*eps.k)/2.;
//                 n.n = std::sqrt(eps_quad+eps.n/2.);
//                 n.k = std::sqrt(eps_quad-eps.n/2.);

//                 for (int n=0; n<n_spec; n++) {
//                     double x = 2.*M_PI*a_c[n]/(lam[k]/1.e4);

//                     double Q_a = std::min(1.,12.*x*eps.k/((eps.n+2.)*(eps.n+2.)+eps.k*eps.k));
//                     double Q_s, g_asym;
                    
//                     if (x < 1.3) {
//                         Q_s = 8./3.*std::pow(x,4.)*((eps.n-1.)*(eps.n-1.)+eps.k*eps.k)/((eps.n+2.)*(eps.n+2.)+eps.k*eps.k);
//                     }
//                     else {
//                         Q_s = 2.*x*x*(n.n-1.)*(n.n-1.)*(1.+(n.k/(n.n-1.))*(n.k/(n.n-1.)));
//                     }

//                     if (n.k < 1.) {
//                         if (x < 3.) { g_asym = 0.7*(x/3.)*(x/3.); }
//                         else { g_asym = 0.7; }
//                     }
//                     else {
//                         if (x < 3.) { g_asym = -0.2; }
//                         else { g_asym = 0.5; } 
//                     }

//                     Q_s = std::min(Q_s*(1.-g_asym),1.);

//                     k_abs_ptr[n*n_lam + k] = M_PI*a_c[n]*a_c[n]*Q_a/m_c[n];
//                     k_sca_ptr[n*n_lam + k] = M_PI*a_c[n]*a_c[n]*Q_s/m_c[n];
//                 }

//             }
//         }
//     }
// }

// template<class CompMix>
// void CuzziOpacs<CompMix>::calc_opacs_vol(Grid& g, Molecule& mol, SizeGrid& sizes, double por) {

//     dim3 threads(16,8,4) ;
//     dim3 blocks((g.NR + 2*g.Nghost+15)/16,(g.Nphi + 2*g.Nghost+7)/8, (q.Nd+3)/4) ;

//     CudaArray<double> vfs = make_CudaArray<double>(comp.n_comp);

//     // Calc optical constants via Garnett EMT

//     double rho_1 = 0;
//     for (int i=0; i<comp.n_comp; i++) {rho_1 += comp[i].mf/comp[i].dens;}
//     double rho_av = 1./rho_1;
//     std::cout << "Bulk density: " << rho_av << " gcm-3";

//     for (int i=0; i<comp.n_comp; i++) {
//         vfs[i] = (1.-por)*rho_av*comp[i].mf/comp[i].dens;
//     }

//     for (int i=0; i<n_lam; i++) {
//         CudaArray<double> sigs = make_CudaArray<double>(comp.n_comp);
//         CudaArray<double> gams = make_CudaArray<double>(comp.n_comp);
//         RefIndx eps, n;
//         double vfsig=0, vf2sig2gam2=0, vfgam=0;

//         for (int j=0; j<comp.n_comp; j++) {
//             double nr2 = comp[j].opt[i].n*comp[j].opt[i].n;
//             double ni2 = comp[j].opt[i].k*comp[j].opt[i].k;
//             sigs[j] = ((nr2-ni2-1.)*(nr2-ni2+2.)+4.*nr2*ni2)/((nr2-ni2+2.)*(nr2-ni2+2.)+4.*nr2*ni2);
//             gams[j] = 2.*comp[j].opt[i].n*comp[j].opt[i].k/((nr2-ni2+2.)*(nr2-ni2+2.)+4.*nr2*ni2);

//             vfsig += vfs[j]*sigs[j];
//             vfgam += vfs[j]*gams[j];
//         }

//         for (int j=0; j<comp.n_comp; j++) {
//             for (int k=0; k<comp.n_comp; k++) {
//                 vf2sig2gam2 += vfs[j]*vfs[k]*(sigs[j]*sigs[k]+9.*gams[j]*gams[k]);
//             }
//         }
//         double D = 1.-2.*vfsig+vf2sig2gam2;
//         eps.n = (1.+vfsig-2.*vf2sig2gam2)/D;
//         eps.k = 9.*vfgam/D;

//         double eps_quad = std::sqrt(eps.n*eps.n+eps.k*eps.k)/2.;
//         n.n = std::sqrt(eps_quad+eps.n/2.);
//         n.k = std::sqrt(eps_quad-eps.n/2.);

//         // std::cout << n.n << ", ";

//         for (int k=0; k<sizes.size(); k++) {
//             double x = 2.*M_PI*sizes.centre_size(k)/(lam(i)/1.e4);

//             double Q_a = std::min(1.,12.*x*eps.k/((eps.n+2.)*(eps.n+2.)+eps.k*eps.k));
//             double Q_s, g;
            
//             if (x < 1.3) {
//                 Q_s = 8./3.*std::pow(x,4.)*((eps.n-1.)*(eps.n-1.)+eps.k*eps.k)/((eps.n+2.)*(eps.n+2.)+eps.k*eps.k);
//             }
//             else {
//                 Q_s = 2.*x*x*(n.n-1.)*(n.n-1.)*(1.+(n.k/(n.n-1.))*(n.k/(n.n-1.)));
//             }

//             if (n.k < 1.) {
//                 if (x < 3.) { g = 0.7*(x/3.)*(x/3.); }
//                 else { g = 0.7; }
//             }
//             else {
//                 if (x < 3.) { g = -0.2; }
//                 else { g = 0.5; } 
//             }

//             Q_s = std::min(Q_s*(1.-g),1.);

//             k_abs_ptr[k*n_lam + i] = M_PI*sizes.centre_size(k)*sizes.centre_size(k)*Q_a/sizes.centre_mass(k);
//             k_sca_ptr[k*n_lam + i] = M_PI*sizes.centre_size(k)*sizes.centre_size(k)*Q_s/sizes.centre_mass(k);
//         }
//     }
// }

template<class CompMix>
void CuzziOpacs<CompMix>::write_interp(std::filesystem::path folder) const {

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

template class CuzziOpacs<CuzziComp>;
template class CuzziOpacs<DSHARPComp>;
template class CuzziOpacs<DSHARPwCOComp>;

// template void calculate_total_rhokappa<CuzziComp>(Grid& g, SizeGrid& sizes, Field3D<Prims>& qd, Field<Prims>& wg, Field<double>& rho_tot, CuzziOpacs<CuzziComp>& opacs,
//                                     Field3D<double>& rhokappa_abs, Field3D<double>& rhokappa_sca, Molecule& mol) ;
// template void calculate_total_rhokappa<DSHARPComp>(Grid& g, SizeGrid& sizes, Field3D<Prims>& qd, Field<Prims>& wg, Field<double>& rho_tot, CuzziOpacs<DSHARPComp>& opacs,
//                                     Field3D<double>& rhokappa_abs, Field3D<double>& rhokappa_sca, Molecule& mol) ;
// template void calculate_total_rhokappa(Grid& g, SizeGrid& sizes, Field3D<Prims>& qd, Field<Prims>& wg, Field<double>& rho_tot, CuzziOpacs<DSHARPwCOComp>& opacs,
//                                     Field3D<double>& rhokappa_abs, Field3D<double>& rhokappa_sca, Molecule& mol) ;

// template __global__ void _calc_rho_kappa_vol(GridRef g, Field3DConstRef<Prims> qd, FieldConstRef<Prims> wg, RealType* a_c, RealType* m_c,
//                                 CuzziOpacsRef<T> opacs, Field3DRef<double> rhokabs, Field3DRef<double> rhoksca, MoleculeRef mol)