
#include "constants.h"
#include "planck.h"
#include "star.h"

Star::Star(double GM_, double R, int num_wle_, 
           const CudaArray<double>& wle_, const CudaArray<double>& Lband_) 
  : GM(GM_)
{
    // Setup the wavelength grid
    set_wavelengths(num_wle_, wle_) ;

    // Copy the fluxes:
    Lband = make_CudaArray<double>(num_wle) ;
    L = 0 ;
    for(int i = 0; i < num_wle; i++)
       L += Lband[i] = Lband_[i] ;

    // Compute the effective temperature:
    Teff = std::pow(L/(4*M_PI*R*R*sigma_SB),0.25) ;
}

void Star::set_blackbody_fluxes() {
    
    PlanckInegral planck ;
    Lband = make_CudaArray<double>(num_wle) ;
                                    
    for (int i=0; i < num_wle; i++) {
        double f0, f1 ;
        double x ;
        if (i == 0)
            f1 = 1 ;
        else {
            x = planck.WienParameter(std::sqrt(wle[i-1] * wle[i]), 
                                     Teff) ;
            f1 = planck(x) ;
        }
        
        if (i == num_wle-1)
            f0 = 0 ;
        else {
            x = planck.WienParameter(std::sqrt(wle[i+1] * wle[i]), 
                                     Teff) ;
            f0 = planck(x) ;
        }

        Lband[i] = L * (f1 - f0) ;
    }
}
