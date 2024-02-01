
#ifndef _CUDISC_SUPER_STEPPING_H_
#define _CUDISC_SUPER_STEPPING_H_

#include <cmath>
#include <iostream>

#include "field.h"
#include "utils.h"


/* Runge-Kutta-Legendre super-time-stepping method of order 2 from Meyer et 
 * al (2012, doi:10.1111/j.1365-2966.2012.20744.x).
*/
class SuperStepping {
  public:
    SuperStepping(const Grid& g, int num_steps)
      : _grid(g), _num_steps(num_steps)
    { } ;

    /* Update the system using the super-time-stepping method. SourceTerms
     * will be applied in an operator split fashon after every sub-step.
     */
    template<class Method, class SourceTerms, class FieldType>
    void operator()(const Method& method, const SourceTerms& source, 
                    FieldType& u0, double dt) const {
        

        // Storage for temporary field values
        FieldType u[2] = {
            duplicate_field(u0),
            duplicate_field(u0)
        } ;
        copy_field(_grid, u0, u[0]) ;
        set_all(_grid, u[1], 0) ;
        
        // Storage for diffusion rates
        FieldType l  = duplicate_field(u0) ;
        FieldType l0 = duplicate_field(u0) ;


        //=== Compute the first step, which is special.
        double w1 = 4./(_num_steps*(_num_steps + 1) - 2) ;

        // Note update_solution evaluates:
        //  u[1] = mu * u[0] + nu*u[1] + (1-mu-nu) * u0 
        //          + dt*(mup * method(u[0]) + gam*l0)
        //       = u0 + bj(1)*w1*l0*dt  for the first step.
        double 
            mu = 1, nu= 0,
            mup=bj(1)*w1, gam=0 ;

        method(u0, l0) ;
        update_solution(Field3DConstRef<double>(u0),
                Field3DConstRef<double>(u[0]), Field3DRef<double>(u[1]),
                Field3DConstRef<double>(l0), Field3DConstRef<double>(l), 
                mu, nu, mup, gam, dt) ;
        // Apply the source terms
        double dt_j = bj(1)*w1*dt ;
        source(u[1], dt_j) ;

        //=== Iterate using the RKL recursion relations
        for (int j=2; j<=_num_steps; j++) {

            // Get the weight factors for the substep
            double b = bj(j) ;
            double bm1 = bj(j-1) ;
            double bm2 = bj(j-2) ;

            mu = (2-1./j) * b/bm1 ;
            nu = - (1-1./j) * b/bm2 ;

            mup = mu * w1 ;
            gam = (bm1-1) * mup ;

            // Compute the new rate, update the solution, apply source terms.
            method(u[(j-1)%2], l) ;
            update_solution(Field3DConstRef<double>(u0),
                    Field3DConstRef<double>(u[(j-1)%2]), Field3DRef<double>(u[j%2]),
                    Field3DConstRef<double>(l0), Field3DConstRef<double>(l), 
                    mu, nu, mup, gam, dt) ;
            
            if (j == 2) 
                dt_j = (2*w1/3) * dt ;
            else 
                dt_j = 0.5*j*w1 * dt ;
            
            source(u[j%2], dt_j) ;
        }

        // Return the final value.
        copy_field(_grid, u[_num_steps%2], u0) ;
    }

    /* Update the system using the super-time-stepping method without
     * source terms.
     */
    template<class Method, class FieldType>
    void operator()(const Method& method, 
                    FieldType& u0, double dt) const {
                        
        auto source = [](const FieldType&, double) { } ;

        (*this)(method, source, u0, dt) ;
    }


    /* Compute the number of steps needed in the super time stepping method
     * for a desired time-step and the stable explicit time-step.
    */
    static int num_steps_required(double dt, double dt_explicit) {
        double x = 4 * dt/dt_explicit ;

        double s = 2 * (2+x) / (1 + sqrt(9 + 4*x)) ;
        int steps = std::max(int(s+1), 2) ;
        
        // Make sure steps is odd. Even numbers are not completely stable.
        return 2*(steps/2) + 1 ;
    } 

  private:
    double bj(int j) const {
        
        if (j < 3)
            return 1/3. ;
        
        double x = j*(j+1) ;
        return 0.5*(1. - 2./x) ;
    }

    template<typename dtype>
    Field<dtype> duplicate_field(const Field<dtype>&) const {
        return create_field<dtype>(_grid) ;
    }

    template<typename dtype>
    Field3D<dtype> duplicate_field(const Field3D<dtype>& f) const {
        return create_field3D<dtype>(_grid, f.Nd) ;
    }

    void update_solution(Field3DConstRef<double> u0, 
                         Field3DConstRef<double> u_current,
                         Field3DRef<double> u_old_new,
                         Field3DConstRef<double> l0, 
                         Field3DConstRef<double> l,
                         double mu, double nu, double mup, double gam,
                         double dt) const ;


    const Grid& _grid ;
    int _num_steps ;
} ;

#endif//_CUDISC_SUPER_STEPPING_H_