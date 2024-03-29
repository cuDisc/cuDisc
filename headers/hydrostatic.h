
#ifndef _CUSDISC_HYDROSTATIC_H_
#define _CUSDISC_HYDROSTATIC_H_

#include "field.h"
#include "grid.h"
#include "star.h"
#include "dustdynamics.h"

void compute_hydrostatic_equilibrium(const Star&, const Grid&, Field<double>&, 
                                     const Field<double>&, const CudaArray<double>&) ;
void compute_hydrostatic_equilibrium(const Star&, const Grid&, Field<Prims>&, 
                                     const Field<double>&, const CudaArray<double>&, double gasfloor=1e-100) ;
void compute_hydrostatic_equilibrium(const Star&, const Grid&, Field<Prims>&, 
                                     const Field<double>&, const CudaArray<double>&, Field3D<Prims>& q_d, double gasfloor=1e-100) ;

#endif// _CUSDISC_HYDROSTATIC_H_