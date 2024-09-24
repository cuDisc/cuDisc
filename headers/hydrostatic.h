
#ifndef _CUSDISC_HYDROSTATIC_H_
#define _CUSDISC_HYDROSTATIC_H_

#include "field.h"
#include "grid.h"
#include "star.h"
#include "dustdynamics.h"
#include "icevapour.h"

void compute_hydrostatic_equilibrium(const Star&, const Grid&, Field<double>&, 
                                     const Field<double>&, const CudaArray<double>&) ;
void compute_hydrostatic_equilibrium(const Star&, const Grid&, Field<Prims>&, 
                                     const Field<double>&, const CudaArray<double>&, double gasfloor=1e-100) ;
void compute_hydrostatic_equilibrium(const Star&, const Grid&, Field<Prims>&, 
                                     const Field<double>&, const CudaArray<double>&, Field3D<Prims>& q_d, double gasfloor=1e-100) ;
void compute_hydrostatic_equilibrium(const Star&, const Grid&, Field<Prims>&, 
                                     const Field<double>&, const CudaArray<double>&, Molecule& mol, double gasfloor=1e-100, double floor=1e-10) ;

#endif// _CUSDISC_HYDROSTATIC_H_