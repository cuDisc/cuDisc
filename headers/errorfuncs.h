#ifndef _CUDISC_HEADERS_ERRORFUNCS_H_
#define _CUDISC_HEADERS_ERRORFUNCS_H_

#include "grid.h"
#include "field.h"

/* Differences between two fields, e.g. temperatures before and after an
 * update, over all cells (including ghosts):
 *
 *   L2err   : sqrt(sum (T2 - T1)^2 / N)
 *   fracerr : sum |T2 - T1| / T1 / N
 *
 * The default versions run on the GPU, so the fields are not migrated to the
 * host. The sum is done in a fixed order, so the result is reproducible.
 * The _cpu versions loop over the fields on the host.
 */
double L2err(Grid& g, Field<double> &T1, Field<double> &T2) ;
double fracerr(Grid& g, Field<double> &T1, Field<double> &T2) ;

double L2err_cpu(Grid& g, Field<double> &T1, Field<double> &T2) ;
double fracerr_cpu(Grid& g, Field<double> &T1, Field<double> &T2) ;

#endif//_CUDISC_HEADERS_ERRORFUNCS_H_
