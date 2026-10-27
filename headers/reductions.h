#ifndef _CUDISC_REDUCTIONS_H_
#define _CUDISC_REDUCTIONS_H_

#include "cuda_array.h"
#include "field.h"
#include "grid.h"

namespace Reduction {

    /* Reductions over the first n values of x, using cuBLAS. The order of the
     * sum is fixed, so the results are reproducible.
     *   sum_abs : sum of |x_i|
     *   norm2   : sqrt(sum of x_i^2)
     */
    double sum_abs(const CudaArray<double>& x, int n) ;
    double norm2(const CudaArray<double>& x, int n) ;

    /* Sum over the species (the third index) of a 3D field, for every cell:
     *   out(i,j) = sum_k in(i,j,k)
     * The order of the sum is fixed, so the results are reproducible.
     */
    void sum_species(const Grid& g, const Field3D<double>& in, Field<double>& out) ;
    
    void volume_integrate_Z(const Grid& g, const Field<double>& in, CudaArray<double>& result) ;
    void volume_integrate_Z_cpu(const Grid& g, const Field<double>& in, CudaArray<double>& result) ;

    // Cumulative sums / products
    void scan_R_sum(const Grid& g, Field<double>&);
    void scan_R_mul(const Grid& g, Field<double>&);
    void scan_R_min(const Grid& g, Field<double>&);
    void scan_R_max(const Grid& g, Field<double>&);
    void scan_Z_sum(const Grid& g, Field<double>&);
    void scan_Z_mul(const Grid& g, Field<double>&);
    void scan_Z_min(const Grid& g, Field<double>&);
    void scan_Z_max(const Grid& g, Field<double>&);

    void scan_R_sum_cpu(const Grid& g, Field<double>&);
    void scan_R_mul_cpu(const Grid& g, Field<double>&);
    void scan_R_min_cpu(const Grid& g, Field<double>&);
    void scan_R_max_cpu(const Grid& g, Field<double>&);
    void scan_Z_sum_cpu(const Grid& g, Field<double>&);
    void scan_Z_mul_cpu(const Grid& g, Field<double>&);
    void scan_Z_min_cpu(const Grid& g, Field<double>&);
    void scan_Z_max_cpu(const Grid& g, Field<double>&);

    // Cumulative sums / products for 3D arrays
    void scan_R_sum(const Grid& g, Field3D<double>&);
    void scan_R_mul(const Grid& g, Field3D<double>&);
    void scan_Z_sum(const Grid& g, Field3D<double>&);
    void scan_Z_mul(const Grid& g, Field3D<double>&);

    void scan_R_sum_cpu(const Grid& g, Field3D<double>&);
    void scan_R_mul_cpu(const Grid& g, Field3D<double>&);
    void scan_Z_sum_cpu(const Grid& g, Field3D<double>&);
    void scan_Z_mul_cpu(const Grid& g, Field3D<double>&);


} // namespace Reduction
#endif//_CUDISC_REDUCTIONS_H_