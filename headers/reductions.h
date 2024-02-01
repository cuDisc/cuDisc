#ifndef _CUDISC_REDUCTIONS_H_
#define _CUDISC_REDUCTIONS_H_

#include "cuda_array.h"
#include "field.h"
#include "grid.h"

namespace Reduction {
    
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