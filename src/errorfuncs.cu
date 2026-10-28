#include <cmath>

#include "errorfuncs.h"
#include "reductions.h"
#include "utils.h"

namespace {

// Per-cell terms: (T2-T1)/T1 (relative) or T2-T1, over all cells
template<bool relative>
__global__ void _err_terms(GridRef g, FieldConstRef<double> T1, FieldConstRef<double> T2, FieldRef<double> terms) {

    int jidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int iidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int jstride = gridDim.x * blockDim.x ;
    int istride = gridDim.y * blockDim.y ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride)
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) {
            double d = T2(i,j) - T1(i,j) ;
            terms(i,j) = relative ? d / T1(i,j) : d ;
        }
}

/* Sum of |terms| (relative) or sqrt of the sum of terms^2.
 * The padding of the terms field is zeroed, so the whole buffer can be summed.
 */
template<bool relative>
double _err_sum(Grid& g, Field<double>& T1, Field<double>& T2) {

    Field<double> terms = create_field<double>(g) ;
    int size = (g.NR + 2*g.Nghost) * terms.stride ;
    (void) cudaMemset(terms.get().get(), 0, size*sizeof(double)) ;

    dim3 threads(32,8) ;
    dim3 blocks((g.Nphi + 2*g.Nghost + 31)/32, (g.NR + 2*g.Nghost + 7)/8) ;
    _err_terms<relative><<<blocks, threads>>>(g, T1, T2, terms) ;
    check_CUDA_errors("_err_terms") ;

    if (relative)
        return Reduction::sum_abs(terms.get(), size) ;
    else
        return Reduction::norm2(terms.get(), size) ;
}

} // namespace

double L2err(Grid& g, Field<double> &T1, Field<double> &T2) {
    int N = (g.NR + 2*g.Nghost)*(g.Nphi + 2*g.Nghost);
    return _err_sum<false>(g, T1, T2) / std::sqrt(N) ;
}

double fracerr(Grid& g, Field<double> &T1, Field<double> &T2) {
    int N = (g.NR + 2*g.Nghost)*(g.Nphi + 2*g.Nghost);
    return _err_sum<true>(g, T1, T2) / N ;
}

double L2err_cpu(Grid& g, Field<double> &T1, Field<double> &T2) {

    double sum=0;
    int N = (g.NR + 2*g.Nghost)*(g.Nphi + 2*g.Nghost);

    for (int i=0; i<g.NR + 2*g.Nghost; i++) {
        for (int j=0; j<g.Nphi + 2*g.Nghost; j++) {
            sum += (T2(i,j) - T1(i,j))*(T2(i,j) - T1(i,j));
        }
    }

    return std::sqrt(sum/N);
}

double fracerr_cpu(Grid& g, Field<double> &T1, Field<double> &T2) {

    double sum=0;
    int N = (g.NR + 2*g.Nghost)*(g.Nphi + 2*g.Nghost);

    for (int i=0; i<g.NR + 2*g.Nghost; i++) {
        for (int j=0; j<g.Nphi + 2*g.Nghost; j++) {
            sum += std::abs(T2(i,j) - T1(i,j))/T1(i,j);
        }
    }

    return sum/N;
}
