#include <sstream>
#include <stdexcept>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

#include "pcg_solver.h"
#include "timing.h"


BlockJacobi_precond::BlockJacobi_precond(const CSR_SpMatrix& mat, int block_size) 
    : A(mat), tmp(mat.rows), _block_size(block_size)
{
    cublasStatus_t status_cub = 
        cublasCreate(&_handle_cublas) ;
    if (status_cub != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("Failed to initialize CUBLAS") ;

    cusparseStatus_t status_cus =
        cusparseCreate(&_handle_cusparse) ;
     if (status_cus != CUSPARSE_STATUS_SUCCESS)
        throw std::runtime_error("Failed to initialize CUSPARSE") ;
}

/* _block_jacobi_solve
 *
 * Approximate the solution of A*y = x using the block jacobi method. I.e. solve
 *    diag(A) y = x 
 * Here diag(A) is the block-diagonal matrix of size block_size.
 */
__global__ void _block_jacobi_solve(CSR_SpMatrixConstRef A,
                                    DnVecConstRef x, DnVecRef y, DnVecRef r, 
                                    int block_size) {

    int block = threadIdx.x + blockIdx.x * blockDim.x ;
    int idx = block * block_size ;

    if (idx < A.rows) {

        //  Method: Solve diag(A) * y = x
        //  Here we rely on the form of the coupled FLD equations:
        //     A00 * E0 + \sum_i A0i * Ei = r0
        //     Ai0 * E0 +        Aii * Ei = ri 
        //  to solve the system directly using 
        //     Ei = (ri - Ai0 * E0) / Aii
       
        // Step 1: Compute:
        //   R  = \sum_i A0i * ri  / Aii
        //   E  = \sum_i A0i * Ai0 / Aii    
        double R = 0, E = 0 ;
        double A00, A0i, Ai0, Aii ;

        int i_start = A.csr_offset[idx] ;
        int i_end = A.csr_offset[idx+1] ;

        for (; i_start < i_end; i_start++)
            if (A.col_index[i_start] == idx) {
                A00 = A.data[i_start++] ;
                break ;
            }

        for (int i=1; i < block_size; i++) {
            for (; i_start < i_end; i_start++)
                if (A.col_index[i_start] == idx+i) {
                    A0i = A.data[i_start++] ;
                    break ;
                }

            int j_start = A.csr_offset[idx+i] ;
            int j_end = A.csr_offset[idx+i+1] ;
            for ( ; j_start < j_end; j_start++) {
                int j = A.col_index[j_start] ;
                if (j == idx)
                    Ai0 = A.data[j_start] ;
                if (j == idx+i)
                    Aii = A.data[j_start] ;
            }

            r.data[idx+i] = x.data[idx+i] / Aii ;
            y.data[idx+i] = Ai0 / Aii ;
            R += A0i * r.data[idx+i] ;
            E += A0i * y.data[idx+i] ;
        }

        // Step 2: Solve for E0, E_i
        E = y.data[idx] = (x.data[idx]-R) / (A00 - E) ;
        for (int i=1; i < block_size; i++)
            y.data[idx+i] = r.data[idx+i] - y.data[idx+i]*E ;
    }
}
void BlockJacobi_precond::solve(const DnVec& x, DnVec& y) {

    CodeTiming::BlockTimer timing_block = 
        timer->StartNewTimer("BlockJacobi_precond::solve") ;

    DnVec tmp(x.rows) ;

    dim3 threads(1024) ;
    dim3 blocks((x.rows/_block_size+1023)/1024) ;

    _block_jacobi_solve<<<blocks, threads>>>(A, x, y, tmp, _block_size) ;
    check_CUDA_errors("_block_jacobi_solve") ;
}