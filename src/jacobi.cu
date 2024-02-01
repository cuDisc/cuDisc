
#include <cuda_runtime.h>

#include "matrix_types.h"
#include "pcg_solver.h"
#include "utils.h"

__global__ void get_diags(CSR_SpMatrixConstRef A, DnVecRef diag, int block_size) {

    int idx = threadIdx.x + blockIdx.x*blockDim.x ;

    if (idx < A.rows) {
        // Take the first element of the block:
        int idx_b = (idx/block_size) * block_size ;
        
        int start = A.csr_offset[idx_b] ;
        int end = A.csr_offset[idx_b+1] ;

        double scal = 1 ;
        for (int i=start; i < end; i++)
            if (A.col_index[i] == idx_b) {
                scal = A.data[i] ;
                if (scal < 0) scal *= -1 ;
                if (scal == 0) scal = 1 ;
                
                scal = sqrt(scal) ;
            }

        diag.data[idx] = scal ;
    }
}


Jacobi_Precond::Jacobi_Precond(const CSR_SpMatrix& A, int block_size)
    : _diag(A.rows)
{
    dim3 threads(1024) ;
    dim3 blocks((A.rows+1023)/1024) ;

    get_diags<<<blocks, threads>>>(A, _diag, block_size) ;
    check_CUDA_errors("get_diags") ;           
}

__global__ void scale_system(DnVecConstRef diag, 
                             CSR_SpMatrixRef A, DnVecRef x, DnVecRef b) {

    int idx = threadIdx.x + blockIdx.x*blockDim.x ;

    if (idx < A.rows) {

        double d_idx = diag.data[idx] ;
        x.data[idx] *= d_idx ;
        b.data[idx] /= d_idx ;

        int start = A.csr_offset[idx] ;
        int end = A.csr_offset[idx+1] ;

        for (int i=start; i < end; i++)
            A.data[i] /= d_idx * diag.data[A.col_index[i]] ;
    }
}

void Jacobi_Precond::transform(CSR_SpMatrix& A, DnVec& x, DnVec& b) const {

    dim3 threads(1024) ;
    dim3 blocks((A.rows+1023)/1024) ;

    scale_system<<<blocks, threads>>>(_diag, A, x, b) ;
    check_CUDA_errors("scale_system") ;       
}

__global__ void scale_x(DnVecConstRef diag, DnVecRef x) {

    int idx = threadIdx.x + blockIdx.x*blockDim.x ;

    if (idx < x.rows) {
        x.data[idx] *=  diag.data[idx] ;
    }
}

void Jacobi_Precond::transform_guess(DnVec& x) const {

    dim3 threads(1024) ;
    dim3 blocks((x.rows+1023)/1024) ;

    scale_x<<<blocks, threads>>>(_diag, x) ;
    check_CUDA_errors("scale_x") ;       
}


__global__ void remove_scale_from_x(DnVecConstRef diag, DnVecRef x) {

    int idx = threadIdx.x + blockIdx.x*blockDim.x ;

    if (idx < x.rows) {
        x.data[idx] /=  diag.data[idx] ;
    }
}

void Jacobi_Precond::invert(DnVec& x) const {

    dim3 threads(1024) ;
    dim3 blocks((x.rows+1023)/1024) ;

    remove_scale_from_x<<<blocks, threads>>>(_diag, x) ;
    check_CUDA_errors("remove_scale_from_x") ;       
}
