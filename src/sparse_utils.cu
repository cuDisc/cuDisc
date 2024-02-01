

#include <stdexcept>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>


#include "pcg_solver.h"

__global__ void _create_identity_device(CSR_SpMatrixConstRef m, CSR_SpMatrixRef I) {

    int row = threadIdx.x + blockIdx.x * blockDim.x ;

    if (row < m.rows) {
        int start = m.csr_offset[row] ;
        int end =  m.csr_offset[row+1] ;
        
        if (row == 0)
            I.csr_offset[0] = start ;
        I.csr_offset[row+1] = end ;

        for ( ; start < end; start++) {
            I.col_index[start] = m.col_index[start] ;
            if (I.col_index[start] == row)
                I.data[start] = 1;
            else 
                I.data[start] = 0;
        }
    }
}


CSR_SpMatrix create_sparse_identity(const CSR_SpMatrix& m) {

    CSR_SpMatrix I(m.rows, m.cols, m.non_zeros) ;

    dim3 threads(1024) ;
    dim3 blocks((m.rows+1023)/1024) ;

    _create_identity_device<<<blocks, threads>>>(m, I) ;
    check_CUDA_errors("_create_identity_device") ;

    return I ;
}

CSR_SpMatrix copy_sparse_matrix(const CSR_SpMatrix& m) {

    CSR_SpMatrix m_copy(m.rows, m.cols, m.non_zeros) ;

    cudaMemcpy(m_copy.csr_offset.get(), m.csr_offset.get(), (m.rows+1)*sizeof(int), 
               cudaMemcpyDeviceToDevice) ;
    cudaMemcpy(m_copy.col_index.get(), m.col_index.get(), m.non_zeros*sizeof(int),
               cudaMemcpyDeviceToDevice) ;
    cudaMemcpy(m_copy.data.get(), m.data.get(), m.non_zeros*sizeof(double), 
               cudaMemcpyDeviceToDevice) ;

    return m_copy; 
}
