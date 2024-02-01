#include <sstream>
#include <stdexcept>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

#include "pcg_solver.h"
#include "timing.h"

NoPrecond::NoPrecond() {
    cublasStatus_t status_cub = 
        cublasCreate(&_handle_cublas) ;
    if (status_cub != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("Failed to initialize CUBLAS") ;
}
NoPrecond::~NoPrecond() {
    cublasDestroy(_handle_cublas) ;
}

void NoPrecond::solve(const DnVec& rhs, DnVec& x) {
    cublasDcopy(_handle_cublas, rhs.rows, rhs.get(), 1,  x.get(), 1) ;

} 

ILU_precond::ILU_precond(const CSR_SpMatrix& mat, int k) 
  :  tmp(mat.rows)
{

    CodeTiming::BlockTimer timing_block = 
        timer->StartNewTimer("ILU_precond::ILU_precond") ;

    cusparseStatus_t status ;

    status = cusparseCreate(&_handle_cusparse) ;
    if (status != CUSPARSE_STATUS_SUCCESS)
        throw std::runtime_error("ILU_precond:Failed to initialize CUSPARSE") ;

    cublasCreate(&_handle_cublas) ;

    // Step 0: Setup info obejcts and policy for choleksy
    LUFacInfo infoFac ;

    CSR_SpMatrix matLU = get_ILUk_shape(mat, k) ;

    // Create a wrapper for the old-style interface
    //    Needed for the LU factorization routines.
    CSR_SpMatrixSimple LUref(matLU) ;

   // Step 1: Create the max buffer size needed.
   int buffer_size1 ;
   cusparseDcsrilu02_bufferSize(_handle_cusparse,
                                LUref.rows, LUref.non_zeros, LUref.descr,
                                LUref.data, LUref.csr_offset, LUref.col_index,
                                infoFac, &buffer_size1) ;

   CudaArray<char> buffer = make_CudaArray<char>(buffer_size1) ;


   // Init Step 2a: Perform analysis of LU factorization
   status = cusparseDcsrilu02_analysis(_handle_cusparse,
                                       LUref.rows, LUref.non_zeros, LUref.descr,
                                       LUref.data, LUref.csr_offset, LUref.col_index,
                                       infoFac, policyFac,
                                       buffer.get()) ;

   if (status != CUSPARSE_STATUS_SUCCESS)
       throw std::runtime_error("LU factorization analysis failed") ;

   int structural_zero ;
   status = cusparseXcsrilu02_zeroPivot(_handle_cusparse, infoFac, &structural_zero);
   
   if (status == CUSPARSE_STATUS_ZERO_PIVOT) {
       std::stringstream msg ;
       msg << "Structural zero at A(" 
           << structural_zero << "," << structural_zero << ")"
           << " in incomplete LU factorization\n" ;
       throw std::runtime_error(msg.str()) ;
   }


   // Init Step 3: Perform incomplete-LU factorization
   status = cusparseDcsrilu02(_handle_cusparse,
                             LUref.rows, LUref.non_zeros, LUref.descr,
                             LUref.data, LUref.csr_offset, LUref.col_index,
                             infoFac, policyFac, buffer.get());

   if (status != CUSPARSE_STATUS_SUCCESS)
       throw std::runtime_error("LU factorization failed") ;

   int numerical_zero ;
   status = cusparseXcsrilu02_zeroPivot(_handle_cusparse, infoFac, &numerical_zero);
   if (status == CUSPARSE_STATUS_ZERO_PIVOT){
       std::stringstream msg ;
       msg << "Zero at A(" << numerical_zero << "," << numerical_zero << ")"
           << " in incomplete LU factorization\n" ;
       throw std::runtime_error(msg.str()) ;
   }


    // Finally, store the LU factorization
    LUfac = CSR_SpLUfac(std::move(matLU)) ;

    // Setup the triangular solve
    DnVec in(mat.rows), out(mat.rows) ;
    setup(in, out) ;
}


void ILU_precond::setup(const DnVec& rhs, DnVec& result) {
    // Init step 4: Setup the triangular solvers

    // Lower triangular matric
    //  allocate an external buffer for analysis
    size_t buffersize ;
    double one = 1 ;
    cusparseSpSV_bufferSize(
            _handle_cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one, LUfac.matL, rhs, tmp, CUDA_R_64F,
            CUSPARSE_SPSV_ALG_DEFAULT, solveL, &buffersize) ;
    bufferL = make_CudaArray<char>(buffersize) ;

    //  Do pre-analysis for the transform      
    cusparseSpSV_analysis(
            _handle_cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one, LUfac.matL, rhs, tmp, CUDA_R_64F,
            CUSPARSE_SPSV_ALG_DEFAULT, solveL, bufferL.get()) ;

    // Upper triangular matric
    //  allocate an external buffer for analysis
    cusparseSpSV_bufferSize(
            _handle_cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one, LUfac.matU, tmp, result, CUDA_R_64F,
            CUSPARSE_SPSV_ALG_DEFAULT, solveU, &buffersize) ;
    bufferU = make_CudaArray<char>(buffersize) ;

    //  Do pre-analysis for the transform      
    cusparseSpSV_analysis(
            _handle_cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one, LUfac.matU, tmp, result, CUDA_R_64F,
            CUSPARSE_SPSV_ALG_DEFAULT, solveU, bufferU.get()) ;
}

void ILU_precond::solve(const DnVec& rhs, DnVec& result) {

    CodeTiming::BlockTimer timing_block = 
        timer->StartNewTimer("ILU_precond::solve") ;

    cusparseStatus_t status ;
    double one = 1 ;

    // execute SpSV
    status =  cusparseSpSV_solve(
            _handle_cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one, LUfac.matL, rhs, tmp, CUDA_R_64F,
            CUSPARSE_SPSV_ALG_DEFAULT, solveL) ;

    if (status != CUSPARSE_STATUS_SUCCESS)
        throw std::runtime_error("BiCGStab: Preconditioning step 1 failed") ;

    status =  cusparseSpSV_solve(
            _handle_cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one, LUfac.matU, tmp, result, CUDA_R_64F,
            CUSPARSE_SPSV_ALG_DEFAULT, solveU) ;

    if (status != CUSPARSE_STATUS_SUCCESS)
        throw std::runtime_error("BiCGStab: Preconditioning step 2 failed") ;

}



CSR_SpMatrix ILU_precond::get_ILUk_shape(const CSR_SpMatrix& mat, int k) {

    if (k == 0) return copy_sparse_matrix(mat) ;

    CodeTiming::BlockTimer timing_block = 
        timer->StartNewTimer("ILU_precond::get_ILUk_shape") ;

    CSR_SpMatrix matA = copy_sparse_matrix(mat) ;
    CSR_SpMatrix matI = create_sparse_identity(mat) ;

    double alpha = 1, beta = 0 ;
    
    cusparseSetPointerMode(_handle_cusparse, CUSPARSE_POINTER_MODE_HOST);
    SpGEMMDescr spgemmDesc ;

    size_t bufferSize1 = 0 ;
    CudaArray<char> buffer1 ;
    size_t bufferSize2 = 0 ;
    CudaArray<char> buffer2 ;
    for ( ; k > 0 ; k--) {

        CSR_SpMatrix matC(matA.rows, matI.cols, 0) ;

        // Get the buffer size
        size_t bufferSize_new ;
        cusparseStatus_t status = 
            cusparseSpGEMM_workEstimation(_handle_cusparse,
                CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, matI, &beta, matC,
                CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                spgemmDesc, &bufferSize_new, NULL) ;
        if (status != CUSPARSE_STATUS_SUCCESS)
            throw std::runtime_error(
                    "ILU_precond: Failed to get create expanded matrix (1)") ;

        if (bufferSize_new > bufferSize1) {
            bufferSize1 = bufferSize_new ;
            buffer1 = make_CudaArray<char>(bufferSize1) ;
        }

        // inspect the matrices A and B to understand the memory requirement for
        // the next step
        status =
            cusparseSpGEMM_workEstimation(_handle_cusparse,
                CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, matI, &beta, matC,
                CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                spgemmDesc, &bufferSize_new, buffer1.get()) ;
        if (status != CUSPARSE_STATUS_SUCCESS)
            throw std::runtime_error(
                    "ILU_precond: Failed to get create expanded matrix (2)") ;

        // ask bufferSize_new bytes for external memory
        status = 
            cusparseSpGEMM_compute(_handle_cusparse,
                CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, matI, &beta, matC,
                CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                spgemmDesc, &bufferSize_new, NULL) ;
        if (status != CUSPARSE_STATUS_SUCCESS)
            throw std::runtime_error(
                    "ILU_precond: Failed to get create expanded matrix (3)") ;

        if (bufferSize_new > bufferSize2) {
            bufferSize2 = bufferSize_new ;
            buffer2 = make_CudaArray<char>(bufferSize2) ;
        }

        // compute the intermediate product of A * B
        status = 
            cusparseSpGEMM_compute(_handle_cusparse,
                CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, matI, &beta, matC,
                CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                spgemmDesc, &bufferSize_new, buffer2.get()) ;
        if (status != CUSPARSE_STATUS_SUCCESS)
            throw std::runtime_error(
                    "ILU_precond: Failed to get create expanded matrix (4)") ;

        // allocate matrix C non-zero entries
        matC.reallocate() ;


        // copy the final products to the matrix C
        status = cusparseSpGEMM_copy(_handle_cusparse,
                    CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha, matA, matI, &beta, matC,
                    CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                    spgemmDesc) ;
        if (status != CUSPARSE_STATUS_SUCCESS)
            throw std::runtime_error(
                    "ILU_precond: Failed to get create expanded matrix (5)") ;
   
        // Save the new result
        matA = std::move(matC) ;
    }
   
    return matA ;

}