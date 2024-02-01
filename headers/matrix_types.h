
#ifndef _CUSDISC_MATRIX_TYPES_H_
#define _CUSDISC_MATRIX_TYPES_H_

#include <iostream>
#include <iomanip>

#include <cublas_v2.h>
#include <cusparse.h>

#include "cuda_array.h"

/* class CSR_SpMatrix
 *
 * Wrapper object for GPU Compressed Sparse Row matrix format
 */
class CSR_SpMatrix 
{
  public:
    CSR_SpMatrix(int rows_, int cols_, int non_zeros_) 
      : rows(rows_), cols(cols_), non_zeros(non_zeros_),
        csr_offset(make_CudaArray<int>(rows_+1)),
        col_index(make_CudaArray<int>(non_zeros_)),
        data(make_CudaArray<double>(non_zeros_))
    { 
        cusparseStatus_t status = 
            cusparseCreateCsr(&descr, rows, cols, non_zeros,
                              csr_offset.get(), col_index.get(), data.get(),
                              CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                              CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) ;
        
        if (status != CUSPARSE_STATUS_SUCCESS)
            throw std::runtime_error(
                "CSR_SpMatrix: Failed to create matrix descriptor") ;

        init = true ;
    }

    CSR_SpMatrix(CSR_SpMatrix&& o) 
       : descr(o.descr), 
         rows(o.rows), cols(o.cols), non_zeros(o.non_zeros),
         csr_offset(std::move(o.csr_offset)),
         col_index(std::move(o.col_index)),
         data(std::move(o.data))
    {
        init = true ;
        o.init = false ;
    }

    CSR_SpMatrix& operator=(CSR_SpMatrix&& o) 
    {
        if (init) cusparseDestroySpMat(descr) ;
        init = false ;

        descr = o.descr ;
        
        rows = o.rows ;
        cols = o.cols ;
        non_zeros = o.non_zeros ;

        csr_offset = std::move(o.csr_offset) ;
        col_index = std::move(o.col_index) ;
        data = std::move(o.data) ;

        init = true ;
        o.init = false ;
    
        return *this ;
    }



    ~CSR_SpMatrix() {
        if (init) cusparseDestroySpMat(descr) ;
    }

    /* Some CUDA functions will update the number of matrix
     * elements, so we need to be able to reallocate. Any existing
     * data will be lost. */
    void reallocate() {

        int64_t C_num_rows1, C_num_cols1, C_nnz1;
        cusparseStatus_t status = 
            cusparseSpMatGetSize(descr, &C_num_rows1, &C_num_cols1, &C_nnz1) ;
        if (status != CUSPARSE_STATUS_SUCCESS)
                throw std::runtime_error(
                    "CSR_SpMatrix: Failed to get matrix size") ;

        /* Don't do anything if the matrix is already the correct shape */
        if (rows == static_cast<int>(C_num_rows1) &&
            cols == static_cast<int>(C_num_cols1) &&
             non_zeros == static_cast<int>(C_nnz1)) 
            return ;

        /* Reallocate the data */
        rows = static_cast<int>(C_num_rows1) ;
        cols = static_cast<int>(C_num_cols1) ;
        non_zeros = static_cast<int>(C_nnz1) ;

        csr_offset = make_CudaArray<int>(rows+1) ;
        col_index = make_CudaArray<int>(non_zeros) ;
        data = make_CudaArray<double>(non_zeros) ;

        status = cusparseCsrSetPointers(
            descr, csr_offset.get(), col_index.get(), data.get()
        ) ;
        
        if (status != CUSPARSE_STATUS_SUCCESS)
                throw std::runtime_error(
                    "CSR_SpMatrix: Failed to set matrix descriptor") ;
    }

    operator cusparseSpMatDescr_t() const {
        return descr ;
    }


    cusparseSpMatDescr_t descr ;
    int rows, cols, non_zeros ;
    CudaArray<int> csr_offset ;
    CudaArray<int> col_index ;
    CudaArray<double> data ;

  private:
    bool init = false ;
} ;


/* class CSR_SpLUfac
 *
 * Wrapper object for LU factorized matrix.
 */
class CSR_SpLUfac {
  public:
    CSR_SpLUfac(CSR_SpMatrix matLU_)
      : matLU(std::move(matLU_))
    {
        _init_LU_matrices() ;
    }

    CSR_SpLUfac() : matLU(1,1,0) {  } ;

    CSR_SpLUfac(CSR_SpLUfac&& o) 
       : matL(o.matL), matU(o.matU), matLU(std::move(o.matLU))
    {
        init = true ;
        o.init = false ;
    }

    CSR_SpLUfac& operator=(CSR_SpLUfac&& o) {

        matL = o.matL ;
        matU = o.matU ;
        matLU = std::move(o.matLU) ;

        init = o.init ;
        o.init = false ;

        return *this ;
    }


    ~CSR_SpLUfac() {
        if (init) {
            cusparseDestroySpMat(matL) ;
            cusparseDestroySpMat(matU) ;
        }
    }

    cusparseSpMatDescr_t matL, matU ;

  private:
    void _init_LU_matrices() {
        if (init) {
            cusparseDestroySpMat(matL) ;
            cusparseDestroySpMat(matU) ;
            init = false ;
        }

        // Lower matrix
        cusparseStatus_t status = 
            cusparseCreateCsr(&matL, matLU.rows, matLU.cols, matLU.non_zeros,
                              matLU.csr_offset.get(), matLU.col_index.get(), matLU.data.get(),
                              CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                              CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) ;
        if (status != CUSPARSE_STATUS_SUCCESS)
            throw std::runtime_error(
                    "CSR_SpLUfac: Failed to crease lower triangular matrix") ;

        // Set fill mode / diagnonal
        cusparseFillMode_t fill_mode = CUSPARSE_FILL_MODE_LOWER;
        cusparseSpMatSetAttribute(matL, CUSPARSE_SPMAT_FILL_MODE,
            &fill_mode, sizeof(fill_mode));
        
        cusparseDiagType_t diag_type = CUSPARSE_DIAG_TYPE_UNIT;
        cusparseSpMatSetAttribute(matL, CUSPARSE_SPMAT_DIAG_TYPE,
            &diag_type, sizeof(diag_type));

        // Upper matrix
        status = 
            cusparseCreateCsr(&matU, matLU.rows, matLU.cols, matLU.non_zeros,
                              matLU.csr_offset.get(), matLU.col_index.get(), matLU.data.get(),
                              CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                              CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) ;
        if (status != CUSPARSE_STATUS_SUCCESS)
            throw std::runtime_error(
                    "CSR_SpLUfac: Failed to crease lower triangular matrix") ;

        // Set fill mode / diagnonal
        fill_mode = CUSPARSE_FILL_MODE_UPPER;
        cusparseSpMatSetAttribute(matU, CUSPARSE_SPMAT_FILL_MODE,
            &fill_mode, sizeof(fill_mode));
        
        diag_type = CUSPARSE_DIAG_TYPE_NON_UNIT;
        cusparseSpMatSetAttribute(matU, CUSPARSE_SPMAT_DIAG_TYPE,
            &diag_type, sizeof(diag_type));

        init = true ;
    }

    CSR_SpMatrix matLU ;
    bool init = false ;
} ;


/* struct CSR_SpMatrixSimple
 *
 * Wrapper object for GPU Compressed Sparse Row matrix format using
 * cusparseMatDescr_t interface
 */
struct CSR_SpMatrixSimple
{
    CSR_SpMatrixSimple(const CSR_SpMatrix& mat)
      : rows(mat.rows), cols(mat.cols), non_zeros(mat.non_zeros),
        csr_offset(mat.csr_offset.get()),
        col_index(mat.col_index.get()),
        data(mat.data.get())
    { 
        cusparseStatus_t status = cusparseCreateMatDescr(&descr) ;
        
        if (status != CUSPARSE_STATUS_SUCCESS)
            throw std::runtime_error(
                "CSR_SpMatrixSimple: Failed to create matrix descriptor") ;

        cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
        cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
        
        init = true ;
    }

    void set_FillMode(cusparseFillMode_t fill_mode) {
        cusparseSetMatFillMode(descr, fill_mode);
    }
    void set_DiagType(cusparseDiagType_t diag_type) {
        cusparseSetMatDiagType(descr, diag_type) ;
    }

    ~CSR_SpMatrixSimple() {
        if (init) cusparseDestroyMatDescr(descr) ;
    }

    cusparseMatDescr_t descr ;
    int rows, cols, non_zeros ;
    int* csr_offset ;
    int* col_index ;
    double* data ;
  private:
    bool init = false ;
} ;


/* struct CSR_SpMatrixRef
 *
 * Reference type for CSR_SpMatrix class.
 *  
 * This class exists to enable copying CSR_SpMatrix objects to the GPU. 
 * Since objects can't be passed by reference to __global__ functions we need
 * a special class to handle this. Note that passing by value is impossible 
 * because CudaArrays are non-copyable.
 */
struct CSR_SpMatrixRef 
{
    CSR_SpMatrixRef(CSR_SpMatrix& spm)
     : rows(spm.rows), cols(spm.cols), non_zeros(spm.non_zeros),
       csr_offset(spm.csr_offset.get()),
       col_index(spm.col_index.get()),
       data(spm.data.get()) 
    { } ;

    CSR_SpMatrixRef(CSR_SpMatrixSimple& spm)
     : rows(spm.rows), cols(spm.cols), non_zeros(spm.non_zeros),
       csr_offset(spm.csr_offset),
       col_index(spm.col_index),
       data(spm.data) 
    { } ;
  
    int rows, cols, non_zeros ;
    int* csr_offset ;
    int* col_index ;
    double* data ;
} ;



/* struct CSR_SpMatrixConstRef
 *
 * Const Reference type for CSR_SpMatrix class.
 *  
 * This class exists to enable copying CSR_SpMatrix objects to the GPU. 
 * Since objects can't be passed by reference to __global__ functions we need
 * a special class to handle this. Note that passing by value is impossible 
 * because CudaArrays are non-copyable.
 */
struct CSR_SpMatrixConstRef 
{
    CSR_SpMatrixConstRef(const CSR_SpMatrix& spm)
     : rows(spm.rows), cols(spm.cols), non_zeros(spm.non_zeros),
       csr_offset(spm.csr_offset.get()),
       col_index(spm.col_index.get()),
       data(spm.data.get()) 
    { } ;

    CSR_SpMatrixConstRef(const CSR_SpMatrixSimple& spm)
     : rows(spm.rows), cols(spm.cols), non_zeros(spm.non_zeros),
       csr_offset(spm.csr_offset),
       col_index(spm.col_index),
       data(spm.data) 
    { } ;
  
  
    int rows, cols, non_zeros ;
    const int* csr_offset ;
    const int* col_index ;
    const double* data ;
} ;


/* struct DnVec
 *
 * Wrapper object for cusparse compatible dense vectors
 */
struct DnVec {
    DnVec(int rows_)
     : rows(rows_),
       data(make_CudaArray<double>(rows_))
    {
        cusparseStatus_t status = 
            cusparseCreateDnVec(&descr, rows, data.get(), CUDA_R_64F) ;

        if (status != CUSPARSE_STATUS_SUCCESS)
            throw std::runtime_error(
                "DnVec: Failed to create vector descriptor") ;
    }
    
    ~DnVec() {
        cusparseDestroyDnVec(descr) ;
    }

    DnVec(DnVec&& o) 
      : rows(o.rows), data(std::move(o.data))
    {
        cusparseStatus_t status = 
            cusparseCreateDnVec(&descr, rows, data.get(), CUDA_R_64F) ;

        if (status != CUSPARSE_STATUS_SUCCESS)
            throw std::runtime_error(
                "DnVec: Failed to create vector descriptor") ;
    }

    DnVec& operator=(DnVec&& o) {
        cusparseDestroyDnVec(descr) ;

        rows = o.rows ;
        data = std::move(o.data) ;
        
        cusparseStatus_t status = 
            cusparseCreateDnVec(&descr, rows, data.get(), CUDA_R_64F) ;

        if (status != CUSPARSE_STATUS_SUCCESS)
            throw std::runtime_error(
                "DnVec: Failed to create vector descriptor") ;
        
        return *this ;
    }

    double* get() {
        return data.get() ;
    }

    const double* get() const {
        return data.get() ;
    }

    operator cusparseDnVecDescr_t() const {
        return descr ;
    }

    cusparseDnVecDescr_t descr ;
    int rows ;
    CudaArray<double> data ;
} ;

/* struct DnVecRef
 *
 * Reference type for DnVec class.
 *  
 * This class exists to enable copying DnVec objects to the GPU. 
 * Since objects can't be passed by reference to __global__ functions we need
 * a special class to handle this. Note that passing by value is impossible 
 * because CudaArrays are non-copyable.
 */
struct DnVecRef 
{
    DnVecRef(DnVec& spm)
     : rows(spm.rows), data(spm.data.get()) 
    { } ;
  
    int rows ;
    double* data ;
} ;

/* struct DnVecConstRef
 *
 * Const Reference type for DnVec class.
 *  
 * This class exists to enable copying DnVec objects to the GPU. 
 * Since objects can't be passed by reference to __global__ functions we need
 * a special class to handle this. Note that passing by value is impossible 
 * because CudaArrays are non-copyable.
 */
struct DnVecConstRef 
{
    DnVecConstRef(const DnVec& spm)
     : rows(spm.rows), data(spm.data.get()) 
    { } ;
  
    int rows ;
    const double* data ;
} ;


// Create a sparse identity matrix with the shape of m
CSR_SpMatrix copy_sparse_matrix(const CSR_SpMatrix& m) ;
CSR_SpMatrix create_sparse_identity(const CSR_SpMatrix& m) ;


/* print_CSR_matrix
 *
 * Does exactly what it says on the tin
 */
inline void print_CSR_struct(CSR_SpMatrixConstRef mat, int max_rows=-1) {

    if (max_rows < 0)
        max_rows = mat.rows ;
    else 
        max_rows = std::min(max_rows, mat.rows) ;

    for (int i = 0; i < max_rows; i++) {
        int j = mat.csr_offset[i] ;
        int n = mat.csr_offset[i+1] ;
        
        std::cout << i << ": [" ;
        for ( ; j <n ;j++) 
            std::cout << "(" << mat.col_index[j] << "," << mat.data[j] << "), " ;
        std::cout << "],\n" ;
    } 
}

inline void print_CSR_matrix(CSR_SpMatrixConstRef mat, int max_rows=-1) {

    int k = 0 ;
    if (max_rows < 0)
        max_rows = mat.rows ;
    else 
        max_rows = std::min(max_rows, mat.rows) ;
        
    for (int i = 0; i < max_rows; i++) {
        for (int j =0; j < mat.cols; j++) {
            std::cout << std::right << std::setfill(' ') << std::setw(9) << std::setprecision(3) ;
            if (j == mat.col_index[k]) {
                std::cout << mat.data[k] << " " ;
                k++ ;
            }
            else {
                std::cout << ".         " ;
            }
        }
        std::cout << "\n" ;
    } 
}

inline void print_CSR_dot_product(CSR_SpMatrixConstRef mat, DnVecConstRef v, int max_rows=-1) {

    if (max_rows < 0)
        max_rows = mat.rows ;
    else 
        max_rows = std::min(max_rows, mat.rows) ;

    for (int i = 0; i < max_rows; i++) {
        int j = mat.csr_offset[i] ;
        int n = mat.csr_offset[i+1] ;
        
        double res = 0 ;
        for ( ; j <n ;j++) 
            res += mat.data[j] * v.data[mat.col_index[j]] ;
        std::cout << i << " " << res << "\n" ;
    } 
}

inline void print_vec(DnVecConstRef vec) {

    for (int i = 0; i < vec.rows; i++)
        std::cout << vec.data[i] << " " ;
    std::cout << "\n" ;
}

inline void write_MM_format(const CSR_SpMatrix& mat, std::ostream& f) {

    f << "%%MatrixMarket matrix coordinate real general\n" ;
    f << mat.rows << " " << mat.cols << " " << mat.non_zeros << "\n";
    f << std::setprecision(16) ;

    for (int i = 0; i < mat.rows; i++) {
        int j = mat.csr_offset[i] ;
        int n = mat.csr_offset[i+1] ;
        
        for ( ; j <n ;j++) 
            f << (i+1) << " " << (mat.col_index[j]+1) << " " << mat.data[j] << "\n" ;
    } 
}
inline void write_MM_format(const DnVec& vec, std::ostream& f) {

    f << "%%MatrixMarket matrix array real general\n" ;
    f << vec.rows << " " << 1 << " " << vec.rows << "\n";
    f << std::setprecision(16) ;

    for (int i = 0; i < vec.rows; i++)
        f << vec.data[i] << "\n" ;
}


#endif// _CUSDISC_MATRIX_TYPES_H_
