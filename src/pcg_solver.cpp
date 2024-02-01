
#include <cmath>
#include <sstream>
#include <stdexcept>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>


#include "pcg_solver.h"
#include "timing.h"



/* PCG_Solver::operator() 
 *
 * Simple Conjugate-Gradient solver from Nvidia:  
        https://docs.nvidia.com/cuda/incomplete-lu-cholesky/index.html
 */
bool PCG_Solver::operator()(const CSR_SpMatrix& mat, const DnVec& rhs, DnVec& x) {
    // Setup preconditioner
    ILU_precond precond(mat) ;

    return this->operator()(mat, rhs, x, precond) ;
}

bool PCG_Solver::operator()(const CSR_SpMatrix& mat, const DnVec& rhs, DnVec& x, Precond& precond) {

    CodeTiming::BlockTimer timing_block = 
        timer->StartNewTimer("PCG_Solver::operator()") ;

    cusparseStatus_t status ;
    const double one = 1, zero = 0, minus_one = -1 ;
    

    // PCG Step 0: Compute initial residual r = f -  A x0 
    //             (using initial guess in x) and normalization

    DnVec r(mat.rows), p(mat.rows), q(mat.rows), z(mat.rows) ;

    // Check storage again
    size_t buffer_size ;
    cusparseSpMV_bufferSize(
        _handle_cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &minus_one, mat.descr, x.descr, &zero, r.descr, 
        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1, &buffer_size) ;

    CudaArray<char> buffer = make_CudaArray<char>(buffer_size) ;

    cusparseSpMV(_handle_cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                  &minus_one, mat.descr, x.descr, &zero, r.descr, 
                  CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1, buffer.get()) ;

    cublasDaxpy(_handle_cublas, mat.rows, &one, rhs.get(), 1, r.get(), 1) ;

    double normrhs, normr ;
    cublasDdot(_handle_cublas, r.rows, r.get(), 1, r.get(), 1, &normr);
    cublasDdot(_handle_cublas, rhs.rows, rhs.get(), 1, rhs.get(), 1, &normrhs);

    //std::cout << "iteration: 0, norm:" << std::sqrt(normr/normrhs) << "\n" ;

    // PCG Loop: 
    double rho = 0 ;
    for (int iter=0; iter < _max_iter; iter++){
        // Step 1 : Solve M z = r (sparse lower and upper triangular solves)
        precond.solve(r, z) ;

        // Step 2 : Compute correction direction 
        // rho = r^T z
        double rho0 = rho;
        cublasDdot(_handle_cublas, r.rows, r.get(), 1, z.get(), 1, &rho);
        if (iter == 0){
            // p = z
            cublasDcopy(_handle_cublas, z.rows, z.get(), 1, p.get(), 1);
        }
        else{
            // \beta = rho_{i} / \rho_{i-1}
            double beta= rho/rho0;
            // p = z = \beta p + z
            cublasDaxpy(_handle_cublas, p.rows, &beta, p.get(), 1, z.get(), 1) ;
            cublasDcopy(_handle_cublas, z.rows, z.get(), 1, p.get(), 1);
        }

        // Step 3 : Compute q = A p (sparse matrix-vector multiplication)
        status = cusparseSpMV(
                    _handle_cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &one, mat.descr, p.descr, &zero, q.descr, 
                    CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1, buffer.get()) ;
        if (status != CUSPARSE_STATUS_SUCCESS)
            throw std::runtime_error("PCG: Matrix mult failed") ;
       
        // Step 4: Compute step length and correct
        //  \alpha = \rho_{i} / (p^{T} q)	
        //  x = x + \alpha p
        //  r = r - \alpha q
        double temp; 
        cublasDdot(_handle_cublas, r.rows, p.get(), 1, q.get(), 1, &temp);

        double alpha = rho/temp ; 
        cublasDaxpy(_handle_cublas, p.rows, &alpha, p.get(), 1, x.get(), 1) ;
        double minus_alpha = -1*alpha ;
        cublasDaxpy(_handle_cublas, q.rows, &minus_alpha, q.get(), 1, r.get(), 1) ;
        

        //check for convergence		      
        cublasDdot(_handle_cublas, r.rows, r.get(), 1, r.get(), 1, &normr);
        //cublasDdot(_handle_cublas, q.rows, q.get(), 1, q.get(), 1, &normr);
        if ((*_check_convergence)(x, r, rhs) && normr < 1e-8*normrhs){
            std::cout << "CG iteration converged. Iterations=" << iter+1 
                      << ", norm=" <<std::sqrt(normr/normrhs) << "\n" ;
            return 1;
        }
    }  

    std::cout << "CG iteration did not converge. Iterations=" << _max_iter 
             << "\n\tnorm=" << std::sqrt(normr)<< "/" << std::sqrt(normrhs) 
             << "=" << std::sqrt(normr/normrhs) << "\n" ;
    return 0 ;
}


/* PCG_Solver::solve_non_symmetric() 
 *
 * Simple Bi-conjugate Gradient Stabilized solver from
        https://docs.nvidia.com/cuda/incomplete-lu-cholesky/index.html
 */
bool PCG_Solver::solve_non_symmetric(const CSR_SpMatrix& mat, const DnVec& rhs, DnVec& x) {
    // Setup preconditioner
    ILU_precond precond(mat) ;
    return solve_non_symmetric(mat, rhs, x, precond) ;
}

bool PCG_Solver::solve_non_symmetric(const CSR_SpMatrix& mat, const DnVec& rhs, DnVec& x, Precond& precond) {

    CodeTiming::BlockTimer timing_block = 
        timer->StartNewTimer("PCG_Solver::solve_non_symmetric") ;

    cusparseStatus_t status ;
    const double one = 1, zero = 0, minus_one = -1 ;
    
    

    // BiCGStab Step 0: Compute initial residual r = f -  A x0 
    //                (using initial guess in x) and normalization

    DnVec r(mat.rows), rt(mat.rows), p(mat.rows), q(mat.rows) ;
    DnVec y(mat.rows), z(mat.rows) ; // Temporary storage
    DnVec& t = y ;

    // Check storage again
    size_t buffer_size ;
    cusparseSpMV_bufferSize(
        _handle_cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &minus_one, mat.descr, x.descr, &zero, r.descr, 
        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1, &buffer_size) ;

    CudaArray<char> buffer = make_CudaArray<char>(buffer_size) ;

    cusparseSpMV(_handle_cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                  &minus_one, mat.descr, x.descr, &zero, r.descr, 
                  CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1, buffer.get()) ;

    cublasDaxpy(_handle_cublas, mat.rows, &one, rhs.get(), 1, r.get(), 1) ;

    // Step 0.5: Set p = r, rp = r, and compute norms
    cublasDcopy(_handle_cublas, r.rows, r.get(), 1, rt.get(), 1);
    cublasDcopy(_handle_cublas, r.rows, r.get(), 1, p.get(), 1);

    double normrhs, normr ;
    cublasDdot(_handle_cublas, r.rows, r.get(), 1, r.get(), 1, &normr);
    cublasDdot(_handle_cublas, rhs.rows, rhs.get(), 1, rhs.get(), 1, &normrhs);


    //std::cout << "iteration: 0, norm:" << std::sqrt(normr/normrhs) << "\n" ;

    // BiCGStab Loop: 
    double rho = 1, rhop = 1, omega = 1, alpha = 1;
    int restart = 1 ;
    for (int iter=0; iter < _max_iter; iter++){

        CodeTiming::BlockTimer sub_block = 
            timer->StartNewTimer("PCG_Solver::solve_non_symmetric::init_step") ;

        // Step 1: Compute rho = rt^T * r
        rhop = rho;
        cublasDdot(_handle_cublas, r.rows, r.get(), 1, rt.get(), 1, &rho);

        if (rho == 0) {
            std::cout << "BiCGStab failed (rho=0). Iterations=" << iter << "\n";
            std::cout << "\tnorm=" << std::sqrt(normr/normrhs) << "\n" ;
            std::cout << "Restarting.." << std::endl ;
                //break;

            cusparseSpMV(_handle_cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                          &minus_one, mat.descr, x.descr, &zero, r.descr, 
                          CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1, buffer.get()) ;

            cublasDaxpy(_handle_cublas, mat.rows, &one, rhs.get(), 1, r.get(), 1) ;

            // Step 0.5: Set p = r, rp = r, and compute norms
            cublasDcopy(_handle_cublas, r.rows, r.get(), 1, rt.get(), 1);
            cublasDcopy(_handle_cublas, r.rows, r.get(), 1, p.get(), 1);
            
            // Recompute rho
            cublasDdot(_handle_cublas, r.rows, r.get(), 1, rt.get(), 1, &rho);
        }
        //std::cout << "rho: " << rho << "\n" ;

        // Step 2: Compute \beta and p
        if (restart == 0){
            // \beta = (\rho_{i} / \rho_{i-1}) ( \alpha / \omega )
            double beta = (rho/rhop)*(alpha/omega);
            // p = r + \beta (p - \omega v)
            double minus_omega = -omega ;
            double one = 1 ;
            cublasDaxpy(_handle_cublas, p.rows, &minus_omega, q.get(), 1, p.get(), 1) ;
            cublasDscal(_handle_cublas, p.rows, &beta, p.get(), 1);
            cublasDaxpy(_handle_cublas, p.rows, &one, r.get(), 1, p.get(), 1) ;
        }
        restart = 0 ;

        sub_block.EndTiming() ;

        // Step 2 : Solve M p = y (sparse lower and upper triangular solves)
        precond.solve(p, y) ;

        sub_block.StartNewBlock("PCG_Solver::solve_non_symmetric::step1_matvec") ;

        // Step 3 : Compute q = A y (sparse matrix-vector multiplication)
        status = cusparseSpMV(
                    _handle_cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &one, mat.descr, y.descr, &zero, q.descr, 
                    CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1, buffer.get()) ;
        if (status != CUSPARSE_STATUS_SUCCESS)
            throw std::runtime_error("BiCGStab: 1st Matrix mult failed") ;

        sub_block.StartNewBlock("PCG_Solver::solve_non_symmetric::step1_vecvec") ;

        // Step 4 : Update alpha, r, x
        //    \alpha = \rho_{i} / (rt^{T} q)
        //    x = x + \alpha y
        //    r = r - \alpha q
        double temp ;
        cublasDdot(_handle_cublas, rt.rows, rt.get(), 1, q.get(), 1, &temp);

        //std::cout << "alpha: " << rho << "/" << temp ;
        alpha = rho/temp;       
        //std::cout << "=" << alpha << "\n" ;
        cublasDaxpy(_handle_cublas, y.rows, &alpha, y.get(), 1, x.get(), 1) ;
        double minus_alpha = -1*alpha ;
        cublasDaxpy(_handle_cublas, q.rows, &minus_alpha, q.get(), 1, r.get(), 1) ; 


        //check for convergence		      
        cublasDdot(_handle_cublas, r.rows, r.get(), 1, r.get(), 1, &normr);
        
        
        // converged = normr <= _tol*_tol*normrhs ;
        cudaDeviceSynchronize();
        bool converged = (*_check_convergence)(x, r, rhs) && normr < 1e-8*normrhs;

        if (iter > 0 && converged){
            std::cout << "BiCGStab iteration converged. Iterations=" << iter+0.5
                      << ", norm=" <<std::sqrt(normr/normrhs) << "\n" ;
            return 1;
        }     

        sub_block.EndTiming() ;

        // Step 5: Second LU solve:  M z = r 
        precond.solve(r, z) ;

        sub_block.StartNewBlock("PCG_Solver::solve_non_symmetric::step2_matvec") ;

        // Step 6 : Compute t = A z (sparse matrix-vector multiplication)
        status = cusparseSpMV(
                    _handle_cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &one, mat.descr, z.descr, &zero, t.descr, 
                    CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1, buffer.get()) ;
        if (status != CUSPARSE_STATUS_SUCCESS)
            throw std::runtime_error("BiCGStab: 2nd Matrix mult failed") ;

        sub_block.StartNewBlock("PCG_Solver::solve_non_symmetric::step2_vecvec") ;

        // Step 7 : Update omega, and x, r for the second time
        //    \omega = (t^{T} r) / (t^{T} t)
        //    x = x + \omega z
        //    r = s - \omega t
        cublasDdot(_handle_cublas, t.rows, t.get(), 1, r.get(), 1, &omega);
        cublasDdot(_handle_cublas, t.rows, t.get(), 1, t.get(), 1, &temp);

        if (omega == 0 || temp == 0) {
            std::cout << "BiCGStab Failed. Omega=" << omega << "/" << temp
                      << ". Iterations=" << iter+0.5 << "\n" ;
            return 0;
        }
        //std::cout << "omega: " << omega << "/" << temp ;
        omega /= temp ;
        //std::cout << "=" << omega << "\n" ;

        cublasDaxpy(_handle_cublas, z.rows, &omega, z.get(), 1, x.get(), 1) ;
        double minus_omega = -1*omega ;
        cublasDaxpy(_handle_cublas, t.rows, &minus_omega, t.get(), 1, r.get(), 1) ; 


        //check for convergence		      
        cublasDdot(_handle_cublas, r.rows, r.get(), 1, r.get(), 1, &normr);

        //converged = normr <= _tol*_tol*normrhs ;
        converged = (*_check_convergence)(x, r, rhs) && normr < 1e-8*normrhs;

        if (converged){
            std::cout << "BiCGStab iteration converged. Iterations=" << iter+1 
                      << ", norm=" <<std::sqrt(normr/normrhs) << "\n" ;
            return 1;
        }
    }  

    std::cout << "BiCGStab iteration did not converge. Iterations=" << _max_iter 
             << "\n\tnorm=" << std::sqrt(normr)<< "/" << std::sqrt(normrhs) 
             << "=" << std::sqrt(normr/normrhs) << "\n" ;
    return 0 ;
      
}
