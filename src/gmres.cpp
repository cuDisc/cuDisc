#include <cmath>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>


#include "pcg_solver.h"
#include "timing.h"


/* GMRES_Solver::operator() 
*
* Simple restarted GMRES solver.
       https://docs.nvidia.com/cuda/incomplete-lu-cholesky/index.html
*/
void GMRES_Solver::operator()(const CSR_SpMatrix& mat, const DnVec& rhs, DnVec& x) {

    CodeTiming::BlockTimer timing_block = 
        timer->StartNewTimer("GMRES_Solver::operator()") ;

    ILU_precond precond(mat) ;

   // GMRES outer loop
   int total_iterations = 0;
   GMRES_Solver::_gmres_result result ;
   for (int iter = 0; iter < _max_iters+1; iter++) {

        result = gmres_loop(mat, rhs, x, precond) ;

        total_iterations += result.iterations ;
  
        if (result.success) {
            std::cout << "GMRES converged: Iterations=" << total_iterations
                      << ", restarts=" << iter << ". norm=" << result.error 
                      << std::endl ;
            return ;
        }
   }

    std::cout << "GMRES failed: Iterations=" << total_iterations
              << ", restarts=" << _max_iters << ". norm=" << result.error 
              << std::endl ;
}

/* GMRES_Solver::gmres_loop() 
*
* A single k-iteration GMRES loop. 
*/
GMRES_Solver::_gmres_result GMRES_Solver::gmres_loop(
    const CSR_SpMatrix& A, const DnVec& rhs, DnVec& x0,
    Precond& precond) const {

    // GMRES Step 0 : Compute initial residual r = f -  A x0 
    //                (using initial guess in x) and normalization
     
    double minus_one=-1, zero = 0, one=1 ;

    DnVec r(A.rows) ;

    size_t buffer_size ;
    cusparseSpMV_bufferSize(
        _handle_cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &minus_one, A.descr, x0.descr, &zero, r.descr, 
        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1, &buffer_size) ;

    CudaArray<char>  buffer = make_CudaArray<char>(buffer_size) ;

    cusparseSpMV(_handle_cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &minus_one, A.descr, x0.descr, &zero, r.descr, 
        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1, buffer.get()) ;

    cublasDaxpy(_handle_cublas, A.rows, &one, rhs.get(), 1, r.get(), 1) ;

    // Check for convergence already
    double normrhs, normr ;
    cublasDnrm2(_handle_cublas, r.rows, r.get(), 1, &normr);
    cublasDnrm2(_handle_cublas, rhs.rows, rhs.get(), 1, &normrhs);

    double error = normr/normrhs ;
    //std::cout << "iteration: 0, norm:" << error << "\n" ;

    if (error <= _tol){
        GMRES_Solver::_gmres_result result ;
       
        result.success = 1 ;
        result.iterations = 0 ;
        result.error = error;

        return result ;
    }

    // Setup storage
    std::vector<double>
        cn(_n_restart+1), 
        sn(_n_restart+1) ;

    std::vector<std::vector<double>> H(_n_restart+1) ;
    
    std::vector<double> beta(_n_restart+1, 0) ;
    beta[0] = normr ;
    
    std::vector<DnVec> Q; Q.reserve(_n_restart+1) ;

    // Initial krylov vector Q[0] = r/norm(r)
    double tmp = 1/normr ;
    Q.push_back(DnVec(A.rows)) ;
    Q.push_back(DnVec(A.rows)) ;

    cublasDcopy(_handle_cublas, A.rows, r.get(), 1, Q[0].get(), 1);
    cublasDscal(_handle_cublas, A.rows, &tmp, Q[0].get(), 1) ;

    int n, success = false ;
    for (n=0; n < _n_restart; n++) {
        // Get the new vector
        std::vector<double>& h = H[n] ; h.resize(n+2) ;
        arnoldi(A, Q, h, n, precond, buffer.get()) ;

        // Find the new residual
        normr = minimize_residual(n, h, cn, sn, beta) ;
        //std::cout << "iteration: " << n + 1
        //          << " norm=" << normr << "/" << normrhs << "=" << normr/normrhs << "\n" ;
        if (normr < _tol*normrhs)  {
            n++ ;
            success = true ;
            break ;
        }
    }
    n-- ;

    // Update the solution
    //   Solve triangular system for y_n in place
    for (int j=n; j >= 0; j--) {
        beta[j] /= H[j][j] ;

        for (int k = j-1; k >= 0; k--)
            beta[k] -= H[j][k] * beta[j] ;
    }

    // Set x = x0 + Q * y
    cublasDscal(_handle_cublas, A.rows, &zero, r.get(), 1) ;

    for (int j=0; j <= n; j++) 
        cublasDaxpy(_handle_cublas, r.rows, &beta[j], Q[j].get(), 1, r.get(), 1) ;

    precond.solve(r, r) ;
    cublasDaxpy(_handle_cublas, r.rows, &one, r.get(), 1, x0.get(), 1) ;     

    GMRES_Solver::_gmres_result result ;

    result.success = success ;
    result.iterations = n+1 ;
    result.error = normr / normrhs ;

    return result ;
}


/* GMRES_Solver::arnoldi() 
*
* Find the next orthonormal vector via modified-Gram-Schmidt iteration. 
*/
void GMRES_Solver::arnoldi(
    const CSR_SpMatrix& A, std::vector<DnVec>& Q, 
    std::vector<double>& h, int k, Precond& precond, char* buffer) const {

    Q.emplace_back(A.rows) ;
    if ((int) Q.size() != k+3)
        throw std::runtime_error("Bad size for Q matrix");

    if ((int) h.size() != k+2)
        throw std::runtime_error("Bad size for h vector") ;

    precond.solve(Q[k], Q[k+2]) ;    

    // Q[k+1] = A*Q[k]
    double zero = 0, one = 1 ;
    cusparseSpMV(_handle_cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &one, A.descr, Q[k+2].descr, &zero, Q[k+1].descr, 
        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1, buffer) ;
    

    // Orthogonalize
    double tmp ;
    for (int i=0; i <= k; i++) {
        cublasDdot(_handle_cublas, A.rows, Q[i].get(), 1, Q[k+1].get(), 1, &h[i]);
        tmp = -h[i] ; 
        cublasDaxpy(_handle_cublas, A.rows, &tmp, Q[i].get(), 1, Q[k+1].get(), 1) ;
    }

    // Normalize
    cublasDnrm2(_handle_cublas, A.rows, Q[k+1].get(), 1, &h[k+1]);
    tmp = 1/h[k+1] ;
    cublasDscal(_handle_cublas, A.rows, &tmp, Q[k+1].get(), 1) ;
}

/* GMRES_Solver::minimize_residual() 
*
* Solve the residual minimization problem.
*/
double GMRES_Solver::minimize_residual(int k, std::vector<double>& h, 
    std::vector<double>& cn, std::vector<double>& sn, std::vector<double>& beta) const {

    // Eliminate last element via Given's rotation
    for (int i=0; i < k; i++) {
        double 
            tmp =  cn[i] * h[i] + sn[i] * h[i+1] ;
        h[i+1]  = -sn[i] * h[i] + cn[i] * h[i+1] ;
        h[i] = tmp ;
    }

    double t = std::sqrt(h[k]*h[k] + h[k+1]*h[k+1]) ;
    cn[k] = h[k] / t ;
    sn[k] = h[k+1] / t ;

    h[k] = cn[k] * h[k] + sn[k] * h[k+1] ;
    h[k+1] = 0 ;

    // Compute the error
    beta[k+1] = -sn[k] * beta[k] ;
    beta[k] = cn[k] * beta[k] ;

    return std::abs(beta[k+1]) ;

}
