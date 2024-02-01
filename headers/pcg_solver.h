
#ifndef _CUSDISC_PCG_SOLVER_H_
#define _CUSDISC_PCG_SOLVER_H_

#include <vector>

#include <cublas_v2.h>
#include <cusparse.h>

#include "field.h"
#include "grid.h"
#include "star.h"

#include "matrix_types.h"
#include "matrix_utils.h"

class Precond {
  public:

   virtual void solve(const DnVec& rhs, DnVec& x) = 0; 
   virtual ~Precond() {} ;
} ;

class NoPrecond : public Precond {
  public:
    NoPrecond() ;
    ~NoPrecond();
    virtual void solve(const DnVec& rhs, DnVec& x); 
  
  private:
    cublasHandle_t _handle_cublas ;
} ;

/* class Jacobi_Precond
 *
 * Scales the matrix system to create a unit diagonal.
 */
class Jacobi_Precond {
  public:
    Jacobi_Precond(const CSR_SpMatrix& A, int block_size=1) ;

    void transform(CSR_SpMatrix& A, DnVec& x, DnVec& b) const ;
    void transform_guess(DnVec& x) const ;
    void invert(DnVec& x_tilde) const ;
  private:
    DnVec _diag ;
} ;

/* class ILU_precond
 * 
 * Wrapper class for incomplete LU preconditioning
 */
class ILU_precond : public Precond
{
    CSR_SpLUfac LUfac;
    DnVec tmp;

    cusparseHandle_t _handle_cusparse ;
    cublasHandle_t _handle_cublas ;

    SpSVDescr solveL, solveU ;  
    CudaArray<char> bufferL, bufferU ;

    static const cusparseSolvePolicy_t policyFac = CUSPARSE_SOLVE_POLICY_USE_LEVEL;

  public:
    ILU_precond(const CSR_SpMatrix& mat, int k=0) ;
    
    ~ILU_precond() {
        cusparseDestroy(_handle_cusparse) ;
        cublasDestroy(_handle_cublas) ;
    }
    
    void setup(const DnVec& rhs, DnVec& x) ; 
    void solve(const DnVec& rhs, DnVec& x) ; 

    private:
        CSR_SpMatrix get_ILUk_shape(const CSR_SpMatrix& mat, int k) ;
} ;

/* class BlockJacobi_precond
 * 
 * A simplified block-jacobi pre-conditioner for matrices with the specific
 * form of the FLD equations.
 */
class BlockJacobi_precond : public Precond 
{
  public:
    BlockJacobi_precond(const CSR_SpMatrix& A, int block_size=1) ;

    ~BlockJacobi_precond() {
        cublasDestroy(_handle_cublas) ;
        cusparseDestroy(_handle_cusparse) ;
    }

    void solve(const DnVec& rhs, DnVec& x) ; 

  private:
    cublasHandle_t _handle_cublas ;
    cusparseHandle_t _handle_cusparse ;
    
    const CSR_SpMatrix& A ;
    DnVec tmp ;
    CudaArray<char> buffer ;
    int _block_size ;
} ;

class CheckConvergence {
  public:
   virtual bool operator()(const DnVec& x, const DnVec& residual, const DnVec& rhs) const = 0 ;
   virtual ~CheckConvergence() {} ;
} ;


bool check_tolerance_scaled(const DnVec& x, const DnVec& residual, const DnVec& rhs, 
                            double tol) ;

class CheckResidual: public CheckConvergence {
  public:
    CheckResidual(double tol)
      : _tol(tol), err(1)
    { } ;

    bool operator()(const DnVec& x, const DnVec& residual, const DnVec& rhs) const ;

  private:
    double _tol ;
    mutable DnVec err ;
} ;


class CheckTemperatureResidual: public CheckConvergence {
  public:
    CheckTemperatureResidual(double tol, int num_bands=1, double tol_J=0.1)
      : _tol(tol), _tolJ(tol_J), _num_vars(num_bands+1), err(1)
    { } ;

    bool operator()(const DnVec& x, const DnVec& residual, const DnVec& rhs) const ;

  private:
    double _tol, _tolJ ;
    int _num_vars ;
    mutable DnVec err ;
} ;



/* class PCG_Solver
 *
 * Incomplete-LU preconditioned Conjugate Gradient Solver for
 * large sparse matrices.
 */
class PCG_Solver {
  public:
    PCG_Solver(double tol=1e-7, int max_iter=1000)
      : _check_convergence(new CheckResidual(tol)),
        _max_iter(max_iter)
    { 
        _init() ;
    } ;

    PCG_Solver(std::unique_ptr<CheckConvergence> convergence_test, 
               int max_iter=1000)
      : _check_convergence(std::move(convergence_test)),
        _max_iter(max_iter)
    { 
        _init() ;
    } ;

    ~PCG_Solver() {
        cublasDestroy(_handle_cublas) ;
        cusparseDestroy(_handle_cusparse) ;
    }


    bool operator()(const CSR_SpMatrix& mat, const DnVec& rhs, DnVec& result) ;    
    bool operator()(const CSR_SpMatrix& mat, const DnVec& rhs, DnVec& result, Precond&) ;

    bool solve_non_symmetric(const CSR_SpMatrix& mat, const DnVec& rhs, DnVec& result) ;
    bool solve_non_symmetric(const CSR_SpMatrix& mat, const DnVec& rhs, DnVec& result, Precond&) ;

  private:
    cublasHandle_t _handle_cublas ;
    cusparseHandle_t _handle_cusparse ;

    void _init() {
        cublasStatus_t status_cub = 
            cublasCreate(&_handle_cublas) ;
        if (status_cub != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("Failed to initialize CUBLAS") ;

        cusparseStatus_t status_cus =
            cusparseCreate(&_handle_cusparse) ;
         if (status_cus != CUSPARSE_STATUS_SUCCESS)
            throw std::runtime_error("Failed to initialize CUSPARSE") ;
    }

    std::unique_ptr<CheckConvergence> _check_convergence ;
    int _max_iter ;
} ;



/* class GMRES_Solver
 *
 * Incomplete-LU preconditioned GMRES Solver for
 * large sparse matrices.
 */
class GMRES_Solver {
  public:
    GMRES_Solver(double tol=1e-7, int n_restart=100, int max_iters=10)
      : _tol(tol), _max_iters(max_iters), _n_restart(n_restart)
    { 
        cublasStatus_t status_cub = 
            cublasCreate(&_handle_cublas) ;
        if (status_cub != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("Failed to initialize CUBLAS") ;

        cusparseStatus_t status_cus =
            cusparseCreate(&_handle_cusparse) ;
         if (status_cus != CUSPARSE_STATUS_SUCCESS)
            throw std::runtime_error("Failed to initialize CUSPARSE") ;
    } ;

    ~GMRES_Solver() {
        cublasDestroy(_handle_cublas) ;
        cusparseDestroy(_handle_cusparse) ;
    }


    void operator()(const CSR_SpMatrix& mat, const DnVec& rhs, DnVec& result) ;

  private:
    struct _gmres_result {
        double error ;
        int iterations, success ;
    } ;
    

    _gmres_result gmres_loop(const CSR_SpMatrix& A, const DnVec& rhs, DnVec& result,
                             Precond& precond) const ; 

    void arnoldi(const CSR_SpMatrix& A, std::vector<DnVec>& Q, 
                 std::vector<double>& h, int k, 
                 Precond& precond, char* buffer) const ;

    double minimize_residual(int k, std::vector<double>& h, 
                             std::vector<double>& cn, std::vector<double>& sn,
                             std::vector<double>& beta) const ;

    cublasHandle_t _handle_cublas ;
    cusparseHandle_t _handle_cusparse ;

    double _tol ;
    int _max_iters, _n_restart ;
} ;

#endif// _CUSDISC_PCG_SOLVER_H_