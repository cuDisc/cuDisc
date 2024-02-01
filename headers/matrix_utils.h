
#ifndef _CUDISC_HEADERS_MATRIX_UTILS_H
#define _CUDISC_HEADERS_MATRIX_UTILS_H

#include <stdexcept>

#include <cusparse.h>
#include <cublas_v2.h>

/* class CholFacInfo
 *
 * RAII wrapper for cusparse info object
 */
class CholFacInfo {
    CholFacInfo(CholFacInfo&) = delete ;
    CholFacInfo(CholFacInfo&&) = delete ;
    CholFacInfo& operator=(CholFacInfo&) = delete ;
    CholFacInfo& operator=(CholFacInfo&&) = delete ;
  public:
    CholFacInfo() {
        cusparseCreateCsric02Info(&_info) ;
    }

    ~CholFacInfo() {
        cusparseDestroyCsric02Info(_info) ;
    }

    operator csric02Info_t() const {
        return _info ;
    }
  private:
    csric02Info_t _info ;
} ;

/* class LUFacInfo
 *
 * RAII wrapper for cusparse info object
 */
class LUFacInfo {
    LUFacInfo(LUFacInfo&) = delete ;
    LUFacInfo(LUFacInfo&&) = delete ;
    LUFacInfo& operator=(LUFacInfo&) = delete ;
    LUFacInfo& operator=(LUFacInfo&&) = delete ;
  public:
    LUFacInfo() {
        cusparseCreateCsrilu02Info(&_info) ;
    }

    ~LUFacInfo() {
        cusparseDestroyCsrilu02Info(_info) ;
    }

    operator csrilu02Info_t() const {
        return _info ;
    }
  private:
    csrilu02Info_t _info ;
} ;

class MatDescr {
    MatDescr(MatDescr&) = delete ;
    MatDescr(MatDescr&&) = delete ;
    MatDescr& operator=(MatDescr&) = delete ;
    MatDescr& operator=(MatDescr&&) = delete ;
  public:
    MatDescr() {
        cusparseStatus_t status = cusparseCreateMatDescr(&descr) ;
        
        if (status != CUSPARSE_STATUS_SUCCESS)
            throw std::runtime_error(
                "MatDescr: Failed to create matrix descriptor") ;

        cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
        cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    }
    ~MatDescr() {
        cusparseDestroyMatDescr(descr) ;
    }

    void set_FillMode(cusparseFillMode_t fill_mode) {
        cusparseSetMatFillMode(descr, fill_mode);
    }
    void set_DiagType(cusparseDiagType_t diag_type) {
        cusparseSetMatDiagType(descr, diag_type) ;
    }

    operator cusparseMatDescr_t() {
        return descr ;
    } ;
  private:
    cusparseMatDescr_t descr ;
} ;


class SpGEMMDescr  {
    SpGEMMDescr(SpGEMMDescr&) = delete ;
    SpGEMMDescr(SpGEMMDescr&&) = delete ;
    SpGEMMDescr& operator=(SpGEMMDescr&) = delete ;
    SpGEMMDescr& operator=(SpGEMMDescr&&) = delete ;
  public:
    SpGEMMDescr() {
        cusparseStatus_t status = cusparseSpGEMM_createDescr(&descr) ;
        
        if (status != CUSPARSE_STATUS_SUCCESS)
            throw std::runtime_error(
                "SpGEMMInfo: Failed to create csrgemm2Info_t") ;
    }
    ~SpGEMMDescr() {
        cusparseSpGEMM_destroyDescr(descr) ;
    }

    operator cusparseSpGEMMDescr_t () {
        return descr ;
    } ;
  private:
    cusparseSpGEMMDescr_t  descr ;
} ;

class SpSVDescr {
    SpSVDescr(SpSVDescr&) = delete ;
    SpSVDescr(SpSVDescr&&) = delete ;
    SpSVDescr& operator=(SpSVDescr&) = delete ;
    SpSVDescr& operator=(SpSVDescr&&) = delete ;
  public:
    SpSVDescr() {
        cusparseStatus_t status = cusparseSpSV_createDescr(&descr) ;
        
        if (status != CUSPARSE_STATUS_SUCCESS)
            throw std::runtime_error(
                "SpGEMMInfo: Failed to create csrgemm2Info_t") ;
    }
    ~SpSVDescr() {
        cusparseSpSV_destroyDescr(descr) ;
    }

    operator cusparseSpSVDescr_t () {
        return descr ;
    } ;

  private:
    cusparseSpSVDescr_t  descr ;
} ;


#endif//_CUDISC_HEADERS_MATRIX_UTILS_H