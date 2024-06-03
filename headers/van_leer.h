#ifndef _CUDISC_HEADERS_VAN_LEER_H_
#define _CUDISC_HEADERS_VAN_LEER_H_

__device__ __host__ 
inline double vanleer_slope(double dQF, double dQB, double cF, double cB) {

    if (dQF*dQB > 0.) {
        double v = dQB/dQF ;
        return dQB * (cF*v + cB) / (v*v + (cF + cB - 2)*v + 1.) ;
    } 
    else {
        return 0. ;
    }
}

template<typename T>
__device__ __host__
double vl_R(GridRef& g, Field3DConstRef<T>& Qty, int i, int j, int k, int qind) {

    double Rc = g.Rc(i);

    double cF = (g.Rc(i+1) - Rc) / (g.Re(i+1)-Rc) ;
    double cB = (g.Rc(i-1) - Rc) / (g.Re(i)-Rc) ;

    double dQF = (Qty(i+1, j, k)[qind] - Qty(i, j, k)[qind]) / (g.Rc(i+1) - Rc) ;
    double dQB = (Qty(i-1, j, k)[qind] - Qty(i, j, k)[qind]) / (g.Rc(i-1) - Rc) ;

    return vanleer_slope(dQF, dQB, cF, cB) ;
}

template<typename T>
__device__ __host__
double vl_Z(GridRef& g, Field3DConstRef<T>& Qty, int i, int j, int k, int qind) {

    double Zc = g.Zc(i,j);

    double cF = (g.Zc(i,j+1) - Zc) / (g.Ze(i,j+1)-Zc) ;
    double cB = (g.Zc(i,j-1) - Zc) / (g.Ze(i,j)-Zc) ;

    double dQF = (Qty(i, j+1, k)[qind] - Qty(i, j, k)[qind]) / (g.Zc(i,j+1) - Zc) ;
    double dQB = (Qty(i, j-1, k)[qind] - Qty(i, j, k)[qind]) / (g.Zc(i,j-1) - Zc) ;

    return vanleer_slope(dQF, dQB, cF, cB) ;
}


#endif//_CUDISC_HEADERS_VAN_LEER_H_