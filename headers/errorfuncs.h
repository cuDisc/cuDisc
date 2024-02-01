#ifndef _CUDISC_HEADERS_ERRORFUNCS_H_
#define _CUDISC_HEADERS_ERRORFUNCS_H_

#include "grid.h"
#include "field.h"
#include <cmath>

double L2err(Grid& g, Field<double> &T1, Field<double> &T2) {

    double sum=0;
    int N = (g.NR + 2*g.Nghost)*(g.Nphi + 2*g.Nghost);

    for (int i=0; i<g.NR + 2*g.Nghost; i++) {
        for (int j=0; j<g.Nphi + 2*g.Nghost; j++) {
            sum += (T2(i,j) - T1(i,j))*(T2(i,j) - T1(i,j));
        }
    }

    return std::sqrt(sum/N);

}

double fracerr(Grid& g, Field<double> &T1, Field<double> &T2) {

    double sum=0;
    int N = (g.NR + 2*g.Nghost)*(g.Nphi + 2*g.Nghost);

    for (int i=0; i<g.NR + 2*g.Nghost; i++) {
        for (int j=0; j<g.Nphi + 2*g.Nghost; j++) {
            sum += std::abs(T2(i,j) - T1(i,j))/T1(i,j);
        }
    }

    return sum/N;

}

#endif//_CUDISC_HEADERS_ERRORFUNCS_H_
