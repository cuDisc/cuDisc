#ifndef _CUDISC_FLAGS_H_
#define _CUDISC_FLAGS_H_

// Flags to specify open or closed boundaries.
enum BoundaryFlags {
    all_closed = 0,
    open_R_inner = 1 << 0,
    open_R_outer = 1 << 1,
    open_Z_inner = 1 << 2,
    open_Z_outer = 1 << 3,

    // INFLOW BOUNDARIES NEED ADDING

    // zero_R_outer = 1 << 4,
    // zero_Z_outer = 1 << 5,
    // zero_R_inner = 1 << 6,
    // zero_Z_inner = 1 << 7
} ;

#endif//_CUDISC_FLAGS_H_