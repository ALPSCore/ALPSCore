/*
 * 
 * Copyright (c) Toon Knapen & Kresimir Fresl & Matthias Troyer 2003
 *
 * Permission to copy, modify, use and distribute this software 
 * for any non-commercial or commercial purpose is granted provided 
 * that this license appear on all copies of the software source code.
 *
 * Authors assume no responsibility whatsoever for its use and makes 
 * no guarantees about its quality, correctness or reliability.
 *
 * KF acknowledges the support of the Faculty of Civil Engineering, 
 * University of Zagreb, Croatia.
 *
 */

#ifndef ALPS_NUMERIC_BINDINGS_LAPACK_LAPACK_NAMES_H
#define ALPS_NUMERIC_BINDINGS_LAPACK_LAPACK_NAMES_H

#include <boost/numeric/bindings/traits/fortran.h>

/********************************************/
/* eigenproblems */ 

/* symmetric/Hermitian positive definite */

#define LAPACK_SSYEV FORTRAN_ID( ssyev )
#define LAPACK_DSYEV FORTRAN_ID( dsyev )

#define LAPACK_CHEEV FORTRAN_ID ( cheev )
#define LAPACK_ZHEEV FORTRAN_ID ( zheev )

#endif 

