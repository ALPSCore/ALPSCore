/*
 * 
 * Copyright (c) Toon Knapen, Kresimir Fresl, Matthias Troyer, & Synge Todo 2003,2004
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

#ifndef ALPS_BINDINGS_LAPACK_NAMES_H
#define ALPS_BINDINGS_LAPACK_NAMES_H

#include <boost/numeric/bindings/traits/fortran.h>
#include <boost/numeric/bindings/lapack/lapack_names.h>

/********************************************/
/* eigenproblems */ 

/* symmetric/Hermitian positive definite */

#ifndef LAPACK_SSYEV
# define LAPACK_SSYEV FORTRAN_ID( ssyev )
#endif
#ifndef LAPACK_DSYEV
# define LAPACK_DSYEV FORTRAN_ID( dsyev )
#endif

#ifndef LAPACK_CHEEV
# define LAPACK_CHEEV FORTRAN_ID( cheev )
#endif
#ifndef LAPACK_ZHEEV
# define LAPACK_ZHEEV FORTRAN_ID( zheev )
#endif

#endif 
