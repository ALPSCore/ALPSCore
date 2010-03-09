/* 
 * Copyright Matthias Troyer 2005
 * Distributed under the Boost Software License, Version 1.0.) (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
*
 */

#include <boost/preprocessor/cat.hpp>

#ifndef ALPS_SPRNG_CALL
#error Do not include this file directly
#else

// This header file declares the C API functions for the SPRNG generators

extern "C" {

int ALPS_SPRNG_CALL(get_rn_int) (int *igenptr);
float  ALPS_SPRNG_CALL(get_rn_flt) (int *igenptr);
double  ALPS_SPRNG_CALL(get_rn_dbl) (int *igenptr);
int *  ALPS_SPRNG_CALL(init_rng) (int rng_type,  int gennum, int total_gen,  int seed, int mult);
int  ALPS_SPRNG_CALL(spawn_rng) (int *igenptr, int nspawned, int ***newgens, int checkid);
int  ALPS_SPRNG_CALL(get_seed_rng) (int *genptr);
int  ALPS_SPRNG_CALL(free_rng) (int *genptr);
int  ALPS_SPRNG_CALL(pack_rng) ( int *genptr, char **buffer);
int * ALPS_SPRNG_CALL(unpack_rng) ( char const *packed);
int  ALPS_SPRNG_CALL(print_rng) ( int *igen);

}

#endif 
