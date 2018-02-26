/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_HDF5_CONFIG_HPP
#define ALPS_HDF5_CONFIG_HPP

#include <alps/config.hpp>

// if defined, no threading libraries are included
// #define ALPS_SINGLE_THREAD

// do not throw an error on accessing a not existing paht in a hdf5 file
// #define ALPS_HDF5_READ_GREEDY

// do not throw an error if closing a hdf5 gets dirty (e.g in Python)
// #define ALPS_HDF5_CLOSE_GREEDY

// blocksize in compressed hdf5. Default: 32
#ifndef ALPS_HDF5_SZIP_BLOCK_SIZE
    #define ALPS_HDF5_SZIP_BLOCK_SIZE 32
#endif

#endif

