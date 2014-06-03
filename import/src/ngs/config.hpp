/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_NGS_CONFIG_HPP
#define ALPS_NGS_CONFIG_HPP

#include <alps/config.h>

// if defined, no threading libraries are included
// #define ALPS_NGS_SINGLE_THREAD

// do not throw an error on accessing a not existing paht in a hdf5 file
// #define ALPS_HDF5_READ_GREEDY

// do not throw an error if closing a hdf5 gets dirty (e.g in Python)
// #define ALPS_HDF5_CLOSE_GREEDY

// blocksize in compressed hdf5. Default: 32
#ifndef ALPS_HDF5_SZIP_BLOCK_SIZE
    #define ALPS_HDF5_SZIP_BLOCK_SIZE 32
#endif

// maximal number of stack frames displayed in stacktrace. Default 63
#ifndef ALPS_NGS_MAX_FRAMES
    #define ALPS_NGS_MAX_FRAMES 63
#endif

// prevent the signal object from registering signals
#ifdef BOOST_MSVC
    #define ALPS_NGS_NO_SIGNALS
#endif

// do not print a stacktrace in error messages
#ifndef __GNUC__
    #define ALPS_NGS_NO_STACKTRACE
#endif

// TODO: have_python
// TODO: have_mpi
// TODO: have_thread

#endif

