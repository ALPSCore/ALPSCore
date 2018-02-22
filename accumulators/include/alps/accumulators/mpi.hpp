/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#pragma once

#include <alps/config.hpp>

#ifdef ALPS_HAVE_MPI

    #include <alps/utilities/boost_mpi.hpp>

    namespace alps {
        namespace alps_mpi {

            template<typename T, typename Op> void reduce(const alps::mpi::communicator & comm, T const & in_values, Op op, int root);
            template<typename T, typename Op> void reduce(const alps::mpi::communicator & comm, T const & in_values, T & out_values, Op op, int root);

        } // alps_mpi::
    } // alps::

#endif
