/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#pragma once

#include <alps/config.hpp>

#ifdef ALPS_HAVE_MPI

    #include "alps/utilities/mpi.hpp"
    #include <vector>

    namespace alps {
        namespace mpi {

            ///reduce a std::vector<T,A> (T is the type, A the allocator) with operation (type OP) op, using a communicator comm to the root root, and overwrite the vector in_values
            template<typename T, typename A, typename Op> void reduce(const communicator & comm, std::vector<T, A> const & in_values, Op op, int root) {
                reduce(comm, &in_values.front(), in_values.size(), op, root);
            }

            ///reduce a std::vector<T,A> (T is the type, A the allocator) with operation (type OP) op, using a communicator comm to the root root, and write the result into out_values
            template<typename T, typename A, typename Op> void reduce(const communicator & comm, std::vector<T, A> const & in_values, std::vector<T, A> & out_values, Op op, int root) {
                out_values.resize(in_values.size());
                reduce(comm, &in_values.front(), in_values.size(), &out_values.front(), op, root);
            }
        }
    }

#endif

