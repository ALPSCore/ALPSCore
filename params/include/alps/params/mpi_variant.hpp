/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file mpi_variant.hpp
    
    @brief Header for object-oriented interface to MPI for boost::variant

    @todo FIXME: Move to utilities
*/

#ifndef ALPS_PARAMS_MPI_VARIANT_HPP_69d56023ac6048ca987f4be0f67122af
#define ALPS_PARAMS_MPI_VARIANT_HPP_69d56023ac6048ca987f4be0f67122af

#include "./serialize_variant.hpp"
#include <alps/utilities/mpi.hpp>

namespace alps {
    namespace mpi {

        /// MPI_BCast of an boost::variant over MPL type sequence MPLSEQ
        template <typename MPLSEQ>
        inline void broadcast(const communicator& comm, typename boost::make_variant_over<MPLSEQ>::type& var, int root)
        {
            // FIXME!!! Does nothing
        }

    } // mpi::
} // alps::

#endif /* ALPS_PARAMS_MPI_VARIANT_HPP_69d56023ac6048ca987f4be0f67122af */
