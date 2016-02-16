/*
 * Copyright (C) 1998-2016 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/mc/stop_callback.hpp>

#include <alps/utilities/signal.hpp>
#include <alps/utilities/boost_mpi.hpp>

namespace alps {

    stop_callback::stop_callback(std::size_t timelimit)
        : limit(timelimit)
        , start(boost::chrono::high_resolution_clock::now())
    {}

#ifdef ALPS_HAVE_MPI
    stop_callback::stop_callback(alps::mpi::communicator const & cm, std::size_t timelimit)
        : limit(timelimit), start(boost::chrono::high_resolution_clock::now()), comm(cm)
    {}
#endif

    bool stop_callback::operator()() {
#ifdef ALPS_HAVE_MPI
        if (comm) {
            bool to_stop;
            if (comm->rank() == 0)
                to_stop = !signals.empty() || (limit.count() > 0 && boost::chrono::high_resolution_clock::now() > start + limit);
            broadcast(*comm, to_stop, 0);
            return to_stop;
        } else
#endif
            return !signals.empty() || (limit.count() > 0 && boost::chrono::high_resolution_clock::now() > start + limit);
    }
}
