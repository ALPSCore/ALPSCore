/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_NGS_CALLBACK_HPP
#define ALPS_NGS_CALLBACK_HPP

#include <alps/ngs/config.hpp>
#include <alps/ngs/signal.hpp>

#include <boost/chrono.hpp>
#ifdef ALPS_HAVE_MPI
# include <boost/mpi/communicator.hpp>
#endif

namespace alps {

	class ALPS_DECL stop_callback {
		public:
		    stop_callback(std::size_t timelimit);
#ifdef ALPS_HAVE_MPI
			stop_callback(boost::mpi::communicator const & cm, std::size_t timelimit);
#endif
		    bool operator()();
		private:
		    boost::chrono::duration<std::size_t> limit;
		    alps::ngs::signal signals;
		    boost::chrono::high_resolution_clock::time_point start;
#ifdef ALPS_HAVE_MPI
	        boost::optional<boost::mpi::communicator> comm;
#endif
	};

#ifdef ALPS_HAVE_MPI
		// TODO: remove this!
        class ALPS_DECL stop_callback_mpi {
        public:
          stop_callback_mpi(boost::mpi::communicator const& cm, std::size_t timelimit);
          bool operator()();
        private:
          boost::mpi::communicator comm;
          boost::chrono::duration<std::size_t> limit;
          alps::ngs::signal signals;
          boost::chrono::high_resolution_clock::time_point start;
	};
#endif
}

#endif
