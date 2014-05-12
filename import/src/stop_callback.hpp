/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2012 by Lukas Gamper <gamperl@gmail.com>,                  *
 *                              Synge Todo <wistaria@comp-phys.org>                *
 *                                                                                 *
 * This software is part of the ALPS libraries, published under the ALPS           *
 * Library License; you can use, redistribute it and/or modify it under            *
 * the terms of the license, either version 1 or (at your option) any later        *
 * version.                                                                        *
 *                                                                                 *
 * You should have received a copy of the ALPS Library License along with          *
 * the ALPS Libraries; see the file LICENSE.txt. If not, the license is also       *
 * available from http://alps.comp-phys.org/.                                      *
 *                                                                                 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        *
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT       *
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE       *
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,     *
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER     *
 * DEALINGS IN THE SOFTWARE.                                                       *
 *                                                                                 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

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
