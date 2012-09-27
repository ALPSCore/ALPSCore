/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2011 by Lukas Gamper <gamperl@gmail.com>                   *
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

#ifndef ALPS_NGS_SCHEDULER_MPIPARALLELIM_HPP
#define ALPS_NGS_SCHEDULER_MPIPARALLELIM_HPP

#include <alps/ngs/stacktrace.hpp>

#include <boost/mpi.hpp>

#include <stdexcept>

namespace alps {

    #ifdef ALPS_HAVE_MPI

        template<typename Impl> class mpiparallelsim : public Impl {
            public:
                using Impl::collect_results;
                
                mpiparallelsim(typename alps::parameters_type<Impl>::type const & p) {
                    throw std::runtime_error("No communicator passed" + ALPS_STACKTRACE);
                }

                mpiparallelsim(typename alps::parameters_type<Impl>::type const & p, boost::mpi::communicator const & c) 
                    : Impl(p, c)
                    , communicator(c)
                {
                    MPI_Errhandler_set(communicator, MPI_ERRORS_RETURN);
                }

                double fraction_completed() const {
                    return Impl::fraction_completed();
                }

                typename results_type<Impl>::type collect_results(typename result_names_type<Impl>::type const & names) const {
                    return Impl::collect_results(names);
                }

            private:
                boost::mpi::communicator communicator;
        };

    #endif
}

#endif
