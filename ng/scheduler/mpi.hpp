/*****************************************************************************
 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations
 *
 * ALPS Libraries
 *
 * Copyright (C) 2010 by Lukas Gamper <gamperl@gmail.com>
 *                       Matthias Troyer <troyer@comp-phys.org>
 *
 * This software is part of the ALPS libraries, published under the ALPS
 * Library License; you can use, redistribute it and/or modify it under
 * the terms of the license, either version 1 or (at your option) any later
 * version.
 * 
 * You should have received a copy of the ALPS Library License along with
 * the ALPS Libraries; see the file LICENSE.txt. If not, the license is also
 * available from http://alps.comp-phys.org/.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT 
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE 
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, 
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
 * DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/

#include <alps/hdf5.hpp>
#include <alps/ng/api.hpp>
#include <alps/parameter.h>
#include <alps/ng/boost.hpp>
#include <alps/ng/signal.hpp>
#include <alps/ng/parameters.hpp>
#include <alps/ng/observables/base.hpp>
#include <alps/ng/observables/evaluator.hpp>

#include <boost/mpi.hpp>
#include <boost/bind.hpp>
#include <boost/utility.hpp>
#include <boost/variant.hpp>
#include <boost/function.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/contains.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/program_options.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>

#include <map>
#include <vector>
#include <iomanip>
#include <sstream>
#include <cstring>
#include <iomanip>
#include <signal.h>
#include <algorithm>

#ifndef ALPS_NG_MPI_SIMULATION_HPP
#define ALPS_NG_MPI_SIMULATION_HPP

namespace alps {
    namespace ng {
        namespace scheduler {

            // TODO: rename and move in scheduler namespace
            template<typename Impl> class mpi : public Impl {
                public:
                    using Impl::collect_results;
                    mpi(typename parameters_type<Impl>::type const & p, boost::mpi::communicator const & c) 
                        : Impl(p, c.rank())
                        , communicator(c)
                        , binnumber(p.value_or_default("binnumber", std::min(128, 2 * c.size())))
                    {}

                    double fraction_completed() {
                        return boost::mpi::all_reduce(communicator, Impl::fraction_completed(), std::plus<double>());
                    }

                    virtual typename results_type<Impl>::type collect_results(typename result_names_type<Impl>::type const & names) const {
                        typename results_type<Impl>::type local_results = Impl::collect_results(names), partial_results;
                        // TODO: implement
/*                        for(typename results_type<Impl>::type::iterator it = local_results.begin(); it != local_results.end(); ++it)
                            if (it->second->count() > 0 && communicator.rank() == 0)
                                it->second->reduce_master(partial_results, it->first, communicator, binnumber);
                            else if (it->second->count() > 0)
                                it->second->reduce_slave(communicator, binnumber);
*/                        return partial_results;
                    }

                private:
                    boost::mpi::communicator communicator;
                    std::size_t binnumber;
            };

        }
    }
}

#endif
