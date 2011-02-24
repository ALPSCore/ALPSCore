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

#ifndef ALPS_NGS_MCMPISIM_HPP
#define ALPS_NGS_MCMPISIM_HPP

namespace alps {

    #ifdef ALPS_HAVE_MPI

        template<typename Impl> class mcmpisim : public Impl {
            public:
                using Impl::collect_results;
                mcmpisim(typename parameters_type<Impl>::type const & p, boost::mpi::communicator const & c) 
                    : Impl(p, c.rank())
                    , communicator(c)
                    , binnumber(p.value_or_default("binnumber", std::min(128, 2 * c.size())))
                {
                    MPI_Errhandler_set(communicator, MPI_ERRORS_RETURN);
                }

                double fraction_completed() {
                    return boost::mpi::all_reduce(communicator, Impl::fraction_completed(), std::plus<double>());
                }

                virtual typename results_type<Impl>::type collect_results(typename result_names_type<Impl>::type const & names) const {
                    typename results_type<Impl>::type local_results = Impl::collect_results(names), partial_results;
                    for(typename results_type<Impl>::type::iterator it = local_results.begin(); it != local_results.end(); ++it)
                        if (it->second.count())
                            partial_results.insert(it->first, it->second.reduce(communicator, binnumber));
                        else
                            partial_results.insert(it->first, it->second);
                    return partial_results;
                }

            private:
                boost::mpi::communicator communicator;
                std::size_t binnumber;
        };

    #endif
}

#endif
