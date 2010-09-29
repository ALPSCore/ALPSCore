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
#include <alps/ng/alea.hpp>
#include <alps/parameter.h>
#include <alps/ng/boost.hpp>
#include <alps/ng/signal.hpp>
#include <alps/ng/parameters.hpp>

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
#include <boost/ptr_container/ptr_map.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/assign/ptr_map_inserter.hpp>
#include <boost/random/variate_generator.hpp>

#include <map>
#include <vector>
#include <iomanip>
#include <sstream>
#include <cstring>
#include <iomanip>
#include <signal.h>
#include <algorithm>

#ifndef ALPS_NG_SINGLE_SIMULATION_HPP
#define ALPS_NG_SINGLE_SIMULATION_HPP

namespace alps {
    namespace ng {

        class singe_simulation {
            public:
                typedef parameters parameters_type;
                typedef boost::ptr_map<std::string, mcany> results_type;
                typedef std::vector<std::string> result_names_type;

                singe_simulation(parameters_type const & p, std::size_t seed_offset = 0)
                    : params(p)
                    , next_check(8)
                    , start_time(boost::posix_time::second_clock::local_time())
                    , check_time(boost::posix_time::second_clock::local_time() + boost::posix_time::seconds(next_check))
// TODO: this ist not the best solution
                    , random(boost::mt19937(static_cast<std::size_t>(p.value_or_default("SEED", 42)) + seed_offset), boost::uniform_real<>())
                {}

                virtual void do_update() = 0;

                virtual void do_measurements() = 0;

                virtual double fraction_completed() const = 0;

                void save(boost::filesystem::path const & path) const {
                    boost::filesystem::path original = path.parent_path() / (path.filename() + ".h5");
                    boost::filesystem::path backup = path.parent_path() / (path.filename() + ".bak");
                    if (boost::filesystem::exists(backup))
                        boost::filesystem::remove(backup);
                    {
                        hdf5::oarchive ar(backup.file_string());
                        ar 
                            << make_pvp("/parameters", params)
                            << make_pvp("/simulation/realizations/0/clones/0/results", results);
                    }
                    if (boost::filesystem::exists(original))
                        boost::filesystem::remove(original);
                    boost::filesystem::rename(backup, original);
                }

                void load(boost::filesystem::path const & path) {
                    hdf5::iarchive ar(path.file_string() + ".h5");
                    ar >> make_pvp("/simulation/realizations/0/clones/0/results", results);
                }
                // free function save_results(path,collected_results); or similar
                void save_collected(boost::filesystem::path const & path) {
                    results_type collected_results = collect_results();
                    if (collected_results.size()) {
                        boost::filesystem::path original = path.parent_path() / (path.filename() + ".h5");
                        boost::filesystem::path backup = path.parent_path() / (path.filename() + ".bak");
                        if (boost::filesystem::exists(backup))
                            boost::filesystem::remove(backup);
                        {
                            hdf5::oarchive ar(backup.file_string());
                            ar << make_pvp("/parameters", params);
                            for (results_type::const_iterator it = collected_results.begin(); it != collected_results.end(); ++it)
                                if (it->second->count() > 0) {
                                    using namespace ::alps;
                                    ar << make_pvp("/simulation/results/" + it->first, *(it->second));
                                }
                        }
                        if (boost::filesystem::exists(original))
                            boost::filesystem::remove(original);
                        boost::filesystem::rename(backup, original);
                    }
                }

                bool run(boost::function<bool ()> const & stop_callback) {
                    double fraction = 0.;
                    do {
                        do_update();
                        do_measurements();
                        if (boost::posix_time::second_clock::local_time() > check_time) {
                            fraction = fraction_completed();
                            next_check = std::min(
                                2. * next_check, 
                                std::max(
                                    double(next_check), 
                                    0.8 * (boost::posix_time::second_clock::local_time() - start_time).total_seconds() / fraction * (1 - fraction)
                                )
                            );
                            check_time = boost::posix_time::second_clock::local_time() + boost::posix_time::seconds(next_check);
                        }
                    } while(!stop_callback() && fraction < 1.);
                    return !(fraction < 1.);
                }

                result_names_type result_names() const {
                    result_names_type names;
                    for(ObservableSet::const_iterator it = results.begin(); it != results.end(); ++it)
                        names.push_back(it->first);
                    return names;
                }

                result_names_type unsaved_result_names() const {
                    return result_names_type(); 
                }

                results_type collect_results() const {
                    return collect_results(result_names());
                }

                virtual results_type collect_results(result_names_type const & names) const {
                    results_type partial_results;
                    for(result_names_type::const_iterator it = names.begin(); it != names.end(); ++it)
                        if (dynamic_cast<AbstractSimpleObservable<double> const *>(&results[*it]) != NULL)
                            boost::assign::ptr_map_insert<mcdata<double> >(partial_results)(
                                *it, dynamic_cast<AbstractSimpleObservable<double> const &>(results[*it])
                            );
                        else if (dynamic_cast<AbstractSimpleObservable<std::valarray<double> > const *>(&results[*it]) != NULL)
                            boost::assign::ptr_map_insert<mcdata<std::vector<double> > >(partial_results)(
                                *it, dynamic_cast<AbstractSimpleObservable<std::valarray<double> > const &>(results[*it])
                            );
                        else
                            throw std::runtime_error("unknown observable type");
                    return partial_results;
                }

            protected:

                parameters_type params;
                ObservableSet results;
                boost::variate_generator<boost::mt19937, boost::uniform_real<> > random;

            private:

                std::size_t next_check;
                boost::posix_time::ptime start_time;
                boost::posix_time::ptime check_time;
        };

    }
}

#endif
