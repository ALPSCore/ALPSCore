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

#include <alps/alea.h>
#include <alps/hdf5.hpp>
#include <alps/ng/boost.hpp>
#include <alps/ng/short_print.hpp>

#include <boost/mpi.hpp>
#include <boost/utility.hpp>
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

#ifndef ALPS_NG_ALEA_HPP
#define ALPS_NG_ALEA_HPP

namespace alps {
    namespace ng {
    
        class mcany {
            public:
        
                virtual uint64_t count() const { 
                    throw std::logic_error("not Impl"); 
                }

                virtual void serialize(hdf5::iarchive & ar) { 
                    throw std::logic_error("not Impl"); 
                }

                virtual void serialize(hdf5::oarchive & ar) const { 
                    throw std::logic_error("not Impl"); 
                }

                virtual std::string to_string() const { 
                    throw std::logic_error("not Impl"); 
                }

                virtual void reduce_master(boost::ptr_map<std::string, mcany> &, std::string const &, boost::mpi::communicator const &, std::size_t binnumber) { 
                    throw std::logic_error("not Impl"); 
                }

                virtual void reduce_slave(boost::mpi::communicator const &, std::size_t binnumber) { 
                    throw std::logic_error("not Impl"); 
                }
        };

        inline std::ostream & operator<<(std::ostream & os, boost::ptr_map<std::string, mcany> const & results) {
            for (boost::ptr_map<std::string, mcany>::const_iterator it = results.begin(); it != results.end(); ++it)
                std::cout << std::fixed << std::setprecision(5) << it->first << ": " << it->second->to_string() << std::endl;
            return os;
        }

        template <typename T> class mcdata : public mcany, alea::mcdata<T> {
            public:
                typedef typename alea::mcdata<T>::element_type element_type;
                typedef typename alea::mcdata<T>::time_type time_type;
                typedef typename alea::mcdata<T>::value_type value_type;
                typedef typename alea::mcdata<T>::result_type result_type;

                mcdata(): alea::mcdata<T>() {}
                template <typename X> mcdata(mcdata<X> const & rhs): alea::mcdata<T>(rhs) {}
                template <typename X, typename S> mcdata(mcdata<X> const & rhs, S s): alea::mcdata<T>(rhs, s) {}
                template <typename X> mcdata(AbstractSimpleObservable<X> const & obs): alea::mcdata<T>(obs) {}

                mcdata(
                      int64_t count
                    , result_type const & mean
                    , result_type const & error
                    , boost::optional<result_type> const & variance
                    , boost::optional<typename alea::mcdata<T>::time_type> const & tau
                    , uint64_t binsize
                    , std::vector<value_type> const & values
                ): alea::mcdata<T>(count, mean, error, variance, tau, binsize, values) {}

                inline uint64_t count() const { 
                    return alea::mcdata<T>::count();
                }

                inline result_type const & mean() const {
                    return alea::mcdata<T>::mean();
                }

                inline result_type const & error() const {
                    return alea::mcdata<T>::error();
                }

                inline typename alea::mcdata<T>::time_type const & tau() const {
                    return alea::mcdata<T>::tau();
                }

                inline result_type const & variance() const {
                    return alea::mcdata<T>::variance();
                }

                typename std::string to_string() const {
                    if (count() == 0)
                        return "No Measurements";
                    else {
                        std::stringstream s;
                        s << std::fixed << std::setprecision(5) << short_print(alea::mcdata<T>::mean()) << "(" << count() << ") +/-" << short_print(alea::mcdata<T>::error()) << " "
                          << short_print(alea::mcdata<T>::bins()) << "#" << alea::mcdata<T>::bin_size();
                        return s.str();
                    }
                }

                void serialize(hdf5::iarchive & ar) { 
                    alea::mcdata<T>::serialize(ar);
                }

                void serialize(hdf5::oarchive & ar) const { 
                    alea::mcdata<T>::serialize(ar);
                }

                void reduce_master(
                      boost::ptr_map<std::string, mcany> & results
                    , std::string const & name
                    , boost::mpi::communicator const & communicator
                    , std::size_t binnumber
                ) {
                    reduce_master_impl(results, name, communicator, binnumber, typename boost::is_scalar<T>::type());
                }

                virtual void reduce_slave(boost::mpi::communicator const & communicator, std::size_t binnumber) { 
                    reduce_slave_impl(communicator, binnumber, typename boost::is_scalar<T>::type());
                }

            private:

                void reduce_master_impl(
                      boost::ptr_map<std::string, mcany> & results
                    , std::string const & name
                    , boost::mpi::communicator const & communicator
                    , std::size_t binnumber
                    , boost::mpl::true_
                ) {
                    using std::sqrt;
                    using alps::numeric::sq;
                    uint64_t global_count;
                    boost::mpi::reduce(communicator, count(), global_count, std::plus<uint64_t>(), 0);
                    std::vector<result_type> local(2, 0), global(alea::mcdata<T>::has_variance() ? 3 : 2, 0);
                    local[0] = alea::mcdata<T>::mean() * static_cast<element_type>(count());
                    local[1] = sq(alea::mcdata<T>::error()) * sq(static_cast<element_type>(count()));
                    if (alea::mcdata<T>::has_variance())
                        local.push_back(alea::mcdata<T>::variance() * static_cast<element_type>(count()));
                    boost::mpi::reduce(communicator, local, global, std::plus<element_type>(), 0);
                    boost::optional<result_type> global_variance_opt;
                    if (alea::mcdata<T>::has_variance())
                        global_variance_opt = global[2] / static_cast<element_type>(global_count);
                    boost::optional<time_type> global_tau_opt;
                    if (alea::mcdata<T>::has_tau()) {
                        time_type global_tau;
                        boost::mpi::reduce(communicator, alea::mcdata<T>::tau() * static_cast<double>(count()), global_tau, std::plus<double>(), 0);
                        global_tau_opt = global_tau / static_cast<double>(global_count);
                    }
                    std::vector<result_type> global_bins(alea::mcdata<T>::bin_number() > 0 ? binnumber : 0);
                    std::size_t binsize = 0;
                    if (alea::mcdata<T>::bin_number() > 0) {
                        std::vector<result_type> local_bins(binnumber);
                        binsize = partition_bins(local_bins, communicator);
                        boost::mpi::reduce(communicator, local_bins, global_bins, std::plus<result_type>(), 0);
                    }
                    boost::assign::ptr_map_insert<mcdata<value_type> >(results)(name, mcdata<value_type>(
                          global_count
                        , global[0] / static_cast<element_type>(global_count)
                        , sqrt(global[1]) / static_cast<element_type>(global_count)
                        , global_variance_opt
                        , global_tau_opt
                        , binsize
                        , global_bins
                    ));
                }

                void reduce_master_impl(
                      boost::ptr_map<std::string, mcany> & results
                    , std::string const & name
                    , boost::mpi::communicator const & communicator
                    , std::size_t binnumber
                    , boost::mpl::false_
                ) {
                    using alps::numeric::sq;
                    using alps::numeric::sqrt;
                    using boost::numeric::operators::operator*;
                    using boost::numeric::operators::operator/;
                    uint64_t global_count;
                    boost::mpi::reduce(communicator, count(), global_count, std::plus<uint64_t>(), 0);
                    result_type global_mean, global_error, global_variance;
                    boost::mpi::reduce(communicator, alea::mcdata<T>::mean() * static_cast<element_type>(count()), global_mean, std::plus<element_type>(), 0);
                    boost::mpi::reduce(communicator, sq(alea::mcdata<T>::error()) * sq(static_cast<element_type>(count())), global_error, std::plus<element_type>(), 0);
                    boost::optional<result_type> global_variance_opt;
                    if (alea::mcdata<T>::has_variance()) {
                        boost::mpi::reduce(communicator, alea::mcdata<T>::variance() * static_cast<element_type>(count()), global_variance, std::plus<element_type>(), 0);
                        global_variance_opt = global_variance / static_cast<element_type>(global_count);
                    }
                    boost::optional<time_type> global_tau_opt;
                    if (alea::mcdata<T>::has_tau()) {
                        time_type global_tau;
                        boost::mpi::reduce(communicator, alea::mcdata<T>::tau() * static_cast<double>(count()), global_tau, std::plus<double>(), 0);
                        global_tau_opt = global_tau / static_cast<double>(global_count);
                    }
                    std::vector<result_type> global_bins(alea::mcdata<T>::bin_number() > 0 ? binnumber : 0);
                    std::size_t binsize = 0, elementsize = alea::mcdata<T>::mean().size();
                    if (alea::mcdata<T>::bin_number() > 0) {
                        std::vector<result_type> local_bins(binnumber, result_type(elementsize));
                        binsize = partition_bins(local_bins, communicator);
                        std::vector<typename alea::mcdata<T>::element_type> local_raw_bins(binnumber * elementsize), global_raw_bins(binnumber * elementsize);
                        for (typename std::vector<result_type>::iterator it = local_bins.begin(); it != local_bins.end(); ++it)
                            std::copy(it->begin(), it->end(), local_raw_bins.begin() + ((it - local_bins.begin()) * elementsize));
                        boost::mpi::reduce(communicator, local_raw_bins, global_raw_bins, std::plus<element_type>(), 0);
                        for (typename std::vector<result_type>::iterator it = global_bins.begin(); it != global_bins.end(); ++it) {
                            it->resize(elementsize);
                            std::copy(global_raw_bins.begin() + (it - global_bins.begin()) * elementsize, global_raw_bins.begin() + (it - global_bins.begin() + 1) * elementsize, it->begin());
                        }
                    }
                    boost::assign::ptr_map_insert<mcdata<value_type> >(results)(name, mcdata<value_type>(
                          global_count
                        , global_mean / static_cast<element_type>(global_count)
                        , sqrt(global_error) / static_cast<element_type>(global_count)
                        , global_variance_opt
                        , global_tau_opt
                        , elementsize * binsize
                        , global_bins
                    ));
                }

                void reduce_slave_impl(boost::mpi::communicator const & communicator, std::size_t binnumber, boost::mpl::true_) {
                    using alps::numeric::sq;
                    boost::mpi::reduce(communicator, count(), std::plus<uint64_t>(), 0);
                    std::vector<result_type> local(2, 0);
                    local[0] = alea::mcdata<T>::mean() * static_cast<element_type>(count());
                    local[1] = sq(alea::mcdata<T>::error()) * sq(static_cast<element_type>(count()));
                    if (alea::mcdata<T>::has_variance())
                        local.push_back(alea::mcdata<T>::variance() * static_cast<element_type>(count()));
                    boost::mpi::reduce(communicator, local, std::plus<element_type>(), 0);
                    if (alea::mcdata<T>::has_tau())
                        boost::mpi::reduce(communicator, alea::mcdata<T>::tau() * static_cast<double>(count()), std::plus<double>(), 0);
                    if (alea::mcdata<T>::bin_number() > 0) {
                        std::vector<result_type> local_bins(binnumber);
                        partition_bins(local_bins, communicator);
                        boost::mpi::reduce(communicator, local_bins, std::plus<element_type>(), 0);
                    }
                }

                void reduce_slave_impl(boost::mpi::communicator const & communicator, std::size_t binnumber, boost::mpl::false_) {
                    using alps::numeric::sq;
                    using boost::numeric::operators::operator*;
                    boost::mpi::reduce(communicator, count(), std::plus<uint64_t>(), 0);
                    boost::mpi::reduce(communicator, alea::mcdata<T>::mean() * static_cast<element_type>(count()), std::plus<element_type>(), 0);
                    boost::mpi::reduce(communicator, sq(alea::mcdata<T>::error()) * sq(static_cast<element_type>(count())), std::plus<element_type>(), 0);
                    if (alea::mcdata<T>::has_variance())
                        boost::mpi::reduce(communicator, alea::mcdata<T>::variance() * static_cast<element_type>(count()), std::plus<element_type>(), 0);
                    if (alea::mcdata<T>::has_tau())
                        boost::mpi::reduce(communicator, alea::mcdata<T>::tau() * static_cast<double>(count()), std::plus<double>(), 0);
                    std::size_t elementsize = alea::mcdata<T>::mean().size();
                    if (alea::mcdata<T>::bin_number() > 0) {
                        std::vector<result_type> local_bins(binnumber, result_type(elementsize));
                        partition_bins(local_bins, communicator);
                        std::vector<typename alea::mcdata<T>::element_type> local_raw_bins(binnumber * elementsize);
                        for (typename std::vector<result_type>::iterator it = local_bins.begin(); it != local_bins.end(); ++it)
                            std::copy(it->begin(), it->end(), local_raw_bins.begin() + ((it - local_bins.begin()) * elementsize));
                        boost::mpi::reduce(communicator, local_raw_bins, std::plus<element_type>(), 0);
                    }
                }

                std::size_t partition_bins (std::vector<result_type> & bins, boost::mpi::communicator const & communicator) {
                    using boost::numeric::operators::operator+;
                    alea::mcdata<T>::set_bin_size(boost::mpi::all_reduce(communicator, alea::mcdata<T>::bin_size(), boost::mpi::maximum<std::size_t>()));
                    std::vector<int> buffer(2 * communicator.size()), index(communicator.size());
                    int data[2] = {communicator.rank(), alea::mcdata<T>::bin_number()};
                    boost::mpi::all_gather(communicator, data, 2, buffer);
                    for (std::vector<int>::const_iterator it = buffer.begin(); it != buffer.end(); it += 2)
                        index[*it] = *(it + 1);
                    int perbin = std::accumulate(index.begin(), index.end(), 0) / bins.size();
                    if (perbin == 0)
                        throw std::runtime_error("not enough data for the required binnumber");
                    int start = std::accumulate(index.begin(), index.begin() + communicator.rank(), 0);
                    for (int i = start / perbin, j = start % perbin, k = 0; i < bins.size() && k < alea::mcdata<T>::bin_number(); ++k) {
                        bins[i] = bins[i] + alea::mcdata<T>::bins()[k];
                        if (++j == perbin) {
                            ++i;
                            j = 0;
                        }
                    }
                    return perbin;
                }
        };
    }
}

#endif
