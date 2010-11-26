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
#include <alps/parameter.h>

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

#ifndef ALPS_NGS_HPP
#define ALPS_NGS_HPP

namespace alps {
    class mcoptions {
        public:
            typedef enum { SINGLE, MPI } execution_types;

            mcoptions(int argc, char* argv[]) : valid(false), continueing(false), type(SINGLE) {
                boost::program_options::options_description desc("Allowed options");
                desc.add_options()
                    ("help", "produce help message")
                    ("single", "run single process")
                    ("mpi", "run in parallel using MPI")
                    ("continue", "load simulation from checkpoint")
                    ("time-limit,T", boost::program_options::value<std::size_t>(&time_limit)->default_value(0), "time limit for the simulation")
                    ("input-file", boost::program_options::value<std::string>(&input_file), "input file in hdf5 format")
                    ("output-file", boost::program_options::value<std::string>(&output_file)->default_value("sim.h5"), "output file in hdf5 format");
                boost::program_options::positional_options_description p;
                p.add("input-file", 1);
                p.add("output-file", 2);
                boost::program_options::variables_map vm;
                boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
                boost::program_options::notify(vm);
                if (!(valid = !vm.count("help")))
                    std::cout << desc << std::endl;
                else if (input_file.empty())
                    throw std::invalid_argument("No job file specified");
                if (vm.count("mpi"))
                    type = MPI;
                if (vm.count("continue"))
                    continueing = true;
            }

            bool valid;
            bool continueing;
            std::size_t time_limit;
            std::string input_file;
            std::string output_file;
            execution_types type;
    };

    namespace detail {
        typedef boost::mpl::vector<std::string, int, double> mcparamvalue_types;
        typedef boost::make_variant_over<mcparamvalue_types>::type mcparamvalue_base;

        template<typename T> struct mcparamvalue_reader : public boost::static_visitor<> {
            template <typename U> void operator()(U & v) const { value = boost::lexical_cast<T, U>(v); }
            void operator()(T & v) const { value = v; }
            mutable T value;
        };

        struct mcparamvalue_serializer: public boost::static_visitor<> {
            mcparamvalue_serializer(hdf5::oarchive & a, std::string const & p) : ar(a), path(p) {}
            template<typename T> void operator()(T & v) const { ar << make_pvp(path, v); }
            hdf5::oarchive & ar;
            std::string const & path;
        };
    }

    class mcparamvalue : public detail::mcparamvalue_base {
        public:
            mcparamvalue() {}
            template <typename T> mcparamvalue(T const & v): detail::mcparamvalue_base(v) {}
            mcparamvalue(mcparamvalue const & v): detail::mcparamvalue_base(static_cast<detail::mcparamvalue_base const &>(v)) {}

            template <typename T> typename boost::enable_if<typename boost::mpl::contains<detail::mcparamvalue_types, T>::type, mcparamvalue &>::type operator=(T const & v) {
                detail::mcparamvalue_base::operator=(v);
                return *this;
            }

            template <typename T> typename boost::disable_if<typename boost::mpl::contains<detail::mcparamvalue_types, T>::type, mcparamvalue &>::type operator=(T const & v) {
                detail::mcparamvalue_base::operator=(boost::lexical_cast<std::string>(v));
                return *this;
            }

            #define ALPS_NGS_CAST_OPERATOR(T)                                                                                                              \
                operator T () const {                                                                                                                      \
                    detail::mcparamvalue_reader< T > visitor;                                                                                              \
                    boost::apply_visitor(visitor, *this);                                                                                                  \
                    return visitor.value;                                                                                                                  \
                }
            ALPS_NGS_CAST_OPERATOR(short)
            ALPS_NGS_CAST_OPERATOR(unsigned short)
            ALPS_NGS_CAST_OPERATOR(int)
            ALPS_NGS_CAST_OPERATOR(unsigned int)
            ALPS_NGS_CAST_OPERATOR(long)
            ALPS_NGS_CAST_OPERATOR(unsigned long)
            ALPS_NGS_CAST_OPERATOR(long long)
            ALPS_NGS_CAST_OPERATOR(unsigned long long)
            ALPS_NGS_CAST_OPERATOR(float)
            ALPS_NGS_CAST_OPERATOR(double)
            ALPS_NGS_CAST_OPERATOR(long double)
            ALPS_NGS_CAST_OPERATOR(bool)
            ALPS_NGS_CAST_OPERATOR(std::string)
            #undef ALPS_NGS_CAST_OPERATOR
    };

    // can we keep parameter ordering? like in curent class
    class mcparams : public std::map<std::string, mcparamvalue> {
        public: 
            mcparams(std::string const & input_file) {
                hdf5::iarchive ar(input_file);
                ar >> make_pvp("/parameters", *this);
            }

            mcparamvalue & operator[](std::string const & k) {
                return std::map<std::string, mcparamvalue>::operator[](k);
            }

            mcparamvalue const & operator[](std::string const & k) const {
                if (find(k) == end())
                    throw std::invalid_argument("unknown argument: "  + k);
                return find(k)->second;
            }

            mcparamvalue value_or_default(std::string const & k, mcparamvalue const & v) const {
                if (find(k) == end())
                    return mcparamvalue(v);
                return find(k)->second;
            }

            bool defined(std::string const & k) const {
                return find(k) != end();
            }

            void serialize(hdf5::oarchive & ar) const {
                for (const_iterator it = begin(); it != end(); ++it)
                    boost::apply_visitor(detail::mcparamvalue_serializer(ar, it->first), it->second);
            }

            void serialize(hdf5::iarchive & ar) {
                std::vector<std::string> list = ar.list_children(ar.get_context());
                for (std::vector<std::string>::const_iterator it = list.begin(); it != list.end(); ++it) {
                    std::string v;
                    ar >> make_pvp(*it, v);
                    insert(std::make_pair(*it, v));
                }
            }
    };

    namespace detail {
        // GET RID OF Unused and write a 1-line .cpp file!
        template<typename Unused = void> class mcsignal_impl{
            public:
                mcsignal_impl() {
                    #ifndef BOOST_MSVC
                        static bool initialized;
                        if (!initialized) {
                            static struct sigaction action;
                            initialized = true;
                            memset(&action, 0, sizeof(action));
                            action.sa_handler = &mcsignal_impl::slot;
                            sigaction(SIGINT, &action, NULL);
                            sigaction(SIGTERM, &action, NULL);
                            sigaction(SIGXCPU, &action, NULL);
                            sigaction(SIGQUIT, &action, NULL);
                            sigaction(SIGUSR1, &action, NULL);
                            sigaction(SIGUSR2, &action, NULL);
                            sigaction(SIGSTOP, &action, NULL);
                        }
                    #endif
                }

                bool empty() {
                    return !signals_.size();
                }

                int top() {
                    return signals_.back();
                }

                void pop() {
                    return signals_.pop_back();
                }

                static void slot(int signal) {
                    std::cerr << "Received signal " << signal << std::endl;
                    signals_.push_back(signal);
                }

            private:

                static std::vector<int> signals_;
        };
        template<typename Unused> std::vector<int> mcsignal_impl<Unused>::signals_;
    }

    typedef detail::mcsignal_impl<> mcsignal;

    namespace detail {
        template<typename T> struct short_print_proxy {
            public:
                explicit short_print_proxy(T const & v): value(v) {};
                short_print_proxy(short_print_proxy<T> const & rhs): value(rhs.value) {};
                T const & value;
        };
    }

    template<typename T> detail::short_print_proxy<T const> short_print(T const & v) {
        return detail::short_print_proxy<T const>(v);
    }

    template <typename T> std::ostream & operator<<(std::ostream & os, detail::short_print_proxy<T> const & v) {
        return os << v.value;
     }

    template <typename T> std::ostream & operator<<(std::ostream & os, detail::short_print_proxy<std::vector<T> const> const & v) {
        switch (v.value.size()) {
            case 0: 
                return os << "[]";
            case 1: 
                return os << "[" << short_print(v.value.front()) << "]";
            case 2: 
                return os << "[" << short_print(v.value.front()) << "," << short_print(v.value.back()) << "]";
            default: 
                return os << "[" << short_print(v.value.front()) << ",.." << short_print(v.value.size()) << "..," << short_print(v.value.back()) << "]";
        }
    }

}

namespace boost {
    namespace mpi {

        template<typename T, typename Op> void reduce(const communicator & comm, std::vector<T> const & in_values, Op op, int root) {
            reduce(comm, &in_values.front(), in_values.size(), op, root);
        }

        template<typename T, typename Op> void reduce(const communicator & comm, std::vector<T> const & in_values, std::vector<T> & out_values, Op op, int root) {
            out_values.resize(in_values.size());
            reduce(comm, &in_values.front(), in_values.size(), &out_values.front(), op, root);
        }

    }
}

namespace alps {

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

            template<typename T> operator alea::mcdata<T>() const;

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
            typedef typename alea::mcdata<T>::value_type value_type;
            typedef typename alea::mcdata<T>::result_type result_type;
            // TODO: only support double!
            typedef typename alps::element_type<T>::type element_type;
            typedef typename change_value_type<T, double>::type time_type;

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
            
            void reduce_slave(boost::mpi::communicator const & communicator, std::size_t binnumber) { 
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

            void reduce_slave_impl(
                  boost::mpi::communicator const & communicator
                , std::size_t binnumber
                , boost::mpl::true_
            ) {
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

            void reduce_slave_impl(
                  boost::mpi::communicator const & communicator
                , std::size_t binnumber
                , boost::mpl::false_
            ) {
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
    
    template<typename T> mcany::operator alea::mcdata<T>() const {
       return static_cast<alea::mcdata<T> >(dynamic_cast<mcdata<T> const & >(*this));
    }

    template<typename S> struct result_names_type {
        typedef typename S::result_names_type type;
    };

    template<typename S> struct results_type {
        typedef typename S::results_type type;
    };

    template<typename S> struct parameters_type {
        typedef typename S::parameters_type type;
    };

    template<typename S> typename result_names_type<S>::type result_names(S const & s) {
        return s.result_names();
    }

    template<typename S> typename result_names_type<S>::type unsaved_result_names(S const & s) {
        return s.unsaved_result_names();
    }

    template<typename S> typename results_type<S>::type collect_results(S const & s) {
        return s.collect_results();
    }

    template<typename S> typename results_type<S>::type collect_results(S const & s, typename result_names_type<S>::type const & names) {
        return s.collect_results(names);
    }

    template<typename S> typename results_type<S>::type collect_results(S const & s, std::string const & name) {
        return collect_results(s, typename result_names_type<S>::type(1, name));
    }

    template<typename S> double fraction_completed(S const & s) {
        return s.fraction_completed();
    }

    class mcbase {
        public:
            typedef mcparams parameters_type;
            typedef boost::ptr_map<std::string, mcany> results_type;
            typedef std::vector<std::string> result_names_type;

            mcbase(parameters_type const & p, std::size_t seed_offset = 0)
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
                            if (it->second->count() > 0)
                                ar << alps::make_pvp("/simulation/results/" + it->first, *(it->second));
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

    class mcdeprecated : public mcbase {
        public:
            mcdeprecated(parameters_type const & p, std::size_t seed_offset = 0)
                : mcbase(p, seed_offset)
                , parms(make_alps_parameters(p))
                , measurements(results)
                , random_01(random)
            {}

            double fraction_completed() const { return work_done(); }

            virtual double work_done() const = 0;

            virtual void dostep() = 0;

            double random_real(double a = 0., double b = 1.) { return a + b * random(); }

            virtual void do_update() {
                dostep();
            }

            virtual void do_measurements() {}

        protected:
            Parameters parms;
            ObservableSet & measurements;
            boost::variate_generator<boost::mt19937, boost::uniform_real<> > & random_01;

        private:
            static Parameters make_alps_parameters(parameters_type const & s) {
                Parameters p;
                for (parameters_type::const_iterator it = s.begin(); it != s.end(); ++it)
// TODO: why does static_cast<std::string>(it->second) not work?
                    p.push_back(it->first, it->second.operator std::string());
                return p;
            }
    };

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
                    if (it->second->count() > 0 && communicator.rank() == 0)
                        it->second->reduce_master(partial_results, it->first, communicator, binnumber);
                    else if (it->second->count() > 0)
                        it->second->reduce_slave(communicator, binnumber);
                return partial_results;
            }

        private:
            boost::mpi::communicator communicator;
            std::size_t binnumber;
    };
}

#endif
