/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2010 by Lukas Gamper <gamperl -at- gmail.com>
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
#include <alps/parameter.h>
#include <alps/hdf5.hpp>

#include <boost/mpi.hpp>
#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include <boost/utility.hpp>
#include <boost/function.hpp>
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
            typedef enum { SINGLE, THREADED, MPI } execution_types;
            mcoptions(int argc, char* argv[]) : valid(false), type(SINGLE) {
                boost::program_options::options_description desc("Allowed options");
                desc.add_options()
                    ("help", "produce help message")
                    ("single", "run a single thread") 
                    ("threaded", "runing multiple threads") 
                    ("mpi", "run in parallel using MPI") 
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
                    boost::throw_exception(std::runtime_error("No job file specified"));
                if (vm.count("threaded"))
                    type = THREADED;
                else if (vm.count("mpi"))
                    type = MPI;
            }
            bool valid;
            std::size_t time_limit;
            std::string input_file;
            std::string output_file;
            execution_types type;
    };

    // TODO: use boost::variant instead of string
    class mcparamvalue : public std::string {
        public:
            mcparamvalue() {}
            template <typename T> mcparamvalue(mcparamvalue const & v)
                : std::string(v) 
            {}
            template <typename T> mcparamvalue(T const & v)
                : std::string(boost::lexical_cast<std::string>(v)) 
            {}
            template <typename T> typename boost::disable_if<typename boost::is_base_of<std::string, T>::type, mcparamvalue &>::type operator=(T const & v) {
                std::string::operator=(boost::lexical_cast<std::string>(v));
                return *this;
            }
            template <typename T> typename boost::enable_if<typename boost::is_base_of<std::string, T>::type, mcparamvalue &>::type operator=(T const & v) {
                std::string::operator=(v);
                return *this;
            }
            template <typename T> operator T() const {
                return boost::lexical_cast<T, std::string>(*this);
            }
    };

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
                    ar << make_pvp(it->first, static_cast<std::string>(it->second));
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

    template<typename unused = void> class mcsignal_impl{
        public:
            mcsignal_impl() {
                static bool initialized;
                if (!initialized) {
                    static struct sigaction action;
                    initialized = true;
                    memset(&action, 0, sizeof(action));
                    action.sa_handler = &mcsignal_impl<unused>::slot;
                    sigaction(SIGINT, &action, NULL);
                    sigaction(SIGTERM, &action, NULL);
                    sigaction(SIGXCPU, &action, NULL);
                    sigaction(SIGQUIT, &action, NULL);
                    sigaction(SIGUSR1, &action, NULL);
                    sigaction(SIGUSR2, &action, NULL);
                }
            }
            operator bool() const { return which; }
            static int code() { return which; }
            static void slot(int signal) { 
                std::cerr << "Killed by signal " << signal << std::endl;
                which = signal;
            }
        private:
            static int which;
    };
    template<typename unused> int mcsignal_impl<unused>::which = 0;
    typedef mcsignal_impl<> mcsignal;

    template <typename T> void mcmpierror(T code) {
        if (code != MPI_SUCCESS) {
            char buffer[BUFSIZ];
            int size;
            MPI_Error_string(code, buffer, &size);
            throw std::logic_error(std::string("MPI Error: ") + buffer); 
        }
    }

    template <typename T> std::ostream & operator<<(std::ostream & os, std::vector<T> const & value) {
        switch (value.size()) {
            case 0: os << "[]"; break;
            case 1: os << "[" << value.front() << "]"; break;
            case 2: os << "[" << value.front() << "," << value.back() << "]"; break;
            default: os << "[" << value.front() << ",.." << value.size() << "..," << value.back() << "]";
        }
        return os;
    }

    template <typename T> std::size_t mcsize(T const & value) { 
        return 1; 
    }

    template <typename T> std::size_t mcsize(std::vector<T> const & value) { 
        return value.size(); 
    }

    template <typename T> void mcresize(T & value, std::size_t size) {}

    template <typename T> void mcresize(std::vector<T> & value, std::size_t size) { 
        return value.resize(size);
    }

    template <typename T> typename alps::element_type<T>::type * mcpointer(T & value) { 
        return &value; 
    }

    template <typename T> typename alps::element_type<T>::type * mcpointer(std::vector<T> & value) {
        return &value.front(); 
    }

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

    inline std::ostream & operator<<(std::ostream & os, boost::ptr_map<std::string, mcany>  const & results) {
        for (boost::ptr_map<std::string, mcany>::const_iterator it = results.begin(); it != results.end(); ++it)
            std::cout << std::fixed << std::setprecision(5) << it->first << ": " << it->second->to_string() << std::endl;
        return os;
    }

    template <typename T> class mcdata : public mcany, alea::mcdata<T> {
        public:
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

            typename std::string to_string() const {
                if (count() == 0)
                    return "No Measurements";
                else {
                    std::stringstream s;
                    s << std::fixed << std::setprecision(5) << alea::mcdata<T>::mean() << "(" << count() << ") +/-" << alea::mcdata<T>::error() << " "
                      << alea::mcdata<T>::bins() << "#" << alea::mcdata<T>::bin_size();
                    return s.str();
                }
            }

            void serialize(hdf5::iarchive & ar) { 
                alea::mcdata<T>::serialize(ar);
            }

            void serialize(hdf5::oarchive & ar) const { 
                alea::mcdata<T>::serialize(ar);
            }

            void reduce_master(boost::ptr_map<std::string, mcany> & results, std::string const & name, boost::mpi::communicator const & communicator, std::size_t binnumber) {
                using std::sqrt;
                using alps::numeric::sq;
                using alps::numeric::sqrt;
                using boost::numeric::operators::operator*;
                using boost::numeric::operators::operator/;
                uint64_t count_all;
                boost::mpi::reduce(communicator, count(), count_all, std::plus<uint64_t>(), 0);
                result_type mean = alea::mcdata<T>::mean() * static_cast<typename alea::mcdata<T>::element_type>(count()), mean_all;
                mcresize(mean_all, mcsize(mean));
                mcmpierror(MPI_Reduce(mcpointer(mean), mcpointer(mean_all), mcsize(mean), boost::mpi::get_mpi_datatype(typename alea::mcdata<T>::element_type()), MPI_SUM, 0, communicator));
                result_type error = sq(alea::mcdata<T>::error()) * sq(static_cast<typename alea::mcdata<T>::element_type>(count())), error_all;
                mcresize(error_all, mcsize(error));
                mcmpierror(MPI_Reduce(mcpointer(error), mcpointer(error_all), mcsize(error), boost::mpi::get_mpi_datatype(typename alea::mcdata<T>::element_type()), MPI_SUM, 0, communicator));
                boost::optional<typename alea::mcdata<T>::result_type> variance_all_opt;
                if (alea::mcdata<T>::has_variance()) {
                    result_type variance = alea::mcdata<T>::variance() * static_cast<typename alea::mcdata<T>::element_type>(count()), variance_all;
                    mcresize(variance_all, mcsize(variance));
                    mcmpierror(MPI_Reduce(mcpointer(variance), mcpointer(variance_all), mcsize(variance), boost::mpi::get_mpi_datatype(typename alea::mcdata<T>::element_type()), MPI_SUM, 0, communicator));
                    variance_all_opt = variance_all / static_cast<typename alea::mcdata<T>::element_type>(count_all);
                }
                boost::optional<typename alea::mcdata<T>::time_type> tau_all_opt;
                if (alea::mcdata<T>::has_tau()) {
                    result_type tau = alea::mcdata<T>::tau() * static_cast<typename alea::mcdata<T>::element_type>(count()), tau_all;
                    mcresize(tau_all, mcsize(tau));
                    mcmpierror(MPI_Reduce(mcpointer(tau), mcpointer(tau_all), mcsize(tau), boost::mpi::get_mpi_datatype(typename alea::mcdata<T>::element_type()), MPI_SUM, 0, communicator));
                    tau_all_opt = tau_all / static_cast<typename alea::mcdata<T>::element_type>(count_all);
                }
                std::vector<result_type> bins;
                std::size_t binsize = 0;
                if (alea::mcdata<T>::bin_number() > 0) {
                    std::vector<result_type> local_bins(binnumber);
                    for (typename std::vector<result_type>::iterator it = local_bins.begin(); it != local_bins.end(); ++it)
                        mcresize(*it, mcsize(mean));
                    binsize = partition_bins(local_bins, communicator);
                    bins.resize(binnumber);
                    if (boost::is_scalar<result_type>::value) {
                        mcmpierror(MPI_Reduce(&local_bins.front(), &bins.front(), binnumber, boost::mpi::get_mpi_datatype(typename alea::mcdata<T>::element_type()), MPI_SUM, 0, communicator));
                    } else {
                        std::vector<typename alea::mcdata<T>::element_type> raw_local_bins(binnumber * mcsize(mean)), raw_bins(binnumber * mcsize(mean));
                        for (typename std::vector<result_type>::iterator it = local_bins.begin(); it != local_bins.end(); ++it)
                            std::copy(mcpointer(*it), mcpointer(*it) + mcsize(*it), mcpointer(raw_local_bins[(it - local_bins.begin()) * mcsize(mean)]));
                        mcmpierror(MPI_Reduce(&raw_local_bins.front(), &raw_bins.front(), binnumber * mcsize(mean), boost::mpi::get_mpi_datatype(typename alea::mcdata<T>::element_type()), MPI_SUM, 0, communicator));
                        for (typename std::vector<result_type>::iterator it = bins.begin(); it != bins.end(); ++it) {
                            mcresize(*it, mcsize(mean));
                            std::copy(raw_bins.begin() + (it - bins.begin()) * mcsize(mean), raw_bins.begin() + (it - bins.begin() + 1) * mcsize(mean), mcpointer(*it));
                        }
                    }
                }
                boost::assign::ptr_map_insert<mcdata<value_type> >(results)(name, mcdata<value_type>(
                      count_all
                    , mean_all / typename alea::mcdata<T>::element_type(count_all)
                    , sqrt(error_all) / typename alea::mcdata<T>::element_type(count_all)
                    , variance_all_opt
                    , tau_all_opt
                    , mcsize(mean) * binsize
                    , bins
                ));
            }

            void reduce_slave(boost::mpi::communicator const & communicator, std::size_t binnumber) {
                using alps::numeric::sq;
                using boost::numeric::operators::operator*;
                boost::mpi::reduce(communicator, count(), std::plus<uint64_t>(), 0);
                result_type mean = alea::mcdata<T>::mean() * static_cast<typename alea::mcdata<T>::element_type>(count());
                mcmpierror(MPI_Reduce(mcpointer(mean), NULL, mcsize(mean), boost::mpi::get_mpi_datatype(typename alea::mcdata<T>::element_type()), MPI_SUM, 0, communicator));
                result_type error = sq(alea::mcdata<T>::error()) * sq(static_cast<typename alea::mcdata<T>::element_type>(count()));
                mcmpierror(MPI_Reduce(mcpointer(error), NULL, mcsize(error), boost::mpi::get_mpi_datatype(typename alea::mcdata<T>::element_type()), MPI_SUM, 0, communicator));
                if (alea::mcdata<T>::has_variance()) {
                    result_type variance = alea::mcdata<T>::variance() * static_cast<typename alea::mcdata<T>::element_type>(count());
                    mcmpierror(MPI_Reduce(mcpointer(variance), NULL, mcsize(variance), boost::mpi::get_mpi_datatype(typename alea::mcdata<T>::element_type()), MPI_SUM, 0, communicator));
                }
                if (alea::mcdata<T>::has_tau()) {
                    result_type tau = alea::mcdata<T>::tau() * static_cast<typename alea::mcdata<T>::element_type>(count());
                    mcmpierror(MPI_Reduce(mcpointer(tau), NULL, mcsize(tau), boost::mpi::get_mpi_datatype(typename alea::mcdata<T>::element_type()), MPI_SUM, 0, communicator));
                }
                if (alea::mcdata<T>::bin_number() > 0) {
                    std::vector<result_type> local_bins(binnumber);
                    for (typename std::vector<result_type>::iterator it = local_bins.begin(); it != local_bins.end(); ++it)
                        mcresize(*it, mcsize(mean));
                    partition_bins(local_bins, communicator);
                    if (boost::is_scalar<result_type>::value)
                        mcmpierror(MPI_Reduce(&local_bins.front(), NULL, binnumber, boost::mpi::get_mpi_datatype(typename alea::mcdata<T>::element_type()), MPI_SUM, 0, communicator));
                    else {
                        std::vector<typename alea::mcdata<T>::element_type> raw_local_bins(binnumber * mcsize(mean));
                        for (typename std::vector<result_type>::iterator it = local_bins.begin(); it != local_bins.end(); ++it)
                            std::copy(mcpointer(*it), mcpointer(*it) + mcsize(*it), mcpointer(raw_local_bins[(it - local_bins.begin()) * mcsize(mean)]));
                        mcmpierror(MPI_Reduce(&raw_local_bins.front(), NULL, binnumber * mcsize(mean), boost::mpi::get_mpi_datatype(typename alea::mcdata<T>::element_type()), MPI_SUM, 0, communicator));
                    }
                }
            }

        private:
        
            std::size_t partition_bins (std::vector<result_type> & bins, boost::mpi::communicator const & communicator) {
                using boost::numeric::operators::operator+;
                alea::mcdata<T>::set_bin_size(boost::mpi::all_reduce(communicator, alea::mcdata<T>::bin_size(), boost::mpi::maximum<std::size_t>()));
                std::vector<int> buffer(2 * communicator.size()), index(communicator.size());
                int data[2] = {communicator.rank(), alea::mcdata<T>::bin_number()};
                MPI_Allgather (data, 2, MPI_INT, &buffer.front(), 2, MPI_INT, communicator);
                for (std::vector<int>::const_iterator it = buffer.begin(); it != buffer.end(); it += 2)
                    index[*it] = *(it + 1);
                int perbin = std::accumulate(index.begin(), index.end(), 0) / bins.size();
                int start = std::accumulate(index.begin(), index.begin() + communicator.rank(), 0);
                for (int i = start / perbin, j = start - i, k = 0; i < bins.size() && k < alea::mcdata<T>::bin_number(); ++k) {
                    bins[i] = bins[i] + alea::mcdata<T>::bins()[k];
                    if (++j == perbin) {
                        ++i;
                        j = 0;
                    }
                }
                return perbin;
            }
    };

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

            mcbase(parameters_type const & p)
                : params(p)
                , random(boost::mt19937(), boost::uniform_real<>())
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
                hdf5::iarchive ar(path.file_string());
                ar >> make_pvp("/simulation/realizations/0/clones/0/results", results);
            }

            bool run(boost::function<bool ()> const & stop_callback) {
                do {
                    do_update();
                    do_measurements();
                } while(!stop_callback() && fraction_completed() < 1);
                return fraction_completed() >= 1;
            }

            result_names_type result_names() const {
                result_names_type names;
                for(ObservableSet::const_iterator it = results.begin(); it != results.end(); ++it)
                    names.push_back(it->first);
                return names;
            }

            result_names_type unsaved_result_names() const { return result_names_type(); }

            results_type collect_results() const { return collect_results(result_names_type()); }

            results_type collect_results(result_names_type const & names) const {
                results_type partial_results;
                for(result_names_type::const_iterator it = names.begin(); it != names.end(); ++it)
                    if (dynamic_cast<AbstractSimpleObservable<double> const *>(&results[*it]) != NULL)
                        boost::assign::ptr_map_insert<mcdata<double> >(partial_results)(*it, *dynamic_cast<AbstractSimpleObservable<double> const *>(&results[*it]));
                    else if (dynamic_cast<AbstractSimpleObservable<std::valarray<double> > const *>(&results[*it]) != NULL)
                        boost::assign::ptr_map_insert<mcdata<std::vector<double> > >(partial_results)(*it, *dynamic_cast<AbstractSimpleObservable<std::valarray<double> > const *>(&results[*it]));
                    else
                        boost::throw_exception(std::runtime_error("unknown observable type"));
                return partial_results;
            }
     
        protected:
            parameters_type params;
            ObservableSet results;
            boost::variate_generator<boost::mt19937, boost::uniform_real<> > random;
    };

    class mcdeprecated : public mcbase {
        public:
            mcdeprecated(parameters_type const & p)
                : mcbase(p)
                , parms(make_alps_parameters(p))
                , measurements(results)
                , random_01(random)
            {}

            void do_update() {}

            void do_measurements() {}

            double fraction_completed() const { work_done(); }

            virtual double work_done() const = 0;

            virtual void dostep() = 0;

            double random_real(double a = 0., double b = 1.) { return a + b * random(); }

            bool run(boost::function<bool ()> const & stop_callback) {
                do {
                    dostep();
                } while(!stop_callback() && fraction_completed() < 1);
                return fraction_completed() >= 1;
            }

        protected:
            Parameters parms;
            ObservableSet & measurements;
            boost::variate_generator<boost::mt19937, boost::uniform_real<> > & random_01;

        private:
            static Parameters make_alps_parameters(parameters_type const & s) {
                Parameters p;
                for (parameters_type::const_iterator it = s.begin(); it != s.end(); ++it)
                    p.push_back(it->first, static_cast<std::string>(it->second));
                return p;
            }
    };
    
    template<typename T> class mcatomic {
        public:
            mcatomic() {}
            mcatomic(T const & v): value(v) {}
            mcatomic(mcatomic const & v): value(v.value) {}
            
            mcatomic & operator=(mcatomic const & v) {
                boost::lock_guard<boost::mutex> lock(mutex);
                value = v;
            }

            operator T() const { 
                boost::lock_guard<boost::mutex> lock(mutex);
                return value; 
            }
        private:
            T volatile value;
            boost::mutex mutable mutex;
    };
    template<typename Impl> class mcthreadsim : public Impl {
        public:
            mcthreadsim(typename Impl::parameters_type const & p)
                : Impl(p)
                , stop_flag(false)
            {}
            
            bool run(boost::function<bool ()> const & stop_callback) {
                boost::thread thread(boost::bind(&mcthreadsim<Impl>::thread_callback, this, stop_callback));
                Impl::run(boost::bind(&mcthreadsim<Impl>::run_callback, this));
                thread.join();
                return stop_flag;
            }
            
            virtual bool run_callback() {
                return stop_flag;
            }
            
            virtual void thread_callback(boost::function<bool ()> const & stop_callback) {
                while (true) {
                    usleep(0.1 * 1e6);
                    if (stop_flag = stop_callback())
                        return;
                }
            }

        protected:
            mcatomic<bool> stop_flag;
            // boost::mutex mutex;
            // measurements and configuration need to be locked separately
    };
    template<typename Impl> class mcmpisim : public mcthreadsim<Impl> {
        public:
            enum {
                MPI_get_fraction    = 1,
                MPI_stop            = 2,
                MPI_collect         = 3,
                MPI_terminate       = 4
            };

            mcmpisim(typename mcthreadsim<Impl>::parameters_type const & p, boost::mpi::communicator const & c) 
                : mcthreadsim<Impl>(p)
                , next_check(8)
                , start_time(boost::posix_time::second_clock::local_time())
                , check_time(boost::posix_time::second_clock::local_time() + boost::posix_time::seconds(next_check))
                , communicator(c)
                , binnumber(p.value_or_default("binnumber", 2))
            {
                MPI_Errhandler_set(communicator, MPI_ERRORS_RETURN);
            }

            double fraction_completed() {
                assert(communicator.rank() == 0);
                double fraction = mcthreadsim<Impl>::fraction_completed();
                int action = MPI_get_fraction;
                boost::mpi::broadcast(communicator, action, 0);
                boost::mpi::reduce(communicator, mcthreadsim<Impl>::fraction_completed(), fraction, std::plus<double>(), 0);
                return fraction;
            }

            typename mcthreadsim<Impl>::results_type collect_results() const {
                assert(communicator.rank() == 0);
                return collect_results(mcthreadsim<Impl>::result_names());
            }

            typename mcthreadsim<Impl>::results_type collect_results(typename mcthreadsim<Impl>::result_names_type const & names) const {
                typename mcthreadsim<Impl>::results_type local_results = collect_local_results(names), partial_results;
                for(typename mcthreadsim<Impl>::results_type::iterator it = local_results.begin(); it != local_results.end(); ++it)
                    if (it->second->count() > 0) {
                        assert(it->first.size() < 255);
                        int action = MPI_collect;
                        boost::mpi::broadcast(communicator, action, 0);
                        {
                            char name[255];
                            std::strcpy(name, it->first.c_str());
                            boost::mpi::broadcast(communicator, name, 0);
                        }
                        it->second->reduce_master(partial_results, it->first, communicator, binnumber);
                    }
                return partial_results;
            }
            
            typename mcthreadsim<Impl>::results_type collect_local_results() const {
                return collect_local_results(mcthreadsim<Impl>::result_names());
            }

            typename mcthreadsim<Impl>::results_type collect_local_results(typename mcthreadsim<Impl>::result_names_type const & names) const {
                return mcthreadsim<Impl>::collect_results(names);
            }

            void terminate() const {
                if (communicator.rank() == 0) {
                    int action = MPI_terminate;
                    boost::mpi::broadcast(communicator, action, 0);
                }
            }

            void thread_callback(boost::function<bool ()> const & stop_callback) {
                if (communicator.rank() == 0) {
                    for (bool flag = false; !flag && !stop_callback(); ) {
                        usleep(0.1 * 1e6);
                        if (!flag && boost::posix_time::second_clock::local_time() > check_time) {
                            double fraction = fraction_completed();
                            flag = (fraction >= 1.);
                            next_check = std::min(2. * next_check, std::max(double(next_check), 0.8 * (boost::posix_time::second_clock::local_time() - start_time).total_seconds() / fraction * (1 - fraction)));
                            check_time = boost::posix_time::second_clock::local_time() + boost::posix_time::seconds(next_check);
                            std::cerr << std::fixed << std::setprecision(1) << fraction * 100 << "% done next check in " << next_check << "s" << std::endl;
                        }
                    }
                    mcthreadsim<Impl>::stop_flag = true;
                    int action = MPI_stop;
                    boost::mpi::broadcast(communicator, action, 0);
                } else
                    process_requests();
            }

            void process_requests() {
                assert(communicator.rank() > 0);
                while (true) {
                    int action;
                    boost::mpi::broadcast(communicator, action, 0);
                    switch (action) {
                        case MPI_get_fraction:
                            boost::mpi::reduce(communicator, mcthreadsim<Impl>::fraction_completed(), std::plus<double>(), 0);
                            break;
                        case MPI_collect:
                            {
                                char name[255];
                                boost::mpi::broadcast(communicator, name, 0);
                                typename mcthreadsim<Impl>::results_type local_results = mcthreadsim<Impl>::collect_results(typename result_names_type<mcthreadsim<Impl> >::type(1, name));
                                local_results.begin()->second->reduce_slave(communicator, binnumber);
                            }
                            break;
                        case MPI_stop:
                        case MPI_terminate:
                            mcthreadsim<Impl>::stop_flag = true;
                            return;
                    }
                }
            }

        private:
            int next_check;
            boost::posix_time::ptime start_time;
            boost::posix_time::ptime check_time;
            boost::mpi::communicator communicator;
            std::size_t binnumber;
    };
}

#endif
