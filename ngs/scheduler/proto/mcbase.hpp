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

#ifndef ALPS_NGS_SCHEDULER_MCBASE_NG_HPP
#define ALPS_NGS_SCHEDULER_MCBASE_NG_HPP

#include <alps/hdf5/archive.hpp>
#include <alps/ngs/mutex.hpp>
#include <alps/ngs/config.hpp>
#include <alps/ngs/signal.hpp>
#include <alps/ngs/params.hpp>
#include <alps/ngs/mcresults.hpp> // TODO: replace by new alea
#include <alps/ngs/mcobservables.hpp> // TODO: replace by new alea
#include <alps/ngs/thread_exceptions.hpp>

#ifdef ALPS_HAVE_PYTHON
    #include <alps/ngs/boost_python.hpp>
#endif

#include <alps/random/mersenne_twister.hpp>

#include <boost/function.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>

#include <vector>
#include <string>

namespace alps {

    class mcbase_ng {
        public:
            // #ifdef ALPS_NGS_USE_NEW_ALEA
            //     typedef accumulator::accumulator_set observables_type;
            // #else
                typedef mcobservables observables_type;
            // #endif
        private:

            struct lock_guard_impl : boost::noncopyable {

                lock_guard_impl(mutex & arg1, mcbase_ng & arg2)
                    : mtx(arg1)
                    , sim(arg2)
                {
                    mtx.lock();
                }

                ~lock_guard_impl() {
                    mtx.unlock();

                    // TODO: this is ugly, can we avoid that?
                    if (!sim.data_mutex.locked() && !sim.result_mutex.locked())
                        sim.on_unlock();
                }
                
                mutex & mtx;
                mcbase_ng & sim;
            };

        public:
        
            typedef boost::shared_ptr<lock_guard_impl> lock_guard;

            typedef enum { initialized, running, interrupted, finished } status_type;

            typedef alps::params parameters_type;
            typedef alps::mcresults results_type;
            typedef std::vector<std::string> result_names_type;

            mcbase_ng(parameters_type const & p, std::size_t seed_offset = 0)
                  // TODO: this ist not the best solution - any idea?
                : random(boost::mt19937((p["SEED"] | 42) + seed_offset), boost::uniform_real<>())
                , params(p)
                , data_mutex(new noop_lockable())
                , result_mutex(new noop_lockable())
                , m_status(initialized)
            {
                alps::ngs::signal::listen();
            }

            virtual ~mcbase_ng() {}

            virtual double fraction_completed() const = 0;
        
            void save(boost::filesystem::path const & filename) const {
                hdf5::archive ar(filename.string() + file_suffix(), "w");
                ar["/checkpoint"] << *this;
            }

            void load(boost::filesystem::path const & filename) {
                hdf5::archive ar(filename.string() + file_suffix());
                ar["/checkpoint"] >> *this;
            }

            virtual void save(alps::hdf5::archive & ar) const {
                lock_guard result_lock(get_result_lock());
                ar["/parameters"] << params;
                ar["/simulation/realizations/0/clones/0/results"] << measurements;
            }

            // TODO: do we want to load the parameters?
            virtual void load(alps::hdf5::archive & ar) {
                lock_guard result_lock(get_result_lock());
                ar["/simulation/realizations/0/clones/0/results"] >> measurements;
            }

            virtual bool run(
                  boost::function<bool ()> const & stop_callback
            ) {
                return run(stop_callback, boost::function<void (double)>());
            }

            virtual bool run(
                  boost::function<bool ()> const & stop_callback
                , boost::function<void (double)> const & progress_callback
            ) {
                do {
                    {
                        lock_guard data_lock(get_data_lock());
                        update();
                    }
                    {
                        lock_guard data_lock(get_data_lock());
                        lock_guard result_lock(get_result_lock());
                        measure();
                    }
                    if (progress_callback)
                        progress_callback(fraction_completed());
                } while(!stop_callback() && !work_done());
                set_status(finished); // TODO: remove this!
                return !stop_callback();
            }

            #ifdef ALPS_HAVE_PYTHON
                bool run(
                    boost::python::object stop_callback
                ) {
                    return run(boost::bind(callback_wrapper, stop_callback));
                }
            #endif

            result_names_type result_names() const {
                result_names_type names;
                
                for(observables_type::const_iterator it = measurements.begin(); it != measurements.end(); ++it)
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
                    // CHECK: this is ugly make measurements[*it]
                    partial_results.insert(*it, mcresult(measurements[*it]));
                return partial_results;
            }

            // CHECK: how do we handle locks here? Do we need const/nonconst versions?
            double get_random() const { return random(); }

            // CHECK: how do we handle locks here? Do we need const/nonconst versions?
            parameters_type & get_params() { return params; }
            parameters_type const & get_params() const { return params; }

            // CHECK: how do we handle locks here? Do we need const/nonconst versions?
            observables_type & get_measurements() { return measurements; }
            observables_type const & get_measurements() const { return measurements; }

            // CHECK: how do we handle locks here? Do we need const/nonconst versions?
            virtual void check_communication() {}

            // CHECK: this is ugly, can we add a better hook to controlthread
            virtual status_type status() const {
                return m_status;
            }
        
            virtual std::string file_suffix() const {
                return "";
            }

        protected:

            template<typename T> class atomic {
                public:

                    atomic() {}
                    atomic(T const & v): value(v) {}
                    atomic(atomic<T> const & v): value(v.value) {}

                    atomic<T> & operator=(T const & v) {
                        value = v;
                        return *this;
                    }

                    operator T() const { return value; }

                private:

                    T value;
            };

            virtual void update() = 0;

            virtual void measure() = 0;
        
            // CHECK: this is ugly, can we add this check to run and remove it in mpi?
            virtual bool work_done() {
                return fraction_completed() >= 1.;
            }
        
            // CHECK: this is ugly, make this nicer!
            virtual void on_unlock() {
                check_communication();
            }

            // CHECK: this is ugly, can we add a better hook to controlthread
            virtual void set_status(status_type status) {
                m_status = status;
            }

            lock_guard get_data_lock() const {
                return lock_guard(new lock_guard_impl(data_mutex, const_cast<mcbase_ng &>(*this)));
            }

            lock_guard get_result_lock() const {
                return lock_guard(new lock_guard_impl(result_mutex, const_cast<mcbase_ng &>(*this)));
            }

            // CHECK: do we want to expose this to the derived class?
            // CHECK: how do we handle locks here?
            boost::variate_generator<boost::mt19937, boost::uniform_real<> > mutable random;

            parameters_type params; // rename to params

            observables_type measurements;

            // CHECK: do we want to expose this to the derived class? virtual functions do not work form base constructor
            mutex mutable data_mutex;
            mutex mutable result_mutex;

        private:
        
            #ifdef ALPS_HAVE_PYTHON
                static bool callback_wrapper(boost::python::object stop_callback) {
                   return boost::python::call<bool>(stop_callback.ptr());
                }
            #endif

            status_type m_status;
    };
}

#endif
