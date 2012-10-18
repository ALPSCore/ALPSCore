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

#include <alps/ngs/hdf5.hpp>
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

    class ALPS_DECL mcbase_ng {

        private:

/* TODO:
create adobe pattern with mutex (mutex ->(shared_ptr) mutex_base ->(derived)mutex_derived<T>)
T is eigther mutex below or native mutex
base class has a virtual function to create a mutex
check_communication is called in the destructor of lock_guard and checks himself if the muteces are free (with locked)
*/

             struct mutex_type {
                typedef enum { DATA, RESULT } kind_type;

                mutex_type(mcbase_ng & arg1, kind_type arg2) : kind(arg2), target(arg1) {}

                operator const void*() { return guard.get(); }

                bool locked() const { return guard; }

                void lock() {
                    boost::shared_ptr<void> ptr;
                    switch (kind) {
                        case DATA:
                            ptr = target.get_data_guard();
                            break;
                        case RESULT:
                            ptr = target.get_result_guard();
                            break;
                        default:
                            throw std::logic_error("Invalid kind " + ALPS_STACKTRACE);
                    }
                    if (locked())
                        throw boost::lock_error();
                    target.data_guard_ptr = (guard = ptr);
                }

                void unlock() {
                    if (!locked())
                        throw boost::lock_error();
                    guard.reset();
                }
                
                kind_type kind;
                mcbase_ng & target;
                boost::shared_ptr<void> guard;
            };

        public:
        
            struct lock_guard_type {

                lock_guard_type(mutex_type & m)
                    : mutex(m)
                {
                    mutex.lock();
                }

                ~lock_guard_type() {
                    mutex.unlock();
                }
                
                mutex_type & mutex;
            };

            typedef enum { initialized, running, interrupted, finished } status_type;

            typedef alps::params parameters_type;
            typedef alps::mcresults results_type;
            typedef std::vector<std::string> result_names_type;

            mcbase_ng(parameters_type const & p, std::size_t seed_offset = 0)
                : data_mutex(*this, mutex_type::DATA)
                , result_mutex(*this, mutex_type::RESULT)
                , params(p)
                , m_status(initialized)
                  // TODO: this ist not the best solution - any idea?
                , m_random(boost::mt19937((p["SEED"] | 42) + seed_offset), boost::uniform_real<>())
            {
                alps::ngs::signal::listen();
            }

            virtual ~mcbase_ng() {}

            virtual double fraction_completed() const = 0;
        
            void save(boost::filesystem::path const & filename) const {
                hdf5::archive ar(filename, "w");
                ar["/checkpoint"] << *this;
            }

            void load(boost::filesystem::path const & filename) {
                hdf5::archive ar(filename);
                ar["/checkpoint"] >> *this;
            }

            virtual void save(alps::hdf5::archive & ar) const {
                lock_guard_type result_lock(result_mutex);
                ar["/parameters"] << params;
                ar["/simulation/realizations/0/clones/0/results"] << measurements;
            }

            // TODO: do we want to load the parameters?
            virtual void load(alps::hdf5::archive & ar) {
                lock_guard_type result_lock(result_mutex);
                ar["/simulation/realizations/0/clones/0/results"] >> measurements;
            }

            bool run(boost::function<bool ()> const & stop_callback) {
                do {
                    {
                        lock_guard_type data_lock(data_mutex);
                        update();
                    }
                    {
                        lock_guard_type data_lock(data_mutex);
                        lock_guard_type result_lock(result_mutex);
                        measure();
                    }
                } while(!stop_callback() && !work_done());
                set_status(finished); // TODO: remove this!
                return !stop_callback();
            }

            #ifdef ALPS_HAVE_PYTHON
                bool run(boost::python::object stop_callback) {
                    return run(boost::bind(callback_wrapper, stop_callback));
                }
            #endif

            result_names_type result_names() const {
                result_names_type names;
                for(mcobservables::const_iterator it = measurements.begin(); it != measurements.end(); ++it)
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
            parameters_type & get_params() { return params; }
            parameters_type const & get_params() const { return params; }

            // CHECK: how do we handle locks here? Do we need const/nonconst versions?
            #ifdef ALPS_NGS_USE_NEW_ALEA
                alea::accumulator_set & get_measurements() { return measurements; }
                alea::accumulator_set const & get_measurements() const { return measurements; }
            #else
                mcobservables & get_measurements() { return measurements; }
                mcobservables const & get_measurements() const { return measurements; }
            #endif

        
            // CHECK: how do we handle locks here? Do we need const/nonconst versions?
            double random() const { return m_random(); }
        
            // CHECK: how do we handle locks here? Do we need const/nonconst versions?
            virtual void check_communication() {}

            // CHECK: this is ugly, can we add a better hook to controlthread
            virtual status_type status() const {
                return m_status;
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

            // CHECK: this is ugly, can we add a better hook to controlthread
            virtual void set_status(status_type status) {
                m_status = status;
            }

            virtual boost::shared_ptr<void> get_data_guard() const {
                return boost::shared_ptr<void>(new noop_lock_guard(*this));
            }

            virtual boost::shared_ptr<void> get_result_guard() const {
                return boost::shared_ptr<void>(new noop_lock_guard(*this));
            }

            // CHECK: do we want to expose this to the derived class?
            mutex_type mutable data_mutex;
            mutex_type mutable result_mutex;

            parameters_type params; // rename to params

            #ifdef ALPS_NGS_USE_NEW_ALEA
                alea::accumulator_set measurements;
            #else
                mcobservables measurements;
            #endif

        private:
        
            struct noop_lock_guard {
                noop_lock_guard(mcbase_ng const & target) {
                    if (target.data_guard_ptr.expired() && target.result_guard_ptr.expired())
                        const_cast<mcbase_ng &>(target).check_communication();
                }
            };

            #ifdef ALPS_HAVE_PYTHON
                static bool callback_wrapper(boost::python::object stop_callback) {
                   return boost::python::call<bool>(stop_callback.ptr());
                }
            #endif

            status_type m_status;

            boost::variate_generator<boost::mt19937, boost::uniform_real<> > mutable m_random;

            boost::weak_ptr<void> data_guard_ptr;
            boost::weak_ptr<void> result_guard_ptr;
    };
}

#endif
