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

#include <alps/ngs/api.hpp>
#include <alps/ngs/hdf5.hpp>
#include <alps/ngs/signal.hpp>
#include <alps/ngs/scheduler/mcbase.hpp>

namespace alps {

    mcbase::mcbase(parameters_type const & p, std::size_t seed_offset)
        : params(p)
          // TODO: this ist not the best solution - any idea?
        , random(boost::mt19937((p["SEED"] | 42) + seed_offset), boost::uniform_real<>())
    {
        alps::ngs::signal::listen();
    }

    void mcbase::lock_data() {
        // TODO: set locked variable ...
    }

    void mcbase::unlock_data() {
        // TODO: set locked variable ...
    }

    void mcbase::lock_results() {
        // TODO: set locked variable ...
    }

    void mcbase::unlock_results() {
        // TODO: set locked variable ...
    }

    void mcbase::save(boost::filesystem::path const & filename) const {
        hdf5::archive ar(filename, "w");
        ar["/checkpoint"] << *this;
    }

    void mcbase::load(boost::filesystem::path const & filename) {
        hdf5::archive ar(filename);
        ar["/checkpoint"] >> *this;
    }

    void mcbase::save(alps::hdf5::archive & ar) const {
        ar["/parameters"] << params;
        ar["/simulation/realizations/0/clones/0/results"] << measurements;
    }

    void mcbase::load(alps::hdf5::archive & ar) {
        ar["/simulation/realizations/0/clones/0/results"] >> measurements;
    }

    bool mcbase::run(boost::function<bool ()> const & stop_callback) {
        do {
            lock_data();
            update();
            unlock_data();

            lock_results();
            lock_data();
            measure();
            unlock_data();
            unlock_results();
        } while(!stop_callback() && fraction_completed() < 1.);
        return !stop_callback();
    }

    #ifdef ALPS_HAVE_PYTHON
        bool mcbase::run(boost::python::object stop_callback) {
            return run(boost::bind(callback_wrapper, stop_callback));
        }
    #endif

    mcbase::result_names_type mcbase::result_names() const {
        result_names_type names;
        for(mcobservables::const_iterator it = measurements.begin(); it != measurements.end(); ++it)
            names.push_back(it->first);
        return names;
    }

    mcbase::result_names_type mcbase::unsaved_result_names() const {
        return result_names_type(); 
    }

    mcbase::results_type mcbase::collect_results() const {
        return collect_results(result_names());
    }

    mcbase::results_type mcbase::collect_results(result_names_type const & names) const {
        results_type partial_results;
        for(result_names_type::const_iterator it = names.begin(); it != names.end(); ++it)
            partial_results.insert(*it, mcresult(measurements[*it]));
        return partial_results;
    }

    mcbase::parameters_type & mcbase::get_params() {
        return params;
    }

    mcobservables & mcbase::get_measurements() {
        return measurements;
    }
    
    double mcbase::get_random() {
        return random();
    }

    #ifdef ALPS_HAVE_PYTHON
        bool mcbase::callback_wrapper(boost::python::object stop_callback) {
           return boost::python::call<bool>(stop_callback.ptr());
        }
    #endif
}
