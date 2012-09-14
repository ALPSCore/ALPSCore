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

#include <boost/filesystem/path.hpp>

namespace alps {

    mcbase::mcbase(parameters_type const & p, std::size_t seed_offset)
        : params(p)
          // TODO: this ist not the best solution - any idea?
        , random(boost::mt19937((p["SEED"] | 42) + seed_offset), boost::uniform_real<>())
        , fraction(0.)
        , check_duration(8.)
        , start_time_point(boost::chrono::high_resolution_clock::now())
        , last_check_time_point(boost::chrono::high_resolution_clock::now())
    {
        alps::ngs::signal::listen();
    }

    void mcbase::save(boost::filesystem::path const & filename) const {
        hdf5::archive ar(filename, "w");
        ar
            << make_pvp("/checkpoint", *this)
        ;
    }

    void mcbase::load(boost::filesystem::path const & filename) {
        hdf5::archive ar(filename);
        ar 
            >> make_pvp("/checkpoint", *this)
        ;
    }

    void mcbase::save(alps::hdf5::archive & ar) const {
        ar
            << make_pvp("/parameters", params)
            << make_pvp("/simulation/realizations/0/clones/0/results", measurements)
        ;
    }

    void mcbase::load(alps::hdf5::archive & ar) {
        ar 
            >> make_pvp("/simulation/realizations/0/clones/0/results", measurements)
        ;
    }

    bool mcbase::run(boost::function<bool ()> const & stop_callback) {
        do {
            update();
            measure();
        } while(!complete_callback(stop_callback));
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

    bool mcbase::complete_callback(boost::function<bool ()> const & stop_callback) {
        boost::chrono::high_resolution_clock::time_point now_time_point = boost::chrono::high_resolution_clock::now();
        if (now_time_point - last_check_time_point > check_duration) {
            fraction = fraction_completed();
            check_duration = boost::chrono::duration<double>(std::min(
                2. *  check_duration.count(),
                std::max(
                      double(check_duration.count())
                    , 0.8 * (1 - fraction) / fraction * boost::chrono::duration_cast<boost::chrono::duration<double> >(now_time_point - start_time_point).count()
                )
            ));
            last_check_time_point = now_time_point;
        }
        return (stop_callback() || fraction >= 1.);
    }

    #ifdef ALPS_HAVE_PYTHON
        bool mcbase::callback_wrapper(boost::python::object stop_callback) {
           return boost::python::call<bool>(stop_callback.ptr());
        }
    #endif
}
