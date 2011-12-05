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
#include <alps/ngs/base.hpp>

namespace alps {

    void base::save(std::string const & filename) const {
        hdf5::archive ar(filename, hdf5::archive::REPLACE);
        ar
            << make_pvp("/checkpoint", *this)
        ;
    }

    void base::load(std::string const & filename) {
        hdf5::archive ar(filename);
        ar 
            >> make_pvp("/checkpoint", *this)
        ;
    }

    void base::save(alps::param const & filename) const {
        save(filename.str());
    }

    void base::load(alps::param const & filename) {
        load(filename.str());
    }

    void base::save(alps::hdf5::archive & ar) const {
        ar
            << make_pvp("/parameters", params)
            << make_pvp("/simulation/realizations/0/clones/0/results", measurements)
        ;
    }

    void base::load(alps::hdf5::archive & ar) {
        ar 
            >> make_pvp("/simulation/realizations/0/clones/0/results", measurements)
        ;
    }

    bool base::run(boost::function<bool ()> const & stop_callback) {
        do {
            do_update();
            do_measurements();
        } while(!complete_callback(stop_callback));
        return !stop_callback();
    }

    base::result_names_type base::result_names() const {
        result_names_type names;
        for(mcobservables::const_iterator it = measurements.begin(); it != measurements.end(); ++it)
            names.push_back(it->first);
        return names;
    }

    base::result_names_type base::unsaved_result_names() const {
        return result_names_type(); 
    }

    base::results_type base::collect_results() const {
        return collect_results(result_names());
    }

    base::results_type base::collect_results(result_names_type const & names) const {
        results_type partial_results;
        for(result_names_type::const_iterator it = names.begin(); it != names.end(); ++it)
            partial_results.insert(*it, mcresult(measurements[*it]));
        return partial_results;
    }

    bool base::complete_callback(boost::function<bool ()> const & stop_callback) {
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

}
