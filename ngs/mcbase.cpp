/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2011 by Lukas Gamper <gamperl@gmail.com>                   *
 *                           Matthias Troyer <troyer@comp-phys.org>                *
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
#include <alps/ngs/mcbase.hpp>

#include <alps/hdf5.hpp>
#include <alps/alea/observableset.h>

namespace alps {

    void mcbase::save(boost::filesystem::path const & path) const {
        save_results(results, params, path, "/simulation/realizations/0/clones/0/results");
    }

    void mcbase::load(boost::filesystem::path const & path) {
        hdf5::iarchive ar(path.file_string() + ".h5");
        ar >> make_pvp("/simulation/realizations/0/clones/0/results", results);
    }

    bool mcbase::run(boost::function<bool ()> const & stop_callback) {
        do {
            do_update();
            do_measurements();
        } while(!complete_callback(stop_callback));
        return !stop_callback();
    }

    mcbase::result_names_type mcbase::result_names() const {
        result_names_type names;
        for(mcobservables::const_iterator it = results.begin(); it != results.end(); ++it)
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
            partial_results.insert(*it, mcresult(results[*it]));
        return partial_results;
    }

    bool mcbase::complete_callback(boost::function<bool ()> const & stop_callback) {
        if (boost::posix_time::second_clock::local_time() > check_time) {
            fraction = fraction_completed();
            next_check = std::min(
                2. * next_check, 
                std::max(
                      double(next_check)
                    , 0.8 * (boost::posix_time::second_clock::local_time() - start_time).total_seconds() / fraction * (1 - fraction)
                )
            );
           check_time = boost::posix_time::second_clock::local_time() + boost::posix_time::seconds(next_check);
        }
        return (stop_callback() || fraction >= 1.);
    }

}
