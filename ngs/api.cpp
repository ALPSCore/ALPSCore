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

#include <alps/hdf5.hpp>

namespace alps {

    namespace detail {
        template<typename R, typename P> void save_results_impl(R const & results, P const & params, boost::filesystem::path const & filename, std::string const & path) {
            if (results.size()) {
                boost::filesystem::path original = filename.parent_path() / (filename.filename() + ".h5");
                boost::filesystem::path backup = filename.parent_path() / (filename.filename() + ".bak");
                if (boost::filesystem::exists(backup))
                    boost::filesystem::remove(backup);
                {
                    hdf5::oarchive ar(backup.file_string());
                    ar 
                        << make_pvp("/parameters", params)
                        << make_pvp(path, results)
                    ;
                }
                if (boost::filesystem::exists(original))
                    boost::filesystem::remove(original);
                boost::filesystem::rename(backup, original);
            }
        }
    }

    void save_results(mcresults const & results, mcparams const & params, boost::filesystem::path const & filename, std::string const & path) {
        detail::save_results_impl(results, params, filename, path);
    }

    void save_results(mcobservables const & observables, mcparams const & params, boost::filesystem::path const & filename, std::string const & path) {
        detail::save_results_impl(observables, params, filename, path);
    }

}
