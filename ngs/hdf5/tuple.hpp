/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2012 by Lukas Gamper <gamperl@gmail.com>                   *
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

#ifndef ALPS_NGS_HDF5_BOOST_TUPLE
#define ALPS_NGS_HDF5_BOOST_TUPLE

#include <alps/hdf5/archive.hpp>
#include <alps/ngs/cast.hpp>
#include <alps/ngs/stringify.hpp>
#include <alps/ngs/detail/remove_cvr.hpp>

#include <boost/tuple/tuple.hpp>

#include <utility>

namespace alps {
    namespace hdf5 {

        namespace detail {

            template <int N, typename T, typename E> struct save_helper {
                template <
                      typename A, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9
                > static void apply(
                      A & ar
                    , std::string const & path
                    , boost::tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> const & value
                ) {
                    using boost::get;
                    save(ar, path, get<N>(value));
                    if (has_complex_elements<typename alps::detail::remove_cvr<T>::type>::value)
                        ar.set_complex(path);
                }
            };

            template <int N, typename T> struct save_helper<N, T, boost::true_type> {
                template <
                      typename A, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9
                > static void apply(
                      A &
                    , std::string const &
                    , boost::tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> const &
                ) {}
            };

            template <int N, typename T, class E> struct load_helper {
                template <
                      typename A, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9
                > static void apply(
                      A & ar
                    , std::string const & path
                    , boost::tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> & value
                ) {
                    using boost::get;
                    load(ar, path, get<N>(value));
                }
            };

            template <int N, typename T> struct load_helper<N, T, boost::true_type> {
                template <
                      typename A, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9
                > static void apply(
                      A &
                    , std::string const &
                    , boost::tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> &
                ) {}
            };
        }

        template <
            typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9
        > void save(
              archive & ar
            , std::string const & path
            , boost::tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> const & value
            , std::vector<std::size_t> size = std::vector<std::size_t>()
            , std::vector<std::size_t> chunk = std::vector<std::size_t>()
            , std::vector<std::size_t> offset = std::vector<std::size_t>()
        ) {
            detail::save_helper<0, T0, typename boost::is_same<T0, boost::tuples::null_type>::type>::apply(
                ar, ar.complete_path(path) + "/0", value
            );
            detail::save_helper<1, T1, typename boost::is_same<T1, boost::tuples::null_type>::type>::apply(
                ar, ar.complete_path(path) + "/1", value
            );
            detail::save_helper<2, T2, typename boost::is_same<T2, boost::tuples::null_type>::type>::apply(
                ar, ar.complete_path(path) + "/2", value
            );
            detail::save_helper<3, T3, typename boost::is_same<T3, boost::tuples::null_type>::type>::apply(
                ar, ar.complete_path(path) + "/3", value
            );
            detail::save_helper<4, T4, typename boost::is_same<T4, boost::tuples::null_type>::type>::apply(
                ar, ar.complete_path(path) + "/4", value
            );
            detail::save_helper<5, T5, typename boost::is_same<T5, boost::tuples::null_type>::type>::apply(
                ar, ar.complete_path(path) + "/5", value
            );
            detail::save_helper<6, T6, typename boost::is_same<T6, boost::tuples::null_type>::type>::apply(
                ar, ar.complete_path(path) + "/6", value
            );
            detail::save_helper<7, T7, typename boost::is_same<T7, boost::tuples::null_type>::type>::apply(
                ar, ar.complete_path(path) + "/7", value
            );
            detail::save_helper<8, T8, typename boost::is_same<T8, boost::tuples::null_type>::type>::apply(
                ar, ar.complete_path(path) + "/8", value
            );
            detail::save_helper<9, T9, typename boost::is_same<T9, boost::tuples::null_type>::type>::apply(
                ar, ar.complete_path(path) + "/9", value
            );
        }

        template <
            typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9
        > void load(
              archive & ar
            , std::string const & path
            , boost::tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> & value
            , std::vector<std::size_t> chunk = std::vector<std::size_t>()
            , std::vector<std::size_t> offset = std::vector<std::size_t>()
        ) {
            detail::load_helper<0, T0, typename boost::is_same<T0, boost::tuples::null_type>::type>::apply(ar, ar.complete_path(path) + "/0", value);
            detail::load_helper<1, T1, typename boost::is_same<T1, boost::tuples::null_type>::type>::apply(ar, ar.complete_path(path) + "/1", value);
            detail::load_helper<2, T2, typename boost::is_same<T2, boost::tuples::null_type>::type>::apply(ar, ar.complete_path(path) + "/2", value);
            detail::load_helper<3, T3, typename boost::is_same<T3, boost::tuples::null_type>::type>::apply(ar, ar.complete_path(path) + "/3", value);
            detail::load_helper<4, T4, typename boost::is_same<T4, boost::tuples::null_type>::type>::apply(ar, ar.complete_path(path) + "/4", value);
            detail::load_helper<5, T5, typename boost::is_same<T5, boost::tuples::null_type>::type>::apply(ar, ar.complete_path(path) + "/5", value);
            detail::load_helper<6, T6, typename boost::is_same<T6, boost::tuples::null_type>::type>::apply(ar, ar.complete_path(path) + "/6", value);  
            detail::load_helper<7, T7, typename boost::is_same<T7, boost::tuples::null_type>::type>::apply(ar, ar.complete_path(path) + "/7", value);
            detail::load_helper<8, T8, typename boost::is_same<T8, boost::tuples::null_type>::type>::apply(ar, ar.complete_path(path) + "/8", value);
            detail::load_helper<9, T9, typename boost::is_same<T9, boost::tuples::null_type>::type>::apply(ar, ar.complete_path(path) + "/9", value);
        }
    }
}

#endif
