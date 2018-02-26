/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_HDF5_BOOST_TUPLE
#define ALPS_HDF5_BOOST_TUPLE

#include <alps/hdf5/archive.hpp>
#include <alps/utilities/cast.hpp>
#include <alps/utilities/stringify.hpp>
#include <alps/utilities/remove_cvr.hpp>

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

            template <int N, typename T> struct save_helper<N, T, std::true_type> {
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

            template <int N, typename T> struct load_helper<N, T, std::true_type> {
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
            detail::save_helper<0, T0, typename std::is_same<T0, boost::tuples::null_type>::type>::apply(
                ar, ar.complete_path(path) + "/0", value
            );
            detail::save_helper<1, T1, typename std::is_same<T1, boost::tuples::null_type>::type>::apply(
                ar, ar.complete_path(path) + "/1", value
            );
            detail::save_helper<2, T2, typename std::is_same<T2, boost::tuples::null_type>::type>::apply(
                ar, ar.complete_path(path) + "/2", value
            );
            detail::save_helper<3, T3, typename std::is_same<T3, boost::tuples::null_type>::type>::apply(
                ar, ar.complete_path(path) + "/3", value
            );
            detail::save_helper<4, T4, typename std::is_same<T4, boost::tuples::null_type>::type>::apply(
                ar, ar.complete_path(path) + "/4", value
            );
            detail::save_helper<5, T5, typename std::is_same<T5, boost::tuples::null_type>::type>::apply(
                ar, ar.complete_path(path) + "/5", value
            );
            detail::save_helper<6, T6, typename std::is_same<T6, boost::tuples::null_type>::type>::apply(
                ar, ar.complete_path(path) + "/6", value
            );
            detail::save_helper<7, T7, typename std::is_same<T7, boost::tuples::null_type>::type>::apply(
                ar, ar.complete_path(path) + "/7", value
            );
            detail::save_helper<8, T8, typename std::is_same<T8, boost::tuples::null_type>::type>::apply(
                ar, ar.complete_path(path) + "/8", value
            );
            detail::save_helper<9, T9, typename std::is_same<T9, boost::tuples::null_type>::type>::apply(
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
            detail::load_helper<0, T0, typename std::is_same<T0, boost::tuples::null_type>::type>::apply(ar, ar.complete_path(path) + "/0", value);
            detail::load_helper<1, T1, typename std::is_same<T1, boost::tuples::null_type>::type>::apply(ar, ar.complete_path(path) + "/1", value);
            detail::load_helper<2, T2, typename std::is_same<T2, boost::tuples::null_type>::type>::apply(ar, ar.complete_path(path) + "/2", value);
            detail::load_helper<3, T3, typename std::is_same<T3, boost::tuples::null_type>::type>::apply(ar, ar.complete_path(path) + "/3", value);
            detail::load_helper<4, T4, typename std::is_same<T4, boost::tuples::null_type>::type>::apply(ar, ar.complete_path(path) + "/4", value);
            detail::load_helper<5, T5, typename std::is_same<T5, boost::tuples::null_type>::type>::apply(ar, ar.complete_path(path) + "/5", value);
            detail::load_helper<6, T6, typename std::is_same<T6, boost::tuples::null_type>::type>::apply(ar, ar.complete_path(path) + "/6", value);
            detail::load_helper<7, T7, typename std::is_same<T7, boost::tuples::null_type>::type>::apply(ar, ar.complete_path(path) + "/7", value);
            detail::load_helper<8, T8, typename std::is_same<T8, boost::tuples::null_type>::type>::apply(ar, ar.complete_path(path) + "/8", value);
            detail::load_helper<9, T9, typename std::is_same<T9, boost::tuples::null_type>::type>::apply(ar, ar.complete_path(path) + "/9", value);
        }
    }
}

#endif
