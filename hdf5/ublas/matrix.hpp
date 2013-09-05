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

#ifndef ALPS_NGS_HDF5_BOOST_NUMERIC_UBLAS_MATRIX_HPP
#define ALPS_NGS_HDF5_BOOST_NUMERIC_UBLAS_MATRIX_HPP

#include <alps/hdf5/archive.hpp>

#include <boost/numeric/ublas/matrix.hpp>

#include <iterator>

namespace alps {
    namespace hdf5 {

        template <typename T, typename F, typename A> struct scalar_type<boost::numeric::ublas::matrix<T, F, A> > {
            typedef typename scalar_type<typename boost::remove_reference<typename boost::remove_cv<T>::type>::type>::type type;
        };

        template <typename T, typename F, typename A> struct has_complex_elements<boost::numeric::ublas::matrix<T, F, A> >
            : public has_complex_elements<typename alps::detail::remove_cvr<T>::type>
        {};

        namespace detail {

            template<typename T, typename F, typename A> struct get_extent<boost::numeric::ublas::matrix<T, F, A> > {
                static std::vector<std::size_t> apply(boost::numeric::ublas::matrix<T, F, A> const & value) {
                    using alps::hdf5::get_extent;
                    std::vector<std::size_t> extent(2, value.size1());
                    extent[1] = value.size2();
                    if (value.size1() && value.size2()) {
                        std::vector<std::size_t> first(get_extent(value(0, 0)));
                        for (std::size_t i = 0; i < value.size1(); ++i)
                            for (std::size_t j = 0; j < value.size2(); ++j)  {
                                std::vector<std::size_t> size(get_extent(value(i, j)));
                                if (
                                       first.size() != size.size()
                                    || !std::equal(first.begin(), first.end(), size.begin())
                                )
                                    throw archive_error("no rectengual matrix" + ALPS_STACKTRACE);
                            }
                        std::copy(first.begin(), first.end(), std::back_inserter(extent));
                    }
                    return extent;
                }
            };

            template<typename T, typename F, typename A> struct set_extent<boost::numeric::ublas::matrix<T, F, A> > {
                static void apply(boost::numeric::ublas::matrix<T, F, A> & value, std::vector<std::size_t> const & size) {
                    using alps::hdf5::set_extent;
                    value.resize(size[0], size[1], false);
                    if (!is_continuous<T>::value && size.size() != 2)
                        for (std::size_t i = 0; i < value.size1(); ++i)
                            for (std::size_t j = 0; j < value.size2(); ++j)
                                set_extent(value(i, j), std::vector<std::size_t>(size.begin() + 2, size.end()));
                }
            };

            template<typename T, typename F, typename A> struct is_vectorizable<boost::numeric::ublas::matrix<T, F, A> > {
                static bool apply(boost::numeric::ublas::matrix<T, F, A> const & value) {
                    using alps::hdf5::get_extent;
                    using alps::hdf5::is_vectorizable;
                    if (!boost::is_scalar<typename boost::numeric::ublas::matrix<T, F, A>::value_type>::value) {
                        std::vector<std::size_t> size(get_extent(value(0, 0)));
                        for (std::size_t i = 0; i < value.size1(); ++i)
                            for (std::size_t j = 1; j < value.size2(); ++j)
                                if (!is_vectorizable(value(i, j)) || !std::equal(size.begin(), size.end(), get_extent(value(i, j)).begin()))
                                    return false;
                    }
                    return true;
                }
            };

            template<typename T, typename F, typename A> struct get_pointer<boost::numeric::ublas::matrix<T, F, A> > {
                static typename alps::hdf5::scalar_type<boost::numeric::ublas::matrix<T, F, A> >::type * apply(boost::numeric::ublas::matrix<T, F, A> & value) {
                    using alps::hdf5::get_pointer;
                    return get_pointer(value(0, 0));
                }
            };

            template<typename T, typename F, typename A> struct get_pointer<boost::numeric::ublas::matrix<T, F, A> const> {
                static typename alps::hdf5::scalar_type<boost::numeric::ublas::matrix<T, F, A> >::type const * apply(boost::numeric::ublas::matrix<T, F, A> const & value) {
                    using alps::hdf5::get_pointer;
                    return get_pointer(value(0, 0));
                }
            };

        }

        template <typename T, typename F, typename A> void save(
              archive & ar
            , std::string const & path
            , boost::numeric::ublas::matrix<T, F, A> const & value
            , std::vector<std::size_t> size = std::vector<std::size_t>()
            , std::vector<std::size_t> chunk = std::vector<std::size_t>()
            , std::vector<std::size_t> offset = std::vector<std::size_t>()
        ) {
            if (is_continuous<T>::value) {
                std::vector<std::size_t> extent(get_extent(value));
                std::copy(extent.begin(), extent.end(), std::back_inserter(size));
                std::copy(extent.begin(), extent.end(), std::back_inserter(chunk));
                std::fill_n(std::back_inserter(offset), extent.size(), 0);
                ar.write(path, get_pointer(value), size, chunk, offset);
            } else {
                throw wrong_type("invalid type" + ALPS_STACKTRACE);
            }
        }

        template <typename T, typename F, typename A> void load(
              archive & ar
            , std::string const & path
            , boost::numeric::ublas::matrix<T, F, A> & value
            , std::vector<std::size_t> chunk = std::vector<std::size_t>()
            , std::vector<std::size_t> offset = std::vector<std::size_t>()
        ) {
            if (ar.is_group(path))
                throw invalid_path("invalid path" + ALPS_STACKTRACE);
            else {
                std::vector<std::size_t> size(ar.extent(path));
                set_extent(value, std::vector<std::size_t>(size.begin() + chunk.size(), size.end()));
                if (is_continuous<T>::value) {
                    std::copy(size.begin(), size.end(), std::back_inserter(chunk));
                    std::fill_n(std::back_inserter(offset), size.size(), 0);
                    ar.read(path, get_pointer(value), chunk, offset);
                } else
                    throw invalid_path("invalid type" + ALPS_STACKTRACE);
            }
        }
    }
}

#endif
