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

#ifndef ALPS_NGS_HDF5_STD_VALARRAY_HPP
#define ALPS_NGS_HDF5_STD_VALARRAY_HPP

#include <alps/hdf5.hpp>
#include <alps/ngs/cast.hpp>

#include <valarray>
#include <iterator>
#include <algorithm>

namespace alps {
    namespace hdf5 {

        template<typename T> struct scalar_type<std::valarray<T> > {
            typedef typename scalar_type<T>::type type;
        };

        template<typename T> struct is_content_continuous<std::valarray<T> >
            : public is_continuous<T> 
        {};

        template<typename T> struct has_complex_elements<std::valarray<T> > 
            : public has_complex_elements<typename alps::detail::remove_cvr<T>::type>
        {};

        namespace detail {

            template<typename T> struct get_extent<std::valarray<T> > {
                static std::vector<std::size_t> apply(std::valarray<T> const & value) {
                    using alps::hdf5::get_extent;
                    std::vector<std::size_t> result(1, value.size());
                    if (value.size()) {
                        std::vector<std::size_t> extent(get_extent(const_cast<std::valarray<T> &>(value)[0]));
                        if (!boost::is_scalar<T>::value)
                            for (std::size_t i = 1; i < value.size(); ++i)
                                if (!std::equal(extent.begin(), extent.end(), get_extent(const_cast<std::valarray<T> &>(value)[i]).begin()))
                                    throw archive_error("no rectengual matrix" + ALPS_STACKTRACE);
                        std::copy(extent.begin(), extent.end(), std::back_inserter(result));
                    }
                    return result;
                }
            };

            template<typename T> struct set_extent<std::valarray<T> > {
                static void apply(std::valarray<T> & value, std::vector<std::size_t> const & extent) {
                    using alps::hdf5::set_extent;
                    value.resize(extent[0]);
                    if (extent.size() > 1)
                        for(std::size_t i = 0; i < value.size(); ++i)
                            set_extent(value[i], std::vector<std::size_t>(extent.begin() + 1, extent.end()));
                    else if (extent.size() == 0 && !boost::is_same<typename scalar_type<T>::type, T>::value)
                        throw archive_error("dimensions do not match" + ALPS_STACKTRACE);
                }
            };

            template<typename T> struct is_vectorizable<std::valarray<T> > {
                static bool apply(std::valarray<T> const & value) {
                    using alps::hdf5::get_extent;
                    using alps::hdf5::is_vectorizable;
                    if (value.size()) {
                        if (!is_vectorizable(const_cast<std::valarray<T> &>(value)[0]))
                            return false;
                        std::vector<std::size_t> first(get_extent(const_cast<std::valarray<T> &>(value)[0]));
                        if (!boost::is_scalar<T>::value) {
                            for(std::size_t i = 0; i < value.size(); ++i)
                                if (!is_vectorizable(const_cast<std::valarray<T> &>(value)[i])) {
                                    return false;
                                } else {
                                    std::vector<std::size_t> size(get_extent(const_cast<std::valarray<T> &>(value)[i]));
                                    if (
                                           first.size() != size.size() 
                                        || !std::equal(first.begin(), first.end(), size.begin())
                                    ) {
                                        return false;
                                    }
                                }
                        }
                    }
                    return true;
                }
            };

            template<typename T> struct get_pointer<std::valarray<T> > {
                static typename alps::hdf5::scalar_type<std::valarray<T> >::type * apply(std::valarray<T> & value) {
                    using alps::hdf5::get_pointer;
                    return get_pointer(value[0]);
                }
            };

            template<typename T> struct get_pointer<std::valarray<T> const> {
                static typename alps::hdf5::scalar_type<std::valarray<T> >::type const * apply(std::valarray<T> const & value) {
                    using alps::hdf5::get_pointer;
                    return get_pointer(const_cast<std::valarray<T> &>(value)[0]);
                }
            };
        }

        template<typename T> void save(
              archive & ar
            , std::string const & path
            , std::valarray<T> const & value
            , std::vector<std::size_t> size = std::vector<std::size_t>()
            , std::vector<std::size_t> chunk = std::vector<std::size_t>()
            , std::vector<std::size_t> offset = std::vector<std::size_t>()
        ) {
            if (ar.is_group(path))
                ar.delete_group(path);
            if (is_continuous<T>::value && value.size() == 0)
                ar.write(path, static_cast<typename scalar_type<std::valarray<T> >::type const *>(NULL), std::vector<std::size_t>());
            else if (is_continuous<T>::value) {
                std::vector<std::size_t> extent(get_extent(value));
                std::copy(extent.begin(), extent.end(), std::back_inserter(size));
                std::copy(extent.begin(), extent.end(), std::back_inserter(chunk));
                std::fill_n(std::back_inserter(offset), extent.size(), 0);
                ar.write(path, get_pointer(value), size, chunk, offset);
            } else if (value.size() == 0)
                ar.write(path, static_cast<int const *>(NULL), std::vector<std::size_t>());
            else if (is_vectorizable(value)) {
                size.push_back(value.size());
                chunk.push_back(1);
                offset.push_back(0);
                for(std::size_t i = 0; i < value.size(); ++i) {
                    offset.back() = i;
                    save(ar, path, const_cast<std::valarray<T> &>(value)[i], size, chunk, offset);
                }
            } else {
                if (ar.is_data(path))
                    ar.delete_data(path);
                for(std::size_t i = 0; i < value.size(); ++i)
                    save(ar, ar.complete_path(path) + "/" + cast<std::string>(i), const_cast<std::valarray<T> &>(value)[i]);
            }
        }

        template<typename T> void load(
              archive & ar
            , std::string const & path
            , std::valarray<T> & value
            , std::vector<std::size_t> chunk = std::vector<std::size_t>()
            , std::vector<std::size_t> offset = std::vector<std::size_t>()
        ) {
            if (ar.is_group(path)) {
                std::vector<std::string> children = ar.list_children(path);
                value.resize(children.size());
                for (typename std::vector<std::string>::const_iterator it = children.begin(); it != children.end(); ++it)
                    load(ar, ar.complete_path(path) + "/" + *it, value[cast<std::size_t>(*it)]);
            } else {
                if (ar.is_complex(path) != has_complex_elements<T>::value)
                    throw archive_error("no complex value in archive" + ALPS_STACKTRACE);
                std::vector<std::size_t> size(ar.extent(path));
                if (is_continuous<T>::value) {
                    set_extent(value, std::vector<std::size_t>(size.begin() + chunk.size(), size.end()));
                    std::copy(size.begin() + chunk.size(), size.end(), std::back_inserter(chunk));
                    std::fill_n(std::back_inserter(offset), size.size() - offset.size(), 0);
                    ar.read(path, get_pointer(value), chunk, offset);
                } else {
                    set_extent(value, std::vector<std::size_t>(1, *(size.begin() + chunk.size())));
                    chunk.push_back(1);
                    offset.push_back(0);
                    for(std::size_t i = 0; i < value.size(); ++i) {
                        offset.back() = i;
                        load(ar, path, value[i], chunk, offset);
                    }
                }
            }
        }
    }
}

#endif
