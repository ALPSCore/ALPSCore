/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_HDF5_STD_VALARRAY_HPP
#define ALPS_HDF5_STD_VALARRAY_HPP

#include <alps/hdf5.hpp>
#include <alps/utilities/cast.hpp>

#include <type_traits>
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
                        if (!std::is_scalar<T>::value)
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
                    else if (extent.size() == 0 && !std::is_same<typename scalar_type<T>::type, T>::value)
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
                        if (!std::is_scalar<T>::value) {
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
