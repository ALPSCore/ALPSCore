/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_HDF5_BOOST_ARRAY_HPP
#define ALPS_HDF5_BOOST_ARRAY_HPP

#include <alps/hdf5/archive.hpp>
#include <alps/utilities/cast.hpp>

#include <boost/array.hpp>

#include <vector>
#include <type_traits>
#include <iterator>
#include <algorithm>

namespace alps {
    namespace hdf5 {

        template<typename T, std::size_t N> struct scalar_type<boost::array<T, N> > {
            typedef typename scalar_type<typename boost::array<T, N>::value_type>::type type;
        };

        template<typename T, std::size_t N> struct is_continuous<boost::array<T, N> >
            : public is_continuous<T>
        {};
        template<typename T, std::size_t N> struct is_continuous<boost::array<T, N> const >
            : public is_continuous<T>
        {};

        template<typename T, std::size_t N> struct has_complex_elements<boost::array<T, N> >
            : public has_complex_elements<typename alps::detail::remove_cvr<typename boost::array<T, N>::value_type>::type>
        {};

        namespace detail {

            template<typename T, std::size_t N> struct get_extent<boost::array<T, N> > {
                static std::vector<std::size_t> apply(boost::array<T, N> const & value) {
                    using alps::hdf5::get_extent;
                    std::vector<std::size_t> result(1, value.size());
                    if (value.size()) {
                        std::vector<std::size_t> first(get_extent(value[0]));
                        std::copy(first.begin(), first.end(), std::back_inserter(result));
                    }
                    return result;
                }
            };

            template<typename T, std::size_t N> struct set_extent<boost::array<T, N> > {
                static void apply(boost::array<T, N> & value, std::vector<std::size_t> const & extent) {
                    using alps::hdf5::set_extent;
                    if (extent.size() > 1)
                        for(typename boost::array<T, N>::iterator it = value.begin(); it != value.end(); ++it)
                            set_extent(*it, std::vector<std::size_t>(extent.begin() + 1, extent.end()));
                    else if (extent.size() == 0 && !std::is_same<typename scalar_type<T>::type, T>::value)
                        throw archive_error("dimensions do not match" + ALPS_STACKTRACE);
                }
            };

            template<typename T, std::size_t N> struct is_vectorizable<boost::array<T, N> > {
                static bool apply(boost::array<T, N> const & value) {
                    using alps::hdf5::get_extent;
                    using alps::hdf5::is_vectorizable;
                    if (!is_continuous<boost::array<T, N> >::value) {
                        if (!is_vectorizable(value[0]))
                            return false;
                        std::vector<std::size_t> first(get_extent(value[0]));
                        for(typename boost::array<T, N>::const_iterator it = value.begin(); it != value.end(); ++it)
                            if (!is_vectorizable(*it))
                                return false;
                            else {
                                std::vector<std::size_t> size(get_extent(*it));
                                if (
                                       first.size() != size.size()
                                    || !std::equal(first.begin(), first.end(), size.begin())
                                )
                                    return false;
                            }
                    }
                    return true;
                }
            };

            template<typename T, std::size_t N> struct get_pointer<boost::array<T, N> > {
                static typename alps::hdf5::scalar_type<boost::array<T, N> >::type * apply(boost::array<T, N> & value) {
                    using alps::hdf5::get_pointer;
                    return get_pointer(value[0]);
                }
            };

            template<typename T, std::size_t N> struct get_pointer<boost::array<T, N> const > {
                static typename alps::hdf5::scalar_type<boost::array<T, N> >::type const * apply(boost::array<T, N> const & value) {
                    using alps::hdf5::get_pointer;
                    return get_pointer(value[0]);
                }
            };
        }

        template<typename T, std::size_t N> void save(
              archive & ar
            , std::string const & path
            , boost::array<T, N> const & value
            , std::vector<std::size_t> size = std::vector<std::size_t>()
            , std::vector<std::size_t> chunk = std::vector<std::size_t>()
            , std::vector<std::size_t> offset = std::vector<std::size_t>()
        ) {
            using alps::cast;
            if (ar.is_group(path))
                ar.delete_group(path);
            if (is_continuous<T>::value && value.size() == 0)
                ar.write(path, static_cast<typename scalar_type<boost::array<T, N> >::type const *>(NULL), std::vector<std::size_t>());
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
                for(typename boost::array<T, N>::const_iterator it = value.begin(); it != value.end(); ++it) {
                    offset.back() = it - value.begin();
                    save(ar, path, *it, size, chunk, offset);
                }
            } else {
                if (ar.is_data(path))
                    ar.delete_data(path);
                for(typename boost::array<T, N>::const_iterator it = value.begin(); it != value.end(); ++it)
                    save(ar, ar.complete_path(path) + "/" + cast<std::string>(it - value.begin()), *it);
            }
        }

        template<typename T, std::size_t N> void load(
              archive & ar
            , std::string const & path
            , boost::array<T, N> & value
            , std::vector<std::size_t> chunk = std::vector<std::size_t>()
            , std::vector<std::size_t> offset = std::vector<std::size_t>()
        ) {
            using alps::cast;
            if (ar.is_group(path)) {
                std::vector<std::string> children = ar.list_children(path);
                if (children.size() != N)
                    throw invalid_path("size does not match: " + path + ALPS_STACKTRACE);
                for (typename std::vector<std::string>::const_iterator it = children.begin(); it != children.end(); ++it)
                    load(ar, ar.complete_path(path) + "/" + *it, value[cast<std::size_t>(*it)]);
            } else {
                if (ar.is_complex(path) != has_complex_elements<T>::value)
                    throw archive_error("no complex value in archive" + ALPS_STACKTRACE);
                std::vector<std::size_t> size(ar.extent(path));
                if (size.size() > 0 && N != *(size.begin() + chunk.size()) && (is_continuous<T>::value || *(size.begin() + chunk.size()) > 0))
                    throw archive_error("dimensions do not match" + ALPS_STACKTRACE);
                if (is_continuous<T>::value) {
                    set_extent(value, std::vector<std::size_t>(size.begin() + chunk.size(), size.end()));
                    if (value.size()) {
                        std::copy(size.begin() + chunk.size(), size.end(), std::back_inserter(chunk));
                        std::fill_n(std::back_inserter(offset), size.size() - offset.size(), 0);
                        ar.read(path, get_pointer(value), chunk, offset);
                    }
                } else {
                    set_extent(value, std::vector<std::size_t>(1, *(size.begin() + chunk.size())));
                    chunk.push_back(1);
                    offset.push_back(0);
                    for(typename boost::array<T, N>::iterator it = value.begin(); it != value.end(); ++it) {
                        offset.back() = it - value.begin();
                        load(ar, path, *it, chunk, offset);
                    }
                }
            }
        }
    }
}

#endif
