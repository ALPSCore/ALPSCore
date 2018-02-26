
/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_HDF5_STD_VECTOR_HPP
#define ALPS_HDF5_STD_VECTOR_HPP

#include <alps/hdf5/archive.hpp>
#include <alps/utilities/cast.hpp>

#include <type_traits>
#include <vector>
#include <iterator>
#include <algorithm>

namespace alps {
    namespace hdf5 {

        template<typename T, typename A> struct scalar_type<std::vector<T, A> > {
            typedef typename scalar_type<typename std::vector<T, A>::value_type>::type type;
        };

        template<typename T, typename A> struct is_content_continuous<std::vector<T, A> >
            : public is_continuous<T>
        {};
        template<typename A> struct is_content_continuous<std::vector<bool, A> >
            : public std::false_type
        {};

        template<typename T, typename A> struct has_complex_elements<std::vector<T, A> >
            : public has_complex_elements<typename alps::detail::remove_cvr<typename std::vector<T, A>::value_type>::type>
        {};

        namespace detail {

            template<typename T, typename A> struct get_extent<std::vector<T, A> > {
                static std::vector<std::size_t> apply(std::vector<T, A> const & value) {
                    using alps::hdf5::get_extent;
                    std::vector<std::size_t> result(1, value.size());
                    if (value.size()) {
                        std::vector<std::size_t> first(get_extent(value[0]));
                        if (!std::is_scalar<typename std::vector<T, A>::value_type>::value)
                            for(typename std::vector<T, A>::const_iterator it = value.begin() + 1; it != value.end(); ++it) {
                                std::vector<std::size_t> size(get_extent(*it));
                                if (
                                       first.size() != size.size()
                                    || !std::equal(first.begin(), first.end(), size.begin())
                                )
                                    throw archive_error("no rectangular matrix" + ALPS_STACKTRACE);
                            }
                        std::copy(first.begin(), first.end(), std::back_inserter(result));
                    }
                    return result;
                }
            };

            template<typename T, typename A> struct set_extent<std::vector<T, A> > {
                static void apply(std::vector<T, A> & value, std::vector<std::size_t> const & extent) {
                    using alps::hdf5::set_extent;
                    value.resize(extent[0]);
                    if (extent.size() > 1)
                        for(typename std::vector<T, A>::iterator it = value.begin(); it != value.end(); ++it)
                            set_extent(*it, std::vector<std::size_t>(extent.begin() + 1, extent.end()));
                    else if (extent.size() == 1 && (
                           (!std::is_enum<T>::value && !std::is_same<typename scalar_type<T>::type, T>::value)
                        || (std::is_enum<T>::value && is_continuous<T>::value && sizeof(T) != sizeof(typename scalar_type<T>::type))
                    ))
                        throw archive_error("dimensions do not match" + ALPS_STACKTRACE);
                }
            };

            template<typename A> struct set_extent<std::vector<bool, A> > {
                static void apply(std::vector<bool, A> & value, std::vector<std::size_t> const & extent) {
                    if (extent.size() != 1)
                        throw archive_error("dimensions do not match" + ALPS_STACKTRACE);
                    value.resize(extent[0]);
                }
            };

            template<typename T, typename A> struct is_vectorizable<std::vector<T, A> > {
                static bool apply(std::vector<T, A> const & value) {
                    using alps::hdf5::get_extent;
                    using alps::hdf5::is_vectorizable;
                    if (value.size()) {
                        if (!is_vectorizable(value[0]))
                            return false;
                        std::vector<std::size_t> first(get_extent(value[0]));
                        if (!std::is_scalar<typename std::vector<T, A>::value_type>::value) {
                            for(typename std::vector<T, A>::const_iterator it = value.begin(); it != value.end(); ++it)
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
                    }
                    return true;
                }
            };

            template<typename A> struct is_vectorizable<std::vector<bool, A> > {
                static bool apply(std::vector<bool, A> const & value) {
                    return true;
                }
            };

            template<typename T, typename A> struct get_pointer<std::vector<T, A> > {
                static typename alps::hdf5::scalar_type<std::vector<T, A> >::type * apply(std::vector<T, A> & value) {
                    using alps::hdf5::get_pointer;
                    return get_pointer(value[0]);
                }
            };

            template<typename T, typename A> struct get_pointer<std::vector<T, A> const> {
                static typename alps::hdf5::scalar_type<std::vector<T, A> >::type const * apply(std::vector<T, A> const & value) {
                    using alps::hdf5::get_pointer;
                    return get_pointer(value[0]);
                }
            };

            template<typename A> struct get_pointer<std::vector<bool, A> > {
                static typename alps::hdf5::scalar_type<std::vector<bool, A> >::type * apply(std::vector<bool, A> & value) {
                    throw archive_error("std::vector<bool, A>[0] cannot be dereferenced" + ALPS_STACKTRACE);
                    return NULL;
                }
            };

            template<typename A> struct get_pointer<std::vector<bool, A> const> {
                static typename alps::hdf5::scalar_type<std::vector<bool, A> >::type const * apply(std::vector<bool, A> const & value) {
                    throw archive_error("std::vector<bool>[0] cannot be dereferenced" + ALPS_STACKTRACE);
                    return NULL;
                }
            };

        }


        template<typename T, typename A> void save(
              archive & ar
            , std::string const & path
            , std::vector<T, A> const & value
            , std::vector<std::size_t> size = std::vector<std::size_t>()
            , std::vector<std::size_t> chunk = std::vector<std::size_t>()
            , std::vector<std::size_t> offset = std::vector<std::size_t>()
        ) {
            using alps::cast;
            if (ar.is_group(path))
                ar.delete_group(path);
            if (is_continuous<T>::value && value.size() == 0)
                ar.write(path, static_cast<typename scalar_type<std::vector<T, A> >::type const *>(NULL), std::vector<std::size_t>());
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
                for(typename std::vector<T, A>::const_iterator it = value.begin(); it != value.end(); ++it) {
                    offset.back() = it - value.begin();
                    save(ar, path, *it, size, chunk, offset);
                }
            } else {
                if (path.find_last_of('@') == std::string::npos && ar.is_data(path))
                    ar.delete_data(path);
                else if (path.find_last_of('@') != std::string::npos && ar.is_attribute(path))
                    ar.delete_attribute(path);
                for(typename std::vector<T, A>::const_iterator it = value.begin(); it != value.end(); ++it)
                    save(ar, ar.complete_path(path) + "/" + cast<std::string>(it - value.begin()), *it);
            }
        }

        template<typename A> void save(
              archive & ar
            , std::string const & path
            , std::vector<bool, A> const & value
            , std::vector<std::size_t> size = std::vector<std::size_t>()
            , std::vector<std::size_t> chunk = std::vector<std::size_t>()
            , std::vector<std::size_t> offset = std::vector<std::size_t>()
        ) {
            if (ar.is_group(path))
                ar.delete_group(path);
            if (value.size() == 0)
                ar.write(path, static_cast<bool const *>(NULL), std::vector<std::size_t>());
            else {
                size.push_back(value.size());
                chunk.push_back(1);
                offset.push_back(0);
                for(typename std::vector<bool, A>::const_iterator it = value.begin(); it != value.end(); ++it) {
                    offset.back() = it - value.begin();
                    bool const elem = *it;
                    ar.write(path, &elem, size, chunk, offset);
                }
            }
        }

        template<typename T, typename A> void load(
              archive & ar
            , std::string const & path
            , std::vector<T, A> & value
            , std::vector<std::size_t> chunk = std::vector<std::size_t>()
            , std::vector<std::size_t> offset = std::vector<std::size_t>()
        ) {
            using alps::cast;
            if (ar.is_group(path)) {
                std::vector<std::string> children = ar.list_children(path);
                value.resize(children.size());
                for (typename std::vector<std::string>::const_iterator it = children.begin(); it != children.end(); ++it)
                   load(ar, ar.complete_path(path) + "/" + *it, value[cast<std::size_t>(*it)]);
            } else {
                if (ar.is_complex(path) != has_complex_elements<T>::value)
                    throw archive_error("no complex value in archive" + ALPS_STACKTRACE);
                std::vector<std::size_t> size(ar.extent(path));
                if (size.size() == 0)
                    throw archive_error("invalid dimensions" + ALPS_STACKTRACE);
                else if (size[0] == 0)
                    value.resize(0);
                else if (is_continuous<T>::value) {
                    set_extent(value, std::vector<std::size_t>(size.begin() + chunk.size(), size.end()));
                    if (value.size()) {
                        std::copy(size.begin() + chunk.size(), size.end(), std::back_inserter(chunk));
                        std::fill_n(std::back_inserter(offset), size.size() - offset.size(), 0);
                        ar.read(path, get_pointer(value), chunk, offset);
                    }
                } else {
                    value.resize(*(size.begin() + chunk.size()));
                    chunk.push_back(1);
                    offset.push_back(0);
                    for(typename std::vector<T, A>::iterator it = value.begin(); it != value.end(); ++it) {
                        offset.back() = it - value.begin();
                        load(ar, path, *it, chunk, offset);
                    }
                }
            }
        }

        template<typename A> void load(
              archive & ar
            , std::string const & path
            , std::vector<bool, A> & value
            , std::vector<std::size_t> chunk = std::vector<std::size_t>()
            , std::vector<std::size_t> offset = std::vector<std::size_t>()
        ) {
            if (ar.is_group(path))
                throw archive_error("invalid dimensions" + ALPS_STACKTRACE);
            else {
                if (ar.is_complex(path))
                    throw archive_error("no complex value in archive" + ALPS_STACKTRACE);
                std::vector<std::size_t> size(ar.extent(path));
                if (size.size() == 0)
                    throw archive_error("invalid dimensions" + ALPS_STACKTRACE);
                else if (size[0] == 0)
                    value.resize(0);
                else {
                    value.resize(*(size.begin() + chunk.size()));
                    chunk.push_back(1);
                    offset.push_back(0);
                    for(typename std::vector<bool, A>::iterator it = value.begin(); it != value.end(); ++it) {
                        offset.back() = it - value.begin();
                        bool elem;
                        ar.read(path, &elem, chunk, offset);
                        *it = elem;
                    }
                }
            }
        }

    }
}

#endif
