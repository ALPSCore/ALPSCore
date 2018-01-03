/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_HDF5_STD_PAIR
#define ALPS_HDF5_STD_PAIR

#include <alps/hdf5/archive.hpp>
#include <alps/utilities/cast.hpp>
#include <alps/utilities/remove_cvr.hpp>

#include <utility>

namespace alps {
    namespace hdf5 {

        template <typename T, typename U> void save(
              archive & ar
            , std::string const & path
            , std::pair<T, U> const & value
            , std::vector<std::size_t> /*size*/ = std::vector<std::size_t>()
            , std::vector<std::size_t> /*chunk*/ = std::vector<std::size_t>()
            , std::vector<std::size_t> /*offset*/ = std::vector<std::size_t>()
        ) {
            save(ar, ar.complete_path(path) + "/0", value.first);
            if (has_complex_elements<typename alps::detail::remove_cvr<T>::type>::value)
                ar.set_complex(ar.complete_path(path) + "/0");
            save(ar, ar.complete_path(path) + "/1", value.second);
            if (has_complex_elements<typename alps::detail::remove_cvr<U>::type>::value)
                ar.set_complex(ar.complete_path(path) + "/1");
        }

        template <typename T, typename U> void load(
              archive & ar
            , std::string const & path
            , std::pair<T, U> & value
            , std::vector<std::size_t> /*chunk*/ = std::vector<std::size_t>()
            , std::vector<std::size_t> /*offset*/ = std::vector<std::size_t>()
        ) {
            try {
                load(ar, ar.complete_path(path) + "/0", value.first);
                load(ar, ar.complete_path(path) + "/1", value.second);
            } catch (path_not_found exc) {
                load(ar, ar.complete_path(path) + "/first", value.first);
                load(ar, ar.complete_path(path) + "/second", value.second);
            }
        }

        template<typename T> struct scalar_type<std::pair<T *, std::vector<std::size_t> > > {
            typedef typename scalar_type<typename alps::detail::remove_cvr<T>::type>::type type;
        };

        template<typename T> struct is_content_continuous<std::pair<T *, std::vector<std::size_t> > >
            : public is_continuous<T> 
        {};

        template<typename T> struct has_complex_elements<std::pair<T *, std::vector<std::size_t> > > 
            : public has_complex_elements<typename alps::detail::remove_cvr<T>::type>
        {};

        namespace detail {

            template<typename T> struct get_extent<std::pair<T *, std::vector<std::size_t> > > {
                static std::vector<std::size_t> apply(std::pair<T *, std::vector<std::size_t> > const & value) {
                    using alps::hdf5::get_extent;
                    std::vector<std::size_t> extent(value.second);
                    std::vector<std::size_t> size(value.second.size() ? get_extent(*value.first) : std::vector<std::size_t>());
                    if (!is_continuous<T>::value && value.second.size()) {
                        for (std::size_t i = 1; i < std::accumulate(value.second.begin(), value.second.end(), std::size_t(1), std::multiplies<std::size_t>()); ++i)
                            if (!std::equal(size.begin(), size.end(), get_extent(value.first[i]).begin()))
                                throw archive_error("no rectengual matrix" + ALPS_STACKTRACE);
                    }
                    std::copy(size.begin(), size.end(), std::back_inserter(extent));
                    return extent;
                }
            };

            template<typename T> struct set_extent<std::pair<T *, std::vector<std::size_t> > > {
                static void apply(std::pair<T *, std::vector<std::size_t> > & value, std::vector<std::size_t> const & size) {
                    using alps::hdf5::set_extent;
                    if (value.second.size() > size.size() || !std::equal(value.second.begin(), value.second.end(), size.begin()))
                        throw archive_error("invalid data size" + ALPS_STACKTRACE);
                    if (!is_continuous<T>::value && value.second.size() && value.second.size() < size.size())
                        for (std::size_t i = 0; i < std::accumulate(value.second.begin(), value.second.end(), std::size_t(1), std::multiplies<std::size_t>()); ++i)
                             set_extent(value.first[i], std::vector<std::size_t>(size.begin() + value.second.size(), size.end()));
                }
            };

            template<typename T> struct is_vectorizable<std::pair<T *, std::vector<std::size_t> > > {
                static bool apply(std::pair<T *, std::vector<std::size_t> > const & value) {
                    using alps::hdf5::get_extent;
                    using alps::hdf5::is_vectorizable;
                    std::vector<std::size_t> size(get_extent(*value.first));
                    for (std::size_t i = 1; i < std::accumulate(value.second.begin(), value.second.end(), std::size_t(1), std::multiplies<std::size_t>()); ++i)
                        if (!is_vectorizable(value.first[i]) || !std::equal(size.begin(), size.end(), get_extent(value.first[i]).begin()))
                            return false;
                    return true;
                }
            };

            template<typename T> struct get_pointer<std::pair<T *, std::vector<std::size_t> > > {
                static typename alps::hdf5::scalar_type<std::pair<T *, std::vector<std::size_t> > >::type * apply(std::pair<T *, std::vector<std::size_t> > & value) {
                    using alps::hdf5::get_pointer;
                    return get_pointer(*value.first);
                }
            };

            template<typename T> struct get_pointer<std::pair<T *, std::vector<std::size_t> > const> {
                static typename alps::hdf5::scalar_type<std::pair<T *, std::vector<std::size_t> > >::type const * apply(std::pair<T *, std::vector<std::size_t> > const & value) {
                    using alps::hdf5::get_pointer;
                    return get_pointer(*value.first);
                }
            };

        }

        template<typename T> void save(
              archive & ar                                                               
            , std::string const & path
            , std::pair<T *, std::vector<std::size_t> > const & value
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
            } else if (value.second.size() == 0)
                ar.write(path, static_cast<int const *>(NULL), std::vector<std::size_t>());
            else if (is_vectorizable(value)) {
                std::copy(value.second.begin(), value.second.end(), std::back_inserter(size));
                std::fill_n(std::back_inserter(chunk), value.second.size(), 1);
                for (
                    std::size_t i = 0;
                    i < std::accumulate(value.second.begin(), value.second.end(), std::size_t(1), std::multiplies<std::size_t>());
                    ++i
                ) {
                    std::vector<std::size_t> local_offset(offset);
                    local_offset.push_back(
                        i / std::accumulate(value.second.begin() + 1, value.second.end(), std::size_t(1), std::multiplies<std::size_t>())
                    );
                    for (std::vector<std::size_t>::const_iterator it = value.second.begin() + 1; it != value.second.end(); ++it)
                        local_offset.push_back((i % std::accumulate(
                            it, value.second.end(), std::size_t(1), std::multiplies<std::size_t>()
                        )) / std::accumulate(
                            it + 1, value.second.end(), std::size_t(1), std::multiplies<std::size_t>()
                        ));
                    save(ar, path, value.first[i], size, chunk, local_offset);
                }
            } else {
                if (path.find_last_of('@') != std::string::npos)
                    throw archive_error("attributes needs to be vectorizable: " + path + ALPS_STACKTRACE);
                if (ar.is_data(path))
                    ar.delete_data(path);
                offset = std::vector<std::size_t>(value.second.size(), 0);
                do {
                    std::size_t last = offset.size() - 1, pos = 0;
                    std::string location = "";
                    for (std::vector<std::size_t>::const_iterator it = offset.begin(); it != offset.end(); ++it) {
                        location += "/" + cast<std::string>(*it);
                        pos += *it * std::accumulate(
                            value.second.begin() + (it - offset.begin()) + 1,
                            value.second.end(),
                            std::size_t(1),
                            std::multiplies<std::size_t>()
                        );
                    }
                    save(ar, path + location, value.first[pos]);
                    if (offset[last] + 1 == value.second[last] && last) {
                        for (pos = last; ++offset[pos] == value.second[pos] && pos; --pos);
                        for (++pos; pos <= last; ++pos)
                            offset[pos] = 0;
                    } else
                        ++offset[last];
                } while (offset[0] < value.second[0]);
            }
        }

        template<typename T> void load(
              archive & ar
            , std::string const & path
            , std::pair<T *, std::vector<std::size_t> > & value
            , std::vector<std::size_t> chunk = std::vector<std::size_t>()
            , std::vector<std::size_t> offset = std::vector<std::size_t>()
        ) {
            if (ar.is_group(path)) {
                offset = std::vector<std::size_t>(value.second.size(), 0);
                do {
                    std::size_t last = offset.size() - 1, pos = 0;
                    std::string location = "";
                    for (std::vector<std::size_t>::const_iterator it = offset.begin(); it != offset.end(); ++it) {
                        location += "/" + cast<std::string>(*it);
                        pos += *it * std::accumulate(
                            value.second.begin() + (it - offset.begin()) + 1,
                            value.second.end(),
                            std::size_t(1),
                            std::multiplies<std::size_t>()
                        );
                    }
                    load(ar, path + location, value.first[pos]);
                    if (offset[last] + 1 == value.second[last] && last) {
                        for (pos = last; ++offset[pos] == value.second[pos] && pos; --pos);
                        for (++pos; pos <= last; ++pos)
                            offset[pos] = 0;
                    } else
                        ++offset[last];
                } while (offset[0] < value.second[0]);
            } else {
                std::vector<std::size_t> size(ar.extent(path));
                set_extent(value, std::vector<std::size_t>(size.begin() + chunk.size(), size.end()));
                if (is_continuous<T>::value) {
                    std::copy(size.begin(), size.end(), std::back_inserter(chunk));
                    std::fill_n(std::back_inserter(offset), size.size(), 0);
                    ar.read(path, get_pointer(value), chunk, offset);
                } else if (value.second.size()) {
                    std::fill_n(std::back_inserter(chunk), value.second.size(), 1);
                    for (
                        std::size_t i = 0;
                        i < std::accumulate(value.second.begin(), value.second.end(), std::size_t(1), std::multiplies<std::size_t>());
                        ++i
                    ) {
                        std::vector<std::size_t> local_offset(offset);
                        local_offset.push_back(
                            i / std::accumulate(value.second.begin() + 1, value.second.end(), std::size_t(1), std::multiplies<std::size_t>())
                        );
                        for (std::vector<std::size_t>::iterator it = value.second.begin() + 1; it != value.second.end(); ++it)
                            local_offset.push_back((i % std::accumulate(
                                it, value.second.end(), std::size_t(1), std::multiplies<std::size_t>()
                            )) / std::accumulate(
                                it + 1, value.second.end(), std::size_t(1), std::multiplies<std::size_t>()
                            ));
                        load(ar, path, value.first[i], chunk, local_offset);
                    }
                }
            }                                                                            
        }
    }
}

#endif

