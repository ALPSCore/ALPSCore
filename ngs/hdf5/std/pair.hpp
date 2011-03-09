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

#ifndef ALPS_NGS_HDF5_STD_PAIR
#define ALPS_NGS_HDF5_STD_PAIR

#include <alps/ngs/mchdf5.hpp>
#include <alps/ngs/convert.hpp>

#include <boost/type_traits/remove_const.hpp>

#include <utility>

namespace alps {

    template <typename T, typename U> void serialize(
          mchdf5 & ar
        , std::string const & path
        , std::pair<T, U> const & value
        , std::vector<std::size_t> size = std::vector<std::size_t>()
        , std::vector<std::size_t> chunk = std::vector<std::size_t>()
        , std::vector<std::size_t> offset = std::vector<std::size_t>()
    ) {
        serialize(ar, path + "/first", value.first);
        serialize(ar, path + "/second", value.second);
    }

    template <typename T, typename U> void unserialize(
          mchdf5 & ar
        , std::string const & path
        , std::pair<T, U> & value
        , std::vector<std::size_t> chunk = std::vector<std::size_t>()
        , std::vector<std::size_t> offset = std::vector<std::size_t>()
    ) {
        unserialize(ar, path + "/first", value.first);
        unserialize(ar, path + "/second", value.second);
    }

    template<typename T> struct scalar_type<std::pair<T *, std::vector<std::size_t> > > {
        typedef typename scalar_type<typename boost::remove_const<T>::type>::type type;
    };

    template<typename T> struct has_complex_elements<std::pair<T *, std::vector<std::size_t> > > 
        : public has_complex_elements<typename boost::remove_const<T>::type>
    {};

    namespace detail {

        template<typename T> struct get_extent<std::pair<T *, std::vector<std::size_t> > > {
            static std::vector<std::size_t> apply(std::pair<T *, std::vector<std::size_t> > const & value) {
                using alps::get_extent;
                std::vector<std::size_t> extent(value.second);
                if (!is_continous<T>::value && value.second.size()) {
                    std::vector<std::size_t> size(get_extent(*value.first));
                    for (std::size_t i = 1; i < std::accumulate(value.second.begin(), value.second.end(), std::size_t(1), std::multiplies<std::size_t>()); ++i)
                        if (!std::equal(size.begin(), size.end(), get_extent(value.first[i]).begin()))
                            ALPS_NGS_THROW_RUNTIME_ERROR("no rectengual matrix")
                    std::copy(size.begin(), size.end(), std::back_inserter(extent));
                }
                return extent;
            }
        };

        template<typename T> struct set_extent<std::pair<T *, std::vector<std::size_t> > > {
            static void apply(std::pair<T *, std::vector<std::size_t> > & value, std::vector<std::size_t> const & size) {
                using alps::set_extent;
                if (!std::equal(value.second.begin(), value.second.end(), size.begin()))
                    ALPS_NGS_THROW_RUNTIME_ERROR("invalid data size")
                if (!is_continous<T>::value && value.second.size() != size.size())
                    for (std::size_t i = 0; i < std::accumulate(value.second.begin(), value.second.end(), std::size_t(1), std::multiplies<std::size_t>()); ++i)
                         set_extent(value.first[i], std::vector<std::size_t>(size.begin() + value.second.size(), size.end()));
            }
        };

        template<typename T> struct is_vectorizable<std::pair<T *, std::vector<std::size_t> > > {
            static bool apply(std::pair<T *, std::vector<std::size_t> > const & value) {
                using alps::get_extent;
                using alps::is_vectorizable;
                std::vector<std::size_t> size(get_extent(*value.first));
                for (std::size_t i = 1; i < std::accumulate(value.second.begin(), value.second.end(), std::size_t(1), std::multiplies<std::size_t>()); ++i)
                    if (!is_vectorizable(value.first[i]) || !std::equal(size.begin(), size.end(), get_extent(value.first[i]).begin()))
                        return false;
                return true;
            }
        };

        template<typename T> struct get_pointer<std::pair<T *, std::vector<std::size_t> > > {
            static typename alps::scalar_type<std::pair<T *, std::vector<std::size_t> > >::type * apply(std::pair<T *, std::vector<std::size_t> > & value) {
                using alps::get_pointer;
                return get_pointer(*value.first);
            }
        };

        template<typename T> struct get_pointer<std::pair<T *, std::vector<std::size_t> > const> {
            static typename alps::scalar_type<std::pair<T *, std::vector<std::size_t> > >::type const * apply(std::pair<T *, std::vector<std::size_t> > const & value) {
                using alps::get_pointer;
                return get_pointer(*value.first);
            }
        };

    }

    template<typename T> void serialize(
          mchdf5 & ar
        , std::string const & path
        , std::pair<T *, std::vector<std::size_t> > const & value
        , std::vector<std::size_t> size = std::vector<std::size_t>()
        , std::vector<std::size_t> chunk = std::vector<std::size_t>()
        , std::vector<std::size_t> offset = std::vector<std::size_t>()
    ) {
        if (is_continous<T>::value) {
            std::vector<std::size_t> extent(get_extent(value));
            std::copy(extent.begin(), extent.end(), std::back_insert_iterator<std::vector<std::size_t> >(size));
            std::copy(extent.begin(), extent.end(), std::back_insert_iterator<std::vector<std::size_t> >(chunk));
            std::fill_n(std::back_insert_iterator<std::vector<std::size_t> >(offset), extent.size(), 0);
            ar.write(path, get_pointer(value), size, chunk, offset);
        } else if (is_vectorizable(value)) {
            std::copy(value.second.begin(), value.second.end(), std::back_insert_iterator<std::vector<std::size_t> >(size));
            std::fill_n(std::back_insert_iterator<std::vector<std::size_t> >(chunk), value.second.size(), 1);
            for (std::size_t i = 1; i < std::accumulate(value.second.begin(), value.second.end(), std::size_t(1), std::multiplies<std::size_t>()); ++i) {
                std::vector<std::size_t> local_offset(offset);
                local_offset.push_back(i / std::accumulate(value.second.begin() + 1, value.second.end(), std::size_t(1), std::multiplies<std::size_t>()));
                for (std::vector<std::size_t>::const_iterator it = value.second.begin() + 1; it != value.second.end(); ++it)
                    local_offset.push_back((i % std::accumulate(
                        it, value.second.end(), std::size_t(1), std::multiplies<std::size_t>()
                    )) / std::accumulate(
                        it + 1, value.second.end(), std::size_t(1), std::multiplies<std::size_t>()
                    ));
                serialize(ar, path, value.first[i], size, chunk, local_offset);
            }
        } else {
            if (path.find_last_of('@') != std::string::npos)
                ALPS_NGS_THROW_RUNTIME_ERROR("attributes needs to be vectorizable: " + path)
            if (ar.is_data(path))
                ar.delete_data(path);
            offset = std::vector<std::size_t>(value.second.size(), 0);
            do {
                std::size_t last = offset.size() - 1, pos = 0;
                std::string location = "";
                for (std::vector<std::size_t>::const_iterator it = offset.begin(); it != offset.end(); ++it) {
                    location += "/" + convert<std::string>(*it);
                    pos += *it * std::accumulate(value.second.begin() + (it - offset.begin()) + 1, value.second.end(), std::size_t(1), std::multiplies<std::size_t>());
                }
                serialize(ar, path + location, value.first[pos]);
                if (offset[last] + 1 == value.second[last] && last) {
                    for (pos = last; ++offset[pos] == value.second[pos] && pos; --pos);
                    for (++pos; pos <= last; ++pos)
                        offset[pos] = 0;
                } else
                    ++offset[last];
            } while (offset[0] < value.second[0]);
        }
    }

    template<typename T> void unserialize(
          mchdf5 & ar
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
                    location += "/" + convert<std::string>(*it);
                    pos += *it * std::accumulate(value.second.begin() + (it - offset.begin()) + 1, value.second.end(), std::size_t(1), std::multiplies<std::size_t>());
                }
                unserialize(ar, path + location, value.first[pos]);
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
            if (is_continous<T>::value) {
                std::copy(size.begin(), size.end(), std::back_insert_iterator<std::vector<std::size_t> >(chunk));
                std::fill_n(std::back_insert_iterator<std::vector<std::size_t> >(offset), size.size(), 0);
                ar.read(path, get_pointer(value), chunk, offset);
            } else {
                std::fill_n(std::back_insert_iterator<std::vector<std::size_t> >(chunk), value.second.size(), 1);
                for (std::size_t i = 1; i < std::accumulate(value.second.begin(), value.second.end(), std::size_t(1), std::multiplies<std::size_t>()); ++i) {
                    std::vector<std::size_t> local_offset(offset);
                    local_offset.push_back(i / std::accumulate(value.second.begin() + 1, value.second.end(), std::size_t(1), std::multiplies<std::size_t>()));
                    for (std::vector<std::size_t>::iterator it = value.second.begin() + 1; it != value.second.end(); ++it)
                        local_offset.push_back((i % std::accumulate(
                            it, value.second.end(), std::size_t(1), std::multiplies<std::size_t>()
                        )) / std::accumulate(
                            it + 1, value.second.end(), std::size_t(1), std::multiplies<std::size_t>()
                        ));
                    serialize(ar, path, value.first[i], chunk, local_offset);
                }
            }
        }
    }

}

#endif

