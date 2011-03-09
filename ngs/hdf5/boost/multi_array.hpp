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

#ifndef ALPS_NGS_HDF5_BOOST_MULTI_ARRAY_HPP
#define ALPS_NGS_HDF5_BOOST_MULTI_ARRAY_HPP

#include <alps/ngs/mchdf5.hpp>

#include <boost/multi_array.hpp>

namespace alps {

    template<typename T, std::size_t N, typename A> struct scalar_type<boost::multi_array<T, N, A> > {
        typedef typename scalar_type<typename boost::remove_const<T>::type>::type type;
    };

    template<typename T, std::size_t N, typename A> struct has_complex_elements<boost::multi_array<T, N, A> > 
        : public has_complex_elements<typename boost::remove_const<T>::type>
    {};

    namespace detail {

        template<typename T, std::size_t N, typename A> struct get_extent<boost::multi_array<T, N, A> > {
            static std::vector<std::size_t> apply(boost::multi_array<T, N, A> const & value) {
                using alps::get_extent;
                std::vector<std::size_t> extent(value.shape(), value.shape() + boost::multi_array<T, N, A>::dimensionality);
                if (!is_continous<T>::value && boost::multi_array<T, N, A>::dimensionality) {
                    std::vector<std::size_t> size(get_extent(*value.data()));
                    for (std::size_t i = 1; i < value.num_elements(); ++i)
                        if (!std::equal(size.begin(), size.end(), get_extent(value.data()[i]).begin()))
                            ALPS_NGS_THROW_RUNTIME_ERROR("no rectengual matrix")
                    std::copy(size.begin(), size.end(), std::back_inserter(extent));
                }
                return extent;
            }
        };

        template<typename T, std::size_t N, typename A> struct set_extent<boost::multi_array<T, N, A> > {
            static void apply(boost::multi_array<T, N, A> & value, std::vector<std::size_t> const & size) {
                using alps::set_extent;
                if (!std::equal(value.shape(), value.shape() + boost::multi_array<T, N, A>::dimensionality, size.begin()))
                    ALPS_NGS_THROW_RUNTIME_ERROR("invalid data size")
                if (!is_continous<T>::value && value.num_elements() != size.size())
                    for (std::size_t i = 0; i < value.num_elements(); ++i)
                         set_extent(value.data()[i], std::vector<std::size_t>(size.begin() + value.num_elements(), size.end()));
            }
        };

        template<typename T, std::size_t N, typename A> struct is_vectorizable<boost::multi_array<T, N, A> > {
            static bool apply(boost::multi_array<T, N, A> const & value) {
                using alps::get_extent;
                using alps::is_vectorizable;
                std::vector<std::size_t> size(get_extent(*value.data()));
                for (std::size_t i = 1; i < value.num_elements(); ++i)
                    if (!is_vectorizable(value.data()[i]) || !std::equal(size.begin(), size.end(), get_extent(value.data()[i]).begin()))
                        return false;
                return true;
            }
        };

        template<typename T, std::size_t N, typename A> struct get_pointer<boost::multi_array<T, N, A> > {
            static typename alps::scalar_type<boost::multi_array<T, N, A> >::type * apply(boost::multi_array<T, N, A> & value) {
                using alps::get_pointer;
                return get_pointer(*value.data());
            }
        };

        template<typename T, std::size_t N, typename A> struct get_pointer<boost::multi_array<T, N, A> const> {
            static typename alps::scalar_type<boost::multi_array<T, N, A> >::type const * apply(boost::multi_array<T, N, A> const & value) {
                using alps::get_pointer;
                return get_pointer(*value.data());
            }
        };

    }

    template<typename T, std::size_t N, typename A> void serialize(
          mchdf5 & ar
        , std::string const & path
        , boost::multi_array<T, N, A> const & value
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
            std::copy(value.shape(), value.shape() + boost::multi_array<T, N, A>::dimensionality, std::back_insert_iterator<std::vector<std::size_t> >(size));
            std::fill_n(std::back_insert_iterator<std::vector<std::size_t> >(chunk), value.num_elements(), 1);
            for (std::size_t i = 1; i < value.num_elements(); ++i) {
                std::vector<std::size_t> local_offset(offset);
                local_offset.push_back(i / value.num_elements() * *value.shape());
                for (typename boost::multi_array<T, N, A>::size_type const * it = value.shape() + 1; it != value.shape() + boost::multi_array<T, N, A>::dimensionality; ++it)
                    local_offset.push_back((i % std::accumulate(
                        it, value.shape() + boost::multi_array<T, N, A>::dimensionality, std::size_t(1), std::multiplies<std::size_t>()
                    )) / std::accumulate(
                        it + 1, value.shape() + boost::multi_array<T, N, A>::dimensionality, std::size_t(1), std::multiplies<std::size_t>()
                    ));
                serialize(ar, path, value.data()[i], size, chunk, local_offset);
            }
        } else
            ALPS_NGS_THROW_RUNTIME_ERROR("invalid type")
    }

    template<typename T, std::size_t N, typename A> void unserialize(
          mchdf5 & ar
        , std::string const & path
        , boost::multi_array<T, N, A> & value
        , std::vector<std::size_t> chunk = std::vector<std::size_t>()
        , std::vector<std::size_t> offset = std::vector<std::size_t>()
    ) {
        if (ar.is_group(path))
            ALPS_NGS_THROW_RUNTIME_ERROR("invalid path")
        else {
            std::vector<std::size_t> size(ar.extent(path));
            set_extent(value, std::vector<std::size_t>(size.begin() + chunk.size(), size.end()));
            if (is_continous<T>::value) {
                std::copy(size.begin(), size.end(), std::back_insert_iterator<std::vector<std::size_t> >(chunk));
                std::fill_n(std::back_insert_iterator<std::vector<std::size_t> >(offset), size.size(), 0);
                ar.read(path, get_pointer(value), chunk, offset);
            } else {
                std::fill_n(std::back_insert_iterator<std::vector<std::size_t> >(chunk), value.num_elements(), 1);
                for (std::size_t i = 1; i < value.num_elements(); ++i) {
                    std::vector<std::size_t> local_offset(offset);
                    local_offset.push_back(i / value.num_elements() * *value.shape());
                    for (typename boost::multi_array<T, N, A>::size_type const * it = value.shape() + 1; it != value.shape() + boost::multi_array<T, N, A>::dimensionality; ++it)
                        local_offset.push_back((i % std::accumulate(
                            it, value.shape() + boost::multi_array<T, N, A>::dimensionality, std::size_t(1), std::multiplies<std::size_t>()
                        )) / std::accumulate(
                            it + 1, value.shape() + boost::multi_array<T, N, A>::dimensionality, std::size_t(1), std::multiplies<std::size_t>()
                        ));
                    serialize(ar, path, value.data()[i], chunk, local_offset);
                }
            }
        }
    }

}

#endif
