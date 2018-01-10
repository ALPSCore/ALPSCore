/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_HDF5_BOOST_MULTI_ARRAY_HPP
#define ALPS_HDF5_BOOST_MULTI_ARRAY_HPP

#include <alps/hdf5/archive.hpp>

// #include <alps/multi_array.hpp>

#include <boost/multi_array.hpp>

namespace alps {
    namespace hdf5 {

        template<typename T, std::size_t N, typename A> struct scalar_type<boost::multi_array<T, N, A> > {
            typedef typename scalar_type<typename boost::remove_reference<typename boost::remove_cv<T>::type>::type>::type type;
        };
        // template<typename T, std::size_t N, typename A> struct scalar_type<alps::multi_array<T, N, A> > 
        //     : public scalar_type<boost::multi_array<T, N, A> > 
        // {};

        template<typename T, std::size_t N, typename A> struct is_content_continuous<boost::multi_array<T, N, A> > 
            : public is_continuous<T> 
        {};
        // template<typename T, std::size_t N, typename A> struct is_content_continuous<alps::multi_array<T, N, A> > 
        //     : public is_continuous<T> 
        // {};

        template<typename T, std::size_t N, typename A> struct has_complex_elements<boost::multi_array<T, N, A> > 
            : public has_complex_elements<typename alps::detail::remove_cvr<T>::type>
        {};
        // template<typename T, std::size_t N, typename A> struct has_complex_elements<alps::multi_array<T, N, A> > 
        //     : public has_complex_elements<boost::multi_array<T, N, A> > 
        // {};

        namespace detail {

            template<typename T, std::size_t N, typename A> struct get_extent<boost::multi_array<T, N, A> > {
                static std::vector<std::size_t> apply(boost::multi_array<T, N, A> const & value) {
                    using alps::hdf5::get_extent;
                    std::vector<std::size_t> result(value.shape(), value.shape() + boost::multi_array<T, N, A>::dimensionality);
                    if (value.num_elements()) {
                        std::vector<std::size_t> extent(get_extent(*value.data()));
                        for (std::size_t i = 1; i < value.num_elements(); ++i)
                            if (!std::equal(extent.begin(), extent.end(), get_extent(value.data()[i]).begin()))
                                throw archive_error("no rectangular matrix");
                        std::copy(extent.begin(), extent.end(), std::back_inserter(result));
                    }
                    return result;
                }
            };
            // template<typename T, std::size_t N, typename A> struct get_extent<alps::multi_array<T, N, A> >
            //     : public get_extent<boost::multi_array<T, N, A> > 
            // {};

            template<typename T, std::size_t N, typename A> struct set_extent<boost::multi_array<T, N, A> > {
                static void apply(boost::multi_array<T, N, A> & value, std::vector<std::size_t> const & size) {
                    using alps::hdf5::set_extent;
                    if (boost::multi_array<T, N, A>::dimensionality > size.size())
                        throw archive_error("invalid data size");
                    if (!std::equal(value.shape(), value.shape() + boost::multi_array<T, N, A>::dimensionality, size.begin())) {
                        typename boost::multi_array<T, N, A>::extent_gen extents;
                        gen_extent(value, extents, size);
                    }
                    if (!is_continuous<T>::value && boost::multi_array<T, N, A>::dimensionality < size.size())
                        for (std::size_t i = 0; i < value.num_elements(); ++i)
                            set_extent(value.data()[i], std::vector<std::size_t>(size.begin() + boost::multi_array<T, N, A>::dimensionality, size.end()));
                }
                private:
                    template<std::size_t M> static void gen_extent(boost::multi_array<T, N, A> & value, boost::detail::multi_array::extent_gen<M> extents, std::vector<std::size_t> const & size) {
                        gen_extent(value, extents[size.front()], std::vector<std::size_t>(size.begin() + 1, size.end()));
                    }
                    static void gen_extent(boost::multi_array<T, N, A> & value, typename boost::detail::multi_array::extent_gen<N> extents, std::vector<std::size_t> const & /*size*/) {
                        value.resize(extents);
                    }
            };
            // template<typename T, std::size_t N, typename A> struct set_extent<alps::multi_array<T, N, A> >
            //     : public set_extent<boost::multi_array<T, N, A> > 
            // {};

            template<typename T, std::size_t N, typename A> struct is_vectorizable<boost::multi_array<T, N, A> > {
                static bool apply(boost::multi_array<T, N, A> const & value) {
                    using alps::hdf5::get_extent;
                    using alps::hdf5::is_vectorizable;
                    std::vector<std::size_t> size(get_extent(*value.data()));
                    for (std::size_t i = 1; i < value.num_elements(); ++i)
                        if (!is_vectorizable(value.data()[i]) || !std::equal(size.begin(), size.end(), get_extent(value.data()[i]).begin()))
                            return false;
                    return true;
                }
            };
            // template<typename T, std::size_t N, typename A> struct is_vectorizable<alps::multi_array<T, N, A> >
            //     : public is_vectorizable<boost::multi_array<T, N, A> > 
            // {};

            template<typename T, std::size_t N, typename A> struct get_pointer<boost::multi_array<T, N, A> > {
                static typename alps::hdf5::scalar_type<boost::multi_array<T, N, A> >::type * apply(boost::multi_array<T, N, A> & value) {
                    using alps::hdf5::get_pointer;
                    return get_pointer(*value.data());
                }
            };
            // template<typename T, std::size_t N, typename A> struct get_pointer<alps::multi_array<T, N, A> >
            //     : public get_pointer<boost::multi_array<T, N, A> > 
            // {};

            template<typename T, std::size_t N, typename A> struct get_pointer<boost::multi_array<T, N, A> const> {
                static typename alps::hdf5::scalar_type<boost::multi_array<T, N, A> >::type const * apply(boost::multi_array<T, N, A> const & value) {
                    using alps::hdf5::get_pointer;
                    return get_pointer(*value.data());
                }
            };
            // template<typename T, std::size_t N, typename A> struct get_pointer<alps::multi_array<T, N, A> const>
            //     : public get_pointer<boost::multi_array<T, N, A> const> 
            // {};

        }

        template<typename T, std::size_t N, typename A> void save(
              archive & ar
            , std::string const & path
            , boost::multi_array<T, N, A> const & value
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
            } else if (is_vectorizable(value)) {
                std::copy(value.shape(), value.shape() + boost::multi_array<T, N, A>::dimensionality, std::back_inserter(size));
                std::fill_n(std::back_inserter(chunk), value.num_elements(), 1);
                for (std::size_t i = 1; i < value.num_elements(); ++i) {
                    std::vector<std::size_t> local_offset(offset);
                    local_offset.push_back(i / value.num_elements() * *value.shape());
                    for (
                        typename boost::multi_array<T, N, A>::size_type const * it = value.shape() + 1;
                        it != value.shape() + boost::multi_array<T, N, A>::dimensionality;
                        ++it
                    )
                        local_offset.push_back((i % std::accumulate(
                            it, value.shape() + boost::multi_array<T, N, A>::dimensionality, std::size_t(1), std::multiplies<std::size_t>()
                        )) / std::accumulate(
                            it + 1, value.shape() + boost::multi_array<T, N, A>::dimensionality, std::size_t(1), std::multiplies<std::size_t>()
                        ));
                    save(ar, path, value.data()[i], size, chunk, local_offset);
                }
            } else
                throw wrong_type("invalid type");
        }
        // template<typename T, std::size_t N, typename A> void save(
        //       archive & ar
        //     , std::string const & path
        //     , alps::multi_array<T, N, A> const & value
        //     , std::vector<std::size_t> size = std::vector<std::size_t>()
        //     , std::vector<std::size_t> chunk = std::vector<std::size_t>()
        //     , std::vector<std::size_t> offset = std::vector<std::size_t>()
        // ) {
        //     save(ar, path, static_cast<boost::multi_array<T, N, A> const &>(value), size, chunk, offset);
        // }

        template<typename T, std::size_t N, typename A> void load(
              archive & ar
            , std::string const & path
            , boost::multi_array<T, N, A> & value
            , std::vector<std::size_t> chunk = std::vector<std::size_t>()
            , std::vector<std::size_t> offset = std::vector<std::size_t>()
        ) {
            if (ar.is_group(path))
                throw invalid_path("invalid path");
            else {
                if (ar.is_complex(path) != has_complex_elements<T>::value)
                    throw archive_error("no complex value in archive" + ALPS_STACKTRACE);
                std::vector<std::size_t> size(ar.extent(path));
                if (boost::multi_array<T, N, A>::dimensionality <= size.size())
                    set_extent(value, std::vector<std::size_t>(size.begin() + chunk.size(), size.end()));
                if (is_continuous<T>::value) {
                    std::copy(size.begin() + chunk.size(), size.end(), std::back_inserter(chunk));
                    std::fill_n(std::back_inserter(offset), size.size() - offset.size(), 0);
                    ar.read(path, get_pointer(value), chunk, offset);
                } else {
                    std::fill_n(std::back_inserter(chunk), value.num_elements(), 1);
                    for (std::size_t i = 1; i < value.num_elements(); ++i) {
                        std::vector<std::size_t> local_offset(offset);
                        local_offset.push_back(i / value.num_elements() * *value.shape());
                        for (
                            typename boost::multi_array<T, N, A>::size_type const * it = value.shape() + 1;
                            it != value.shape() + boost::multi_array<T, N, A>::dimensionality;
                            ++it
                        )
                            local_offset.push_back((i % std::accumulate(
                                it, value.shape() + boost::multi_array<T, N, A>::dimensionality, std::size_t(1), std::multiplies<std::size_t>()
                            )) / std::accumulate(
                                it + 1, value.shape() + boost::multi_array<T, N, A>::dimensionality, std::size_t(1), std::multiplies<std::size_t>()
                            ));
                        load(ar, path, value.data()[i], chunk, local_offset);
                    }
                }
            }
        }
        // template<typename T, std::size_t N, typename A> void load(
        //       archive & ar
        //     , std::string const & path
        //     , alps::multi_array<T, N, A> & value
        //     , std::vector<std::size_t> chunk = std::vector<std::size_t>()
        //     , std::vector<std::size_t> offset = std::vector<std::size_t>()
        // ) {
        //     load(ar, path, static_cast<boost::multi_array<T, N, A> &>(value), chunk, offset);                                                   
        // }
   }
}

#endif
