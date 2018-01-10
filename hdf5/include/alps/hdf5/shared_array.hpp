/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_HDF5_BOOST_SHARED_ARRAY_HPP
#define ALPS_HDF5_BOOST_SHARED_ARRAY_HPP

#include <alps/hdf5/archive.hpp>

#include <boost/shared_array.hpp>

namespace alps {

    #define ALPS_HDF5_BOOST_SHARED_ARRAY_MAKE_PVP(ptr_type, arg_type)                                                                                               \
        template <typename T> hdf5::detail::make_pvp_proxy<std::pair<ptr_type, std::vector<std::size_t> > > make_pvp(                                               \
              std::string const & path                                                                                                                              \
            , arg_type value                                                                                                                                        \
            , std::size_t size                                                                                                                                      \
        ) {                                                                                                                                                         \
            return hdf5::detail::make_pvp_proxy<std::pair<ptr_type, std::vector<std::size_t> > >(                                                                   \
                  path                                                                                                                                              \
                , std::make_pair(value.get(), std::vector<std::size_t>(1, size))                                                                                    \
            );                                                                                                                                                      \
        }                                                                                                                                                           \
                                                                                                                                                                    \
        template <typename T> hdf5::detail::make_pvp_proxy<std::pair<ptr_type, std::vector<std::size_t> > > make_pvp(                                               \
              std::string const & path                                                                                                                              \
            , arg_type value                                                                                                                                        \
            , std::vector<std::size_t> const & size                                                                                                                 \
        ) {                                                                                                                                                         \
            return hdf5::detail::make_pvp_proxy<std::pair<ptr_type, std::vector<std::size_t> > >(path, std::make_pair(value.get(), size));                          \
        }
    ALPS_HDF5_BOOST_SHARED_ARRAY_MAKE_PVP(T *, boost::shared_array<T> &)
    ALPS_HDF5_BOOST_SHARED_ARRAY_MAKE_PVP(T const *, boost::shared_array<T> const &)
    #undef ALPS_HDF5_BOOST_SHARED_ARRAY_MAKE_PVP

}

#endif
