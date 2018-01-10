/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_HDF5_POINTER_HPP
#define ALPS_HDF5_POINTER_HPP

#include <alps/hdf5/pair.hpp>

namespace alps {

    template <typename T> hdf5::detail::make_pvp_proxy<std::pair<T *, std::vector<std::size_t> > > make_pvp(
          std::string const & path
        , T * value
        , std::size_t size
    ) {
        return hdf5::detail::make_pvp_proxy<std::pair<T *, std::vector<std::size_t> > >(
              path
            , std::make_pair(value, size > 0 
                ? std::vector<std::size_t>(1, size)
                : std::vector<std::size_t>()
            )
        );
    }

    template <typename T> hdf5::detail::make_pvp_proxy<std::pair<T *, std::vector<std::size_t> > > make_pvp(
          std::string const & path
        , T * value
        , std::vector<std::size_t> const & size
    ) {
        return hdf5::detail::make_pvp_proxy<std::pair<T *, std::vector<std::size_t> > >(
              path
            , std::make_pair(value, size)
        );
    }

}

#endif
