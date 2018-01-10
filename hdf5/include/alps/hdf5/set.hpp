/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_HDF5_SET_HPP_2a637be7f9f94f558c3f3ec131dbae11
#define ALPS_HDF5_SET_HPP_2a637be7f9f94f558c3f3ec131dbae11

#include <set>

namespace alps {
    namespace hdf5 {

        template <typename T>
        void save(alps::hdf5::archive& ar, const std::string& path,
                  const std::set<T>& value,
                  std::vector<std::size_t> size=std::vector<std::size_t>(),
                  std::vector<std::size_t> chunk=std::vector<std::size_t>(),
                  std::vector<std::size_t> offset=std::vector<std::size_t>())
        {
            throw std::logic_error("save<std::set>() is not yet implemented");
        }
          
        template <typename T>
        void load(alps::hdf5::archive& ar, const std::string& path,
                  std::set<T>& value,
                  std::vector<std::size_t> size=std::vector<std::size_t>(),
                  std::vector<std::size_t> chunk=std::vector<std::size_t>(),
                  std::vector<std::size_t> offset=std::vector<std::size_t>())
        {
            throw std::logic_error("load<std::set>() is not yet implemented");
        }

    } // hdf5::
} // alps::


#endif /* ALPS_HDF5_SET_HPP_2a637be7f9f94f558c3f3ec131dbae11 */


