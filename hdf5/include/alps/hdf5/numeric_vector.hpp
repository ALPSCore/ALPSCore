/*
 * Copyright (C) 1998-2016 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_HDF5_NUMERIC_VECTOR_HPP
#define ALPS_HDF5_NUMERIC_VECTOR_HPP

#include <alps/hdf5/archive.hpp>
#include <alps/numeric/matrix/vector.hpp>

namespace alps {
namespace hdf5 {

        template <typename T, typename MemoryBlock>
        void save(
                  alps::hdf5::archive & ar
                  , std::string const & path
                  , alps::numeric::vector<T, MemoryBlock> const & value
                  , std::vector<std::size_t> size = std::vector<std::size_t>()
                  , std::vector<std::size_t> chunk = std::vector<std::size_t>()
                  , std::vector<std::size_t> offset = std::vector<std::size_t>()
                  ) {
            ar[path] << static_cast<MemoryBlock const&>(value);
        }
        template <typename T, typename MemoryBlock>
        void load(
                  alps::hdf5::archive & ar
                  , std::string const & path
                  , alps::numeric::vector<T, MemoryBlock> & value
                  , std::vector<std::size_t> chunk = std::vector<std::size_t>()
                  , std::vector<std::size_t> offset = std::vector<std::size_t>()
                  ) {
            MemoryBlock tmp;
            ar[path] >> tmp;
            value = alps::numeric::vector<T, MemoryBlock>(tmp.begin(), tmp.end());
        }
}
}
#endif // ALPS_HDF5_NUMERIC_VECTOR_HPP
