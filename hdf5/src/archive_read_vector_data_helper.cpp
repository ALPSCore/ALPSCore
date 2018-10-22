/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <iostream>
#include <vector>

#include <memory>

#include <alps/hdf5/archive.hpp>
#include <alps/utilities/stacktrace.hpp>
#include <alps/utilities/cast.hpp>

#include "common.hpp"

namespace alps {
    namespace hdf5 {
        namespace detail {

            template<typename T, typename U, typename... UTail>
            inline bool hdf5_read_vector_data_helper_impl(T * value, data_type const &data_id, type_type const &native_id,
                  std::vector<std::size_t> const &chunk,
                  std::vector<std::size_t> const &offset,
                  std::vector<std::size_t> const &data_size,
                  std::true_type) {
                if (check_error(
                    H5Tequal(type_type(H5Tcopy(native_id)), type_type(get_native_type(U())))
                ) > 0) {
                    std::size_t len = std::accumulate(chunk.begin(), chunk.end(), std::size_t(1), std::multiplies<std::size_t>());
                    std::unique_ptr<U[]> raw(
                        new U[len]
                    );
                    if (std::equal(chunk.begin(), chunk.end(), data_size.begin())) {
                        check_error(H5Dread(data_id, native_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, raw.get()));
                        cast(raw.get(), raw.get() + len, value);
                    } else {
                        std::vector<hsize_t> offset_hid(offset.begin(), offset.end()),
                                            chunk_hid(chunk.begin(), chunk.end());
                        space_type space_id(H5Dget_space(data_id));
                        check_error(H5Sselect_hyperslab(space_id, H5S_SELECT_SET, &offset_hid.front(), NULL, &chunk_hid.front(), NULL));
                        space_type mem_id(H5Screate_simple(static_cast<int>(chunk_hid.size()), &chunk_hid.front(), NULL));
                        check_error(H5Dread(data_id, native_id, mem_id, space_id, H5P_DEFAULT, raw.get()));
                        cast(raw.get(), raw.get() + len, value);
                    }
                    return true;
                } else
                    return hdf5_read_vector_data_helper_impl<T, UTail...>(value,
                                                                          data_id,
                                                                          native_id,
                                                                          chunk,
                                                                          offset,
                                                                          data_size,
                                                                          std::integral_constant<bool, sizeof...(UTail) != 0>());
            }
            template<typename T, typename... UTail>
            inline bool hdf5_read_vector_data_helper_impl(T *, data_type const &, type_type const &,
                  std::vector<std::size_t> const &,
                  std::vector<std::size_t> const &,
                  std::vector<std::size_t> const &,
                  std::false_type)
            { return false; }

            template<typename T>
            bool hdf5_read_vector_data_helper(T * value, data_type const &data_id, type_type const &native_id,
                                              std::vector<std::size_t> const &chunk,
                                              std::vector<std::size_t> const &offset,
                                              std::vector<std::size_t> const &data_size) {
                return hdf5_read_vector_data_helper_impl<T, ALPS_HDF5_NATIVE_INTEGRAL_TYPES>(value,
                                                                                             data_id,
                                                                                             native_id,
                                                                                             chunk,
                                                                                             offset,
                                                                                             data_size,
                                                                                             std::true_type());
            }

            #define ALPS_HDF5_READ_VECTOR_DATA_HELPER(T)                                                  \
                template bool hdf5_read_vector_data_helper(T *, data_type const &, type_type const &,     \
                                                           std::vector<std::size_t> const &,              \
                                                           std::vector<std::size_t> const &,              \
                                                           std::vector<std::size_t> const &);
            ALPS_FOREACH_NATIVE_HDF5_TYPE(ALPS_HDF5_READ_VECTOR_DATA_HELPER);
        }
    }
}
