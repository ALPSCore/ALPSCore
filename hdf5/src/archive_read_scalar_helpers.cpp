/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/hdf5/archive.hpp>
#include <alps/utilities/stacktrace.hpp>
#include <alps/utilities/cast.hpp>

#include "common.hpp"

namespace alps {
    namespace hdf5 {
        namespace detail {

            //
            // hdf5_read_scalar_data_helper
            //

            template<typename T, typename U, typename... UTail>
            inline bool hdf5_read_scalar_data_helper_impl(T & value, data_type const &data_id, type_type const &native_id, std::true_type) {
                if (check_error(
                    H5Tequal(type_type(H5Tcopy(native_id)), type_type(get_native_type(U())))
                ) > 0) {
                    U u;
                    check_error(H5Dread(data_id, native_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, &u));
                    value = cast<T>(u);
                    return true;
                } else
                    return hdf5_read_scalar_data_helper_impl<T, UTail...>(value,
                                                                          data_id,
                                                                          native_id,
                                                                          std::integral_constant<bool, sizeof...(UTail) != 0>());
            }
            template<typename T, typename... UTail>
            inline bool hdf5_read_scalar_data_helper_impl(T &, data_type const &, type_type const &, std::false_type) { return false; }

            template<typename T>
            bool hdf5_read_scalar_data_helper(T & value, data_type const &data_id, type_type const &native_id) {
                return hdf5_read_scalar_data_helper_impl<T, ALPS_HDF5_NATIVE_INTEGRAL_TYPES>(value, data_id, native_id, std::true_type());
            }

            #define ALPS_HDF5_READ_SCALAR_DATA_HELPER(T)                                                  \
                template bool hdf5_read_scalar_data_helper<T>(T &, data_type const &, type_type const &);
            ALPS_FOREACH_NATIVE_HDF5_TYPE(ALPS_HDF5_READ_SCALAR_DATA_HELPER);

            //
            // hdf5_read_scalar_attribute_helper
            //

            template<typename T, typename U, typename... UTail>
            inline bool hdf5_read_scalar_attribute_helper_impl(T & value, attribute_type const &attribute_id, type_type const &native_id, std::true_type) {
                if (check_error(
                    H5Tequal(type_type(H5Tcopy(native_id)), type_type(get_native_type(U())))
                ) > 0) {
                    U u;
                    check_error(H5Aread(attribute_id, native_id, &u));
                    value = cast< T >(u);
                    return true;
                } else
                    return hdf5_read_scalar_attribute_helper_impl<T, UTail...>(value,
                                                                               attribute_id,
                                                                               native_id,
                                                                               std::integral_constant<bool, sizeof...(UTail) != 0>());
            }
            template<typename T, typename... UTail>
            inline bool hdf5_read_scalar_attribute_helper_impl(T &, attribute_type const &, type_type const &, std::false_type) { return false; }

            template<typename T>
            bool hdf5_read_scalar_attribute_helper(T & value, attribute_type const &attribute_id, type_type const &native_id) {
                return hdf5_read_scalar_attribute_helper_impl<T, ALPS_HDF5_NATIVE_INTEGRAL_TYPES>(value, attribute_id, native_id, std::true_type());
            }

            #define ALPS_HDF5_READ_SCALAR_ATTRIBUTE_HELPER(T)                                                       \
                template bool hdf5_read_scalar_attribute_helper<T>(T &, attribute_type const &, type_type const &);
            ALPS_FOREACH_NATIVE_HDF5_TYPE(ALPS_HDF5_READ_SCALAR_ATTRIBUTE_HELPER);
        }
    }
}
