/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <iostream>
#include <vector>

#include <boost/scoped_array.hpp>

#include <hdf5.h>

#include <alps/hdf5/archive.hpp>
#include <alps/utilities/cast.hpp>

#include "common.hpp"
#include "archivecontext.hpp"

namespace alps {
    namespace hdf5 {

        #define ALPS_HDF5_READ_SCALAR_DATA_HELPER(U, T)                                                                                                                 \
            } else if (detail::check_error(                                                                                                                             \
                H5Tequal(detail::type_type(H5Tcopy(native_id)), detail::type_type(detail::get_native_type(alps::detail::type_wrapper< U >::type())))                    \
            ) > 0) {                                                                                                                                                    \
                U u;                                                                                                                                                    \
                detail::check_error(H5Dread(data_id, native_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, &u));                                                                    \
                value = cast< T >(u);
        #define ALPS_HDF5_READ_SCALAR_ATTRIBUTE_HELPER(U, T)                                                                                                            \
            } else if (detail::check_error(                                                                                                                             \
                H5Tequal(detail::type_type(H5Tcopy(native_id)), detail::type_type(detail::get_native_type(alps::detail::type_wrapper< U >::type())))                    \
            ) > 0) {                                                                                                                                                    \
                U u;                                                                                                                                                    \
                detail::check_error(H5Aread(attribute_id, native_id, &u));                                                                                              \
                value = cast< T >(u);
        #define ALPS_HDF5_READ_SCALAR(T)                                                                                                                                \
            void archive::read(std::string path, T & value) const {                                                                                                     \
                ALPS_HDF5_FAKE_THREADSAFETY                                                                                                                             \
                if (context_ == NULL)                                                                                                                                   \
                    throw archive_closed("the archive is closed" + ALPS_STACKTRACE);                                                                                    \
                if ((path = complete_path(path)).find_last_of('@') == std::string::npos) {                                                                              \
                    if (!is_data(path))                                                                                                                                 \
                        throw path_not_found("the path does not exist: " + path + ALPS_STACKTRACE);                                                                     \
                    else if (!is_scalar(path))                                                                                                                          \
                        throw wrong_type("scalar - vector conflict in path: " + path + ALPS_STACKTRACE);                                                                \
                    detail::data_type data_id(H5Dopen2(context_->file_id_, path.c_str(), H5P_DEFAULT));                                                                 \
                    detail::type_type type_id(H5Dget_type(data_id));                                                                                                    \
                    detail::type_type native_id(H5Tget_native_type(type_id, H5T_DIR_ASCEND));                                                                           \
                    if (H5Tget_class(native_id) == H5T_STRING && !detail::check_error(H5Tis_variable_str(type_id))) {                                                   \
                        std::string raw(H5Tget_size(type_id) + 1, '\0');                                                                                                \
                        detail::check_error(H5Dread(data_id, native_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, &raw[0]));                                                       \
                        value = cast< T >(raw);                                                                                                                         \
                    } else if (H5Tget_class(native_id) == H5T_STRING) {                                                                                                 \
                        char * raw;                                                                                                                                     \
                        detail::check_error(H5Dread(data_id, native_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, &raw));                                                          \
                        value = cast< T >(std::string(raw));                                                                                                            \
                        detail::check_error(H5Dvlen_reclaim(type_id, detail::space_type(H5Dget_space(data_id)), H5P_DEFAULT, &raw));                                    \
                        ALPS_HDF5_FOREACH_NATIVE_TYPE_INTEGRAL(ALPS_HDF5_READ_SCALAR_DATA_HELPER, T)                                                                    \
                    } else                                                                                                                                              \
                        throw wrong_type("invalid type" + ALPS_STACKTRACE);                                                                                             \
                } else {                                                                                                                                                \
                    if (!is_attribute(path))                                                                                                                            \
                        throw path_not_found("the path does not exist: " + path + ALPS_STACKTRACE);                                                                     \
                    else if (!is_scalar(path))                                                                                                                          \
                        throw wrong_type("scalar - vector conflict in path: " + path + ALPS_STACKTRACE);                                                                \
                    detail::attribute_type attribute_id(H5Aopen_by_name(                                                                                                \
                          context_->file_id_                                                                                                                            \
                        , path.substr(0, path.find_last_of('@')).c_str()                                                                                            \
                        , path.substr(path.find_last_of('@') + 1).c_str()                                                                                               \
                        , H5P_DEFAULT, H5P_DEFAULT                                                                                                                      \
                    ));                                                                                                                                                 \
                    detail::type_type type_id(H5Aget_type(attribute_id));                                                                                               \
                    detail::type_type native_id(H5Tget_native_type(type_id, H5T_DIR_ASCEND));                                                                           \
                    if (H5Tget_class(native_id) == H5T_STRING && !detail::check_error(H5Tis_variable_str(type_id))) {                                                   \
                        std::string raw(H5Tget_size(type_id) + 1, '\0');                                                                                                \
                        detail::check_error(H5Aread(attribute_id, native_id, &raw[0]));                                                                                 \
                        value = cast< T >(raw);                                                                                                                         \
                    } else if (H5Tget_class(native_id) == H5T_STRING) {                                                                                                 \
                        char * raw;                                                                                                                                     \
                        detail::check_error(H5Aread(attribute_id, native_id, &raw));                                                                                    \
                        value = cast< T >(std::string(raw));                                                                                                            \
                    ALPS_HDF5_FOREACH_NATIVE_TYPE_INTEGRAL(ALPS_HDF5_READ_SCALAR_ATTRIBUTE_HELPER, T)                                                                   \
                    } else throw wrong_type("invalid type" + ALPS_STACKTRACE);                                                                                          \
                }                                                                                                                                                       \
            }
        ALPS_FOREACH_NATIVE_HDF5_TYPE(ALPS_HDF5_READ_SCALAR)
        #undef ALPS_HDF5_READ_SCALAR
        #undef ALPS_HDF5_READ_SCALAR_DATA_HELPER
        #undef ALPS_HDF5_READ_SCALAR_ATTRIBUTE_HELPER

        #define ALPS_HDF5_READ_VECTOR_DATA_HELPER(U, T)                                                                                                                 \
            } else if (detail::check_error(                                                                                                                             \
                H5Tequal(detail::type_type(H5Tcopy(native_id)), detail::type_type(detail::get_native_type(alps::detail::type_wrapper< U >::type())))                    \
            ) > 0) {                                                                                                                                                    \
                std::size_t len = std::accumulate(chunk.begin(), chunk.end(), std::size_t(1), std::multiplies<std::size_t>());                                          \
                boost::scoped_array<U> raw(                                                                                                                             \
                    new alps::detail::type_wrapper< U >::type[len]                                                                                                      \
                );                                                                                                                                                      \
                if (std::equal(chunk.begin(), chunk.end(), data_size.begin())) {                                                                                        \
                    detail::check_error(H5Dread(data_id, native_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, raw.get()));                                                         \
                    cast(raw.get(), raw.get() + len, value);                                                                                                            \
                } else {                                                                                                                                                \
                    std::vector<hsize_t> offset_hid(offset.begin(), offset.end()),                                                                                      \
                                         chunk_hid(chunk.begin(), chunk.end());                                                                                         \
                    detail::space_type space_id(H5Dget_space(data_id));                                                                                                 \
                    detail::check_error(H5Sselect_hyperslab(space_id, H5S_SELECT_SET, &offset_hid.front(), NULL, &chunk_hid.front(), NULL));                            \
                    detail::space_type mem_id(H5Screate_simple(static_cast<int>(chunk_hid.size()), &chunk_hid.front(), NULL));                                          \
                    detail::check_error(H5Dread(data_id, native_id, mem_id, space_id, H5P_DEFAULT, raw.get()));                                                         \
                    cast(raw.get(), raw.get() + len, value);                                                                                                            \
                }
        #define ALPS_HDF5_READ_VECTOR_ATTRIBUTE_HELPER(U, T)                                                                                                            \
            } else if (detail::check_error(                                                                                                                             \
                H5Tequal(detail::type_type(H5Tcopy(native_id)), detail::type_type(detail::get_native_type(alps::detail::type_wrapper< U >::type())))                    \
            ) > 0) {                                                                                                                                                    \
                std::size_t len = std::accumulate(chunk.begin(), chunk.end(), std::size_t(1), std::multiplies<std::size_t>());                                          \
                boost::scoped_array<U> raw(                                                                                                                             \
                    new alps::detail::type_wrapper< U >::type[len]                                                                                                      \
                );                                                                                                                                                      \
                if (std::equal(chunk.begin(), chunk.end(), data_size.begin())) {                                                                                        \
                    detail::check_error(H5Aread(attribute_id, native_id, raw.get()));                                                                                   \
                    cast(raw.get(), raw.get() + len, value);                                                                                                            \
                } else                                                                                                                                                  \
                    throw std::logic_error("Not Implemented, path: " + path + ALPS_STACKTRACE);
        #define ALPS_HDF5_READ_VECTOR(T)                                                                                                                                \
            void archive::read(std::string path, T * value, std::vector<std::size_t> chunk, std::vector<std::size_t> offset) const {                                    \
                ALPS_HDF5_FAKE_THREADSAFETY                                                                                                                             \
                if (context_ == NULL)                                                                                                                                   \
                    throw archive_closed("the archive is closed" + ALPS_STACKTRACE);                                                                                    \
                std::vector<std::size_t> data_size = extent(path);                                                                                                      \
                if (offset.size() == 0)                                                                                                                                 \
                    offset = std::vector<std::size_t>(dimensions(path), 0);                                                                                             \
                if (data_size.size() != chunk.size() || data_size.size() != offset.size())                                                                              \
                    throw archive_error("wrong size or offset passed for path: " + path + ALPS_STACKTRACE);                                                             \
                for (std::size_t i = 0; i < data_size.size(); ++i)                                                                                                      \
                    if (data_size[i] < chunk[i] + offset[i])                                                                                                            \
                        throw archive_error("passed size of offset exeed data size for path: " + path + ALPS_STACKTRACE);                                               \
                if (is_null(path))                                                                                                                                      \
                    value = NULL;                                                                                                                                       \
                else {                                                                                                                                                  \
                    for (std::size_t i = 0; i < data_size.size(); ++i)                                                                                                  \
                        if (chunk[i] == 0)                                                                                                                              \
                            throw archive_error("size is zero in one dimension in path: " + path + ALPS_STACKTRACE);                                                    \
                    if ((path = complete_path(path)).find_last_of('@') == std::string::npos) {                                                                          \
                        if (!is_data(path))                                                                                                                             \
                            throw path_not_found("the path does not exist: " + path + ALPS_STACKTRACE);                                                                 \
                        if (is_scalar(path))                                                                                                                            \
                            throw archive_error("scalar - vector conflict in path: " + path + ALPS_STACKTRACE);                                                         \
                        detail::data_type data_id(H5Dopen2(context_->file_id_, path.c_str(), H5P_DEFAULT));                                                             \
                        detail::type_type type_id(H5Dget_type(data_id));                                                                                                \
                        detail::type_type native_id(H5Tget_native_type(type_id, H5T_DIR_ASCEND));                                                                       \
                        if (H5Tget_class(native_id) == H5T_STRING && !detail::check_error(H5Tis_variable_str(type_id)))                                                 \
                            throw std::logic_error("multidimensional dataset of fixed string datas is not implemented (" + path + ")" + ALPS_STACKTRACE);               \
                        else if (H5Tget_class(native_id) == H5T_STRING) {                                                                                               \
                            std::size_t len = std::accumulate(chunk.begin(), chunk.end(), std::size_t(1), std::multiplies<std::size_t>());                              \
                            boost::scoped_array<char *> raw(                                                                                                            \
                                new char * [len]                                                                                                                        \
                            );                                                                                                                                          \
                            if (std::equal(chunk.begin(), chunk.end(), data_size.begin())) {                                                                            \
                                detail::check_error(H5Dread(data_id, native_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, raw.get()));                                             \
                                cast(raw.get(), raw.get() + len, value);                                                                                                \
                                detail::check_error(H5Dvlen_reclaim(type_id, detail::space_type(H5Dget_space(data_id)), H5P_DEFAULT, raw.get()));                       \
                            } else {                                                                                                                                    \
                                std::vector<hsize_t> offset_hid(offset.begin(), offset.end()),                                                                          \
                                                     chunk_hid(chunk.begin(), chunk.end());                                                                             \
                                detail::space_type space_id(H5Dget_space(data_id));                                                                                     \
                                detail::check_error(H5Sselect_hyperslab(space_id, H5S_SELECT_SET, &offset_hid.front(), NULL, &chunk_hid.front(), NULL));                \
                                detail::space_type mem_id(H5Screate_simple(static_cast<int>(chunk_hid.size()), &chunk_hid.front(), NULL));                              \
                                detail::check_error(H5Dread(data_id, native_id, mem_id, space_id, H5P_DEFAULT, raw.get()));                                             \
                                cast(raw.get(), raw.get() + len, value);                                                                                                \
                                                                detail::check_error(H5Dvlen_reclaim(type_id, mem_id, H5P_DEFAULT, raw.get()));                          \
                            }                                                                                                                                           \
                        ALPS_HDF5_FOREACH_NATIVE_TYPE_INTEGRAL(ALPS_HDF5_READ_VECTOR_DATA_HELPER, T)                                                                    \
                        } else throw wrong_type("invalid type" + ALPS_STACKTRACE);                                                                                      \
                    } else {                                                                                                                                            \
                        if (!is_attribute(path))                                                                                                                        \
                            throw path_not_found("the path does not exist: " + path + ALPS_STACKTRACE);                                                                 \
                        if (is_scalar(path))                                                                                                                            \
                            throw wrong_type("scalar - vector conflict in path: " + path + ALPS_STACKTRACE);                                                            \
                        hid_t parent_id;                                                                                                                                \
                        if (is_group(path.substr(0, path.find_last_of('@'))))                                                                                       \
                            parent_id = detail::check_error(H5Gopen2(context_->file_id_, path.substr(0, path.find_last_of('@')).c_str(), H5P_DEFAULT));             \
                        else if (is_data(path.substr(0, path.find_last_of('@') - 1)))                                                                                   \
                            parent_id = detail::check_error(H5Dopen2(context_->file_id_, path.substr(0, path.find_last_of('@')).c_str(), H5P_DEFAULT));             \
                        else                                                                                                                                            \
                            throw path_not_found("unknown path: " + path.substr(0, path.find_last_of('@')) + ALPS_STACKTRACE);                                      \
                        detail::attribute_type attribute_id(H5Aopen(parent_id, path.substr(path.find_last_of('@') + 1).c_str(), H5P_DEFAULT));                          \
                        detail::type_type type_id(H5Aget_type(attribute_id));                                                                                           \
                        detail::type_type native_id(H5Tget_native_type(type_id, H5T_DIR_ASCEND));                                                                       \
                        if (H5Tget_class(native_id) == H5T_STRING && !detail::check_error(H5Tis_variable_str(type_id)))                                                 \
                            throw std::logic_error("multidimensional dataset of fixed string datas is not implemented (" + path + ")" + ALPS_STACKTRACE);               \
                        else if (H5Tget_class(native_id) == H5T_STRING) {                                                                                               \
                            std::size_t len = std::accumulate(chunk.begin(), chunk.end(), std::size_t(1), std::multiplies<std::size_t>());                              \
                            boost::scoped_array<char *> raw(                                                                                                            \
                                new char * [len]                                                                                                                        \
                            );                                                                                                                                          \
                            if (std::equal(chunk.begin(), chunk.end(), data_size.begin())) {                                                                            \
                                detail::check_error(H5Aread(attribute_id, native_id, raw.get()));                                                                       \
                                cast(raw.get(), raw.get() + len, value);                                                                                                \
                            } else                                                                                                                                      \
                                throw std::logic_error("non continous multidimensional dataset as attributes are not implemented (" + path + ")" + ALPS_STACKTRACE);    \
                            detail::check_error(H5Dvlen_reclaim(type_id, detail::space_type(H5Aget_space(attribute_id)), H5P_DEFAULT, raw.get()));                      \
                        } else if (H5Tget_class(native_id) == H5T_STRING) {                                                                                             \
                            char ** raw = NULL;                                                                                                                         \
                            detail::check_error(H5Aread(attribute_id, native_id, raw));                                                                                 \
                            throw std::logic_error("multidimensional dataset of variable len string datas is not implemented (" + path + ")" + ALPS_STACKTRACE);        \
                        ALPS_HDF5_FOREACH_NATIVE_TYPE_INTEGRAL(ALPS_HDF5_READ_VECTOR_ATTRIBUTE_HELPER, T)                                                               \
                        } else throw wrong_type("invalid type" + ALPS_STACKTRACE);                                                                                      \
                        if (is_group(path.substr(0, path.find_last_of('@'))))                                                                                       \
                            detail::check_group(parent_id);                                                                                                             \
                        else                                                                                                                                            \
                            detail::check_data(parent_id);                                                                                                              \
                    }                                                                                                                                                   \
                }                                                                                                                                                       \
            }
        ALPS_FOREACH_NATIVE_HDF5_TYPE(ALPS_HDF5_READ_VECTOR)
        #undef ALPS_HDF5_READ_VECTOR
        #undef ALPS_HDF5_READ_VECTOR_DATA_HELPER
    }
}
