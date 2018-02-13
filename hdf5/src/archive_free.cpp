/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <iostream>
#include <string>
#include <vector>

#include <alps/hdf5/archive.hpp>
#include <alps/utilities/cast.hpp>

#include "common.hpp"
#include "archivecontext.hpp"

namespace alps {
    namespace hdf5 {

        namespace detail {

            hid_t open_attribute(archive const & ar, hid_t file_id, std::string path) {
                if ((path = ar.complete_path(path)).find_last_of('@') == std::string::npos)
                    throw invalid_path("no attribute path: " + path + ALPS_STACKTRACE);
                return H5Aopen_by_name(file_id, path.substr(0, path.find_last_of('@')).c_str(), path.substr(path.find_last_of('@') + 1).c_str(), H5P_DEFAULT, H5P_DEFAULT);
            }

            herr_t list_children_visitor(hid_t, char const * n, const H5L_info_t *, void * d) {
                reinterpret_cast<std::vector<std::string> *>(d)->push_back(n);
                return 0;
            }

            herr_t list_attributes_visitor(hid_t, char const * n, const H5A_info_t *, void * d) {
                reinterpret_cast<std::vector<std::string> *>(d)->push_back(n);
                return 0;
            }
        }

        #define ALPS_HDF5_IMPLEMENT_FREE_FUNCTIONS(T)                                                                                                                   \
            namespace detail {                                                                                                                                          \
                alps::hdf5::scalar_type< T >::type * get_pointer< T >::apply( T & value) {                                                                              \
                    return &value;                                                                                                                                      \
                }                                                                                                                                                       \
                                                                                                                                                                        \
                alps::hdf5::scalar_type< T >::type const * get_pointer< T const >::apply( T const & value) {                                                            \
                    return &value;                                                                                                                                      \
                }                                                                                                                                                       \
                                                                                                                                                                        \
                bool is_vectorizable< T >::apply(T const &) {                                                                                                           \
                    return true;                                                                                                                                        \
                }                                                                                                                                                       \
                bool is_vectorizable< T const >::apply(T &) {                                                                                                           \
                    return true;                                                                                                                                        \
                }                                                                                                                                                       \
            }                                                                                                                                                           \
                                                                                                                                                                        \
            void save(                                                                                                                                                  \
                  archive & ar                                                                                                                                        \
                , std::string const & path                                                                                                                              \
                , T const & value                                                                                                                                       \
                , std::vector<std::size_t> size                                                                                                                         \
                , std::vector<std::size_t> chunk                                                                                                                        \
                , std::vector<std::size_t> offset                                                                                                                       \
            ) {                                                                                                                                                          \
                if (!size.size())                                                                                                                                       \
                    ar.write(path, value);                                                                                                                              \
                else                                                                                                                                                    \
                    ar.write(path, get_pointer(value), size, chunk, offset);                                                                                            \
            }                                                                                                                                                           \
                                                                                                                                                                        \
            void load(                                                                                                                                                  \
                  archive & ar                                                                                                                                          \
                , std::string const & path                                                                                                                              \
                , T & value                                                                                                                                             \
                , std::vector<std::size_t> chunk                                                                                                                        \
                , std::vector<std::size_t> offset                                                                                                                       \
            ) {                                                                                                                                                         \
                if (!chunk.size())                                                                                                                                      \
                    ar.read(path, value);                                                                                                                               \
                else                                                                                                                                                    \
                    ar.read(path, get_pointer(value), chunk, offset);                                                                                                   \
            }

        ALPS_FOREACH_NATIVE_HDF5_TYPE(ALPS_HDF5_IMPLEMENT_FREE_FUNCTIONS)
        #undef ALPS_HDF5_IMPLEMENT_FREE_FUNCTIONS

        namespace detail {
            template<typename T> struct is_datatype_impl_compare {
                static bool apply(type_type const & native_id) {
                    return check_error(
                        H5Tequal(type_type(H5Tcopy(native_id)), type_type(get_native_type(typename alps::detail::type_wrapper<T>::type())))
                    ) > 0;
                }
            };
            template<> struct is_datatype_impl_compare<std::string> {
                static bool apply(type_type const & native_id) {
                    return H5Tget_class(native_id) == H5T_STRING;
                }
            };
        }

        template<typename T>
        auto archive::is_datatype_impl(std::string path, T) const -> ONLY_NATIVE(T, bool) {
            ALPS_HDF5_FAKE_THREADSAFETY
            hid_t type_id;
            path = complete_path(path);
            if (context_ == NULL)
                throw archive_closed("the archive is closed" + ALPS_STACKTRACE);
            if (path.find_last_of('@') != std::string::npos && is_attribute(path)) {
                detail::attribute_type attr_id(detail::open_attribute(*this, context_->file_id_, path));
                type_id = H5Aget_type(attr_id);
            } else if (path.find_last_of('@') == std::string::npos && is_data(path)) {
                detail::data_type data_id(H5Dopen2(context_->file_id_, path.c_str(), H5P_DEFAULT));
                type_id = H5Dget_type(data_id);
            } else
                throw path_not_found("no valid path: " + path + ALPS_STACKTRACE);
            detail::type_type native_id(H5Tget_native_type(type_id, H5T_DIR_ASCEND));
            detail::check_type(type_id);
            {
                ALPS_HDF5_LOCK_MUTEX
                return detail::is_datatype_impl_compare< T >::apply(native_id);
            }
        }
        #define ALPS_HDF5_IS_DATATYPE_IMPL_IMPL(T) template bool archive::is_datatype_impl<T>(std::string, T) const;
        ALPS_FOREACH_NATIVE_HDF5_TYPE(ALPS_HDF5_IS_DATATYPE_IMPL_IMPL)
    }
}
