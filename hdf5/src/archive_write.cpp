/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <iostream>

#include <hdf5.h>

#include <alps/hdf5/archive.hpp>
#include <alps/utilities/cast.hpp>

#include "common.hpp"
#include "archivecontext.hpp"

namespace alps {
    namespace hdf5 {

        template<typename T>
        auto archive::write(std::string path, T value) const -> ONLY_NATIVE(T, void) {
            ALPS_HDF5_FAKE_THREADSAFETY
            if (context_ == NULL)
                throw archive_closed("the archive is closed" + ALPS_STACKTRACE);
            if (!context_->write_)
                throw archive_error("the archive is not writeable" + ALPS_STACKTRACE);
            hid_t data_id;
            if ((path = complete_path(path)).find_last_of('@') == std::string::npos) {
                if (is_group(path))
                    delete_group(path);
                data_id = H5Dopen2(context_->file_id_, path.c_str(), H5P_DEFAULT);
                if (data_id < 0) {
                    if (path.find_last_of('/') < std::string::npos && path.find_last_of('/') > 0)
                        create_group(path.substr(0, path.find_last_of('/')));
                } else {
                    H5S_class_t class_type;
                    {
                        detail::space_type current_space_id(H5Dget_space(data_id));
                        class_type = H5Sget_simple_extent_type(current_space_id);
                    }
                    if (class_type != H5S_SCALAR || !is_datatype<T>(path)) {
                        detail::check_data(data_id);
                        if (path.find_last_of('/') < std::string::npos && path.find_last_of('/') > 0) {
                            detail::group_type group_id(H5Gopen2(context_->file_id_, path.substr(0, path.find_last_of('/')).c_str(), H5P_DEFAULT));
                            detail::check_error(H5Ldelete(group_id, path.substr(path.find_last_of('/') + 1).c_str(), H5P_DEFAULT));
                        } else
                            detail::check_error(H5Ldelete(context_->file_id_, path.c_str(), H5P_DEFAULT));
                        data_id = -1;
                    }
                }
                detail::type_type type_id(detail::get_native_type(T()));
                if (data_id < 0) {
                    detail::property_type prop_id(H5Pcreate(H5P_DATASET_CREATE));
                    detail::check_error(H5Pset_attr_creation_order(prop_id, (H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED)));
                    data_id = H5Dcreate2(
                          context_->file_id_
                        , path.c_str()
                        , type_id
                        , detail::space_type(H5Screate(H5S_SCALAR))
                        , H5P_DEFAULT
                        , prop_id
                        , H5P_DEFAULT
                    );
                }
                detail::native_ptr_converter<typename std::remove_cv<typename std::remove_reference<T>::type>::type> converter(1);
                detail::check_error(H5Dwrite(data_id, type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, converter.apply(&value)));
                detail::check_data(data_id);
            } else {
                hid_t parent_id;
                if (is_group(path.substr(0, path.find_last_of('@'))))
                    parent_id = detail::check_error(H5Gopen2(context_->file_id_, path.substr(0, path.find_last_of('@')).c_str(), H5P_DEFAULT));
                else if (is_data(path.substr(0, path.find_last_of('@'))))
                    parent_id = detail::check_error(H5Dopen2(context_->file_id_, path.substr(0, path.find_last_of('@')).c_str(), H5P_DEFAULT));
                else
                    throw path_not_found("unknown path: " + path.substr(0, path.find_last_of('@')) + ALPS_STACKTRACE);
                hid_t data_id = H5Aopen(parent_id, path.substr(path.find_last_of('@') + 1).c_str(), H5P_DEFAULT);
                if (data_id >= 0) {
                    H5S_class_t class_type;
                    {
                        detail::space_type current_space_id(H5Aget_space(data_id));
                        class_type = H5Sget_simple_extent_type(current_space_id);
                    }
                    if (class_type != H5S_SCALAR || !is_datatype<T>(path)) {
                        detail::check_attribute(data_id);
                        detail::check_error(H5Adelete(parent_id, path.substr(path.find_last_of('@') + 1).c_str()));
                        data_id = -1;
                    }
                }
                detail::type_type type_id(detail::get_native_type(T()));
                if (data_id < 0)
                    data_id = H5Acreate2(
                          parent_id
                        , path.substr(path.find_last_of('@') + 1).c_str()
                        , type_id
                        , detail::space_type(H5Screate(H5S_SCALAR))
                        , H5P_DEFAULT
                        , H5P_DEFAULT
                    );
                detail::native_ptr_converter<typename std::remove_cv<typename std::remove_reference<T>::type>::type> converter(1);
                detail::check_error(H5Awrite(data_id, type_id, converter.apply(&value)));
                detail::attribute_type attr_id(data_id);
                if (is_group(path.substr(0, path.find_last_of('@'))))
                    detail::check_group(parent_id);
                else
                    detail::check_data(parent_id);
            }
        }
        #define ALPS_HDF5_WRITE_SCALAR(T) template void archive::write<T>(std::string path, T value) const;
        ALPS_FOREACH_NATIVE_HDF5_TYPE(ALPS_HDF5_WRITE_SCALAR)

        template<typename T>
        auto archive::write(
            std::string path, T const * value, std::vector<std::size_t> size, std::vector<std::size_t> chunk, std::vector<std::size_t> offset
        ) const -> ONLY_NATIVE(T, void) {
            ALPS_HDF5_FAKE_THREADSAFETY
            if (context_ == NULL)
                throw archive_closed("the archive is closed" + ALPS_STACKTRACE);
            if (!context_->write_)
                throw archive_error("the archive is not writeable" + ALPS_STACKTRACE);
            if (chunk.size() == 0)
                chunk = std::vector<std::size_t>(size.begin(), size.end());
            if (offset.size() == 0)
                offset = std::vector<std::size_t>(size.size(), 0);
            if (size.size() != offset.size())
                throw archive_error("wrong chunk or offset passed for path: " + path + ALPS_STACKTRACE);
            hid_t data_id;
            if ((path = complete_path(path)).find_last_of('@') == std::string::npos) {
                if (is_group(path))
                    delete_group(path);
                data_id = H5Dopen2(context_->file_id_, path.c_str(), H5P_DEFAULT);
                if (data_id < 0) {
                    if (path.find_last_of('/') < std::string::npos && path.find_last_of('/') > 0)
                        create_group(path.substr(0, path.find_last_of('/')));
                } else {
                    H5S_class_t class_type;
                    {
                        detail::space_type current_space_id(H5Dget_space(data_id));
                        class_type = H5Sget_simple_extent_type(current_space_id);
                    }
                    if (
                            class_type == H5S_SCALAR
                        || dimensions(path) != size.size()
                        || !std::equal(size.begin(), size.end(), extent(path).begin())
                        || !is_datatype<T>(path)
                    ) {
                        detail::check_data(data_id);
                        detail::check_error(H5Ldelete(context_->file_id_, path.c_str(), H5P_DEFAULT));
                        data_id = -1;
                    }
                }
                detail::type_type type_id(detail::get_native_type(T()));
                if (std::accumulate(size.begin(), size.end(), std::size_t(0)) == 0) {
                    if (data_id < 0) {
                        detail::property_type prop_id(H5Pcreate(H5P_DATASET_CREATE));
                        detail::check_error(H5Pset_attr_creation_order(prop_id, (H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED)));
                        detail::check_data(H5Dcreate2(
                              context_->file_id_
                            , path.c_str()
                            , type_id
                            , detail::space_type(H5Screate(H5S_NULL))
                            , H5P_DEFAULT
                            , prop_id
                            , H5P_DEFAULT
                        ));
                    } else
                        detail::check_data(data_id);
                } else {
                    std::vector<hsize_t> size_hid(size.begin(), size.end())
                                        , offset_hid(offset.begin(), offset.end())
                                        , chunk_hid(chunk.begin(), chunk.end());
                    if (data_id < 0) {
                        detail::property_type prop_id(H5Pcreate(H5P_DATASET_CREATE));
                        detail::check_error(H5Pset_attr_creation_order(prop_id, (H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED)));
                        if (std::is_same< T , std::string>::value)
                            detail::check_error(data_id = H5Dcreate2(
                                  context_->file_id_
                                , path.c_str()
                                , type_id
                                , detail::space_type(H5Screate_simple(static_cast<int>(size_hid.size()), &size_hid.front(), NULL))
                                , H5P_DEFAULT
                                , prop_id
                                , H5P_DEFAULT
                            ));
                        else {
                            detail::check_error(H5Pset_fill_time(prop_id, H5D_FILL_TIME_NEVER));
                            std::size_t dataset_size = std::accumulate(size.begin(), size.end(), std::size_t(sizeof( T )), std::multiplies<std::size_t>());
                            if (dataset_size < ALPS_HDF5_SZIP_BLOCK_SIZE * sizeof( T ))
                                detail::check_error(H5Pset_layout(prop_id, H5D_COMPACT));
                            else if (dataset_size < (1ULL<<32))
                                detail::check_error(H5Pset_layout(prop_id, H5D_CONTIGUOUS));
                            else {
                                detail::check_error(H5Pset_layout(prop_id, H5D_CHUNKED));
                                std::vector<hsize_t> max_chunk(size_hid);
                                std::size_t index = 0;
                                while (std::accumulate(
                                      max_chunk.begin()
                                    , max_chunk.end()
                                    , std::size_t(sizeof( T ))
                                    , std::multiplies<std::size_t>()
                                ) > (1ULL<<32) - 1) {
                                    max_chunk[index] /= 2;
                                    if (max_chunk[index] == 1)
                                        ++index;
                                }
                                detail::check_error(H5Pset_chunk(prop_id, static_cast<int>(max_chunk.size()), &max_chunk.front()));
                            }
                            if (context_->compress_ && dataset_size > ALPS_HDF5_SZIP_BLOCK_SIZE * sizeof( T ))
                                detail::check_error(H5Pset_szip(prop_id, H5_SZIP_NN_OPTION_MASK, ALPS_HDF5_SZIP_BLOCK_SIZE));
                            detail::check_error(H5Pset_attr_creation_order(prop_id, (H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED)));
                            detail::check_error(data_id = H5Dcreate2(
                                  context_->file_id_
                                , path.c_str()
                                , type_id
                                , detail::space_type(H5Screate_simple(static_cast<int>(size_hid.size()), &size_hid.front(), NULL))
                                , H5P_DEFAULT
                                , prop_id
                                , H5P_DEFAULT
                            ));
                        }
                    }
                    detail::data_type raii_id(data_id);
                    detail::native_ptr_converter<T> converter(std::accumulate(chunk.begin(), chunk.end(), std::size_t(1), std::multiplies<std::size_t>()));
                    if (std::equal(chunk.begin(), chunk.end(), size.begin()))
                        detail::check_error(H5Dwrite(raii_id, type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, converter.apply(value)));
                    else {
                        detail::space_type space_id(H5Dget_space(raii_id));
                        detail::check_error(H5Sselect_hyperslab(space_id, H5S_SELECT_SET, &offset_hid.front(), NULL, &chunk_hid.front(), NULL));
                        detail::space_type mem_id(detail::space_type(H5Screate_simple(static_cast<int>(chunk_hid.size()), &chunk_hid.front(), NULL)));
                        detail::check_error(H5Dwrite(raii_id, type_id, mem_id, space_id, H5P_DEFAULT, converter.apply(value)));
                    }
                }
            } else {
                hid_t parent_id;
                if (is_group(path.substr(0, path.find_last_of('@'))))
                    parent_id = detail::check_error(H5Gopen2(context_->file_id_, path.substr(0, path.find_last_of('@')).c_str(), H5P_DEFAULT));
                else if (is_data(path.substr(0, path.find_last_of('@'))))
                    parent_id = detail::check_error(H5Dopen2(context_->file_id_, path.substr(0, path.find_last_of('@')).c_str(), H5P_DEFAULT));
                else
                    throw path_not_found("unknown path: " + path.substr(0, path.find_last_of('@')) + ALPS_STACKTRACE);
                hid_t data_id = H5Aopen(parent_id, path.substr(path.find_last_of('@') + 1).c_str(), H5P_DEFAULT);
                if (data_id >= 0) {
                    H5S_class_t class_type;
                    {
                        detail::space_type current_space_id(H5Aget_space(data_id));
                        class_type = H5Sget_simple_extent_type(current_space_id);
                    }
                    if (class_type != H5S_SCALAR) {
                        detail::check_attribute(data_id);
                        detail::check_error(H5Adelete(parent_id, path.substr(path.find_last_of('@') + 1).c_str()));
                        data_id = -1;
                    }
                }
                detail::type_type type_id(detail::get_native_type(T()));
                if (std::accumulate(size.begin(), size.end(), std::size_t(0)) == 0) {
                    if (data_id < 0)
                        detail::check_attribute(H5Acreate2(
                              parent_id
                            , path.substr(path.find_last_of('@') + 1).c_str()
                            , type_id
                            , detail::space_type(H5Screate(H5S_NULL))
                            , H5P_DEFAULT
                            , H5P_DEFAULT
                        ));
                    else
                        detail::check_attribute(data_id);
                } else {
                    std::vector<hsize_t> size_hid(size.begin(), size.end())
                                        , offset_hid(offset.begin(), offset.end())
                                        , chunk_hid(chunk.begin(), chunk.end());
                    if (data_id < 0)
                        data_id = detail::check_error(H5Acreate2(
                              parent_id
                            , path.substr(path.find_last_of('@') + 1).c_str()
                            , type_id
                            , detail::space_type(H5Screate_simple(static_cast<int>(size_hid.size()), &size_hid.front(), NULL))
                            , H5P_DEFAULT
                            , H5P_DEFAULT
                        ));
                    {
                        detail::attribute_type raii_id(data_id);
                        if (std::equal(chunk.begin(), chunk.end(), size.begin())) {
                            detail::native_ptr_converter<T> converter(
                                std::accumulate(chunk.begin(), chunk.end(), std::size_t(1), std::multiplies<std::size_t>())
                            );
                            detail::check_error(H5Awrite(raii_id, type_id, converter.apply(value)));
                        } else
                            throw std::logic_error("Not Implemented, path: " + path + ALPS_STACKTRACE);
                    }
                }
                if (is_group(path.substr(0, path.find_last_of('@'))))
                    detail::check_group(parent_id);
                else
                    detail::check_data(parent_id);
            }
        }
        #define ALPS_HDF5_WRITE_VECTOR(T) template void archive::write<T>(                                                \
            std::string, T const *, std::vector<std::size_t>, std::vector<std::size_t>, std::vector<std::size_t>) const;
        ALPS_FOREACH_NATIVE_HDF5_TYPE(ALPS_HDF5_WRITE_VECTOR)
    }
}
