/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/hdf5/archive.hpp>
#include <alps/utilities/cast.hpp>
#include <alps/utilities/signal.hpp>
#include <alps/utilities/stacktrace.hpp>

#include <boost/scoped_array.hpp>

#include <hdf5.h>

#include <sstream>
#include <fstream>
#include <iostream>
#include <typeinfo>

#include "common.hpp"
#include "archivecontext.hpp"

namespace alps {
    namespace hdf5 {

        namespace detail {
            hid_t open_attribute(archive const & ar, hid_t file_id, std::string path);
            herr_t list_children_visitor(hid_t, char const * n, const H5L_info_t *, void * d);
            herr_t list_attributes_visitor(hid_t, char const * n, const H5A_info_t *, void * d);
        }

        archive::archive() : context_(NULL) {}

        archive::archive(std::string const & filename, int prop) : context_(NULL) {
            std::cerr << "WARNING: Use of `archive(string name, int mode)` constructor (name=\""
                      << filename << "\") is DEPRECATED!\n";

            std::string mode="";
            if (prop & COMPRESS) mode += "c";
            if (prop & MEMORY) mode += "m";

            prop = prop & ~(COMPRESS|MEMORY);

            if (prop == READ) {
                mode += "r";
            } else if ((prop == WRITE) || (prop == REPLACE) || (prop == (WRITE|REPLACE))) {
                mode += "w";
            } else {
                throw wrong_mode("Unsupported mode flags when openinge file '"+filename+"'" + ALPS_STACKTRACE);
            }
            std::cerr << "WARNING: Use of `archive(string name, string mode) constructor with mode=\""
                      << mode << "\" instead!\n";
            open(filename,mode);
        }

        archive::archive(std::string const & filename, std::string mode) : context_(NULL) {
            open(filename, mode);
        }

        archive::archive(archive const & arg)
            : current_(arg.current_)
            , context_(arg.context_)
        {
            if (context_ != NULL) {
                ALPS_HDF5_LOCK_MUTEX
                ++ref_cnt_[file_key(context_->filename_, context_->memory_)].second;
            }
        }

        archive::~archive() {
            if (context_ != NULL)
                try {
                    close();
                } catch (std::exception & ex) {
                    std::cerr << "Error destructing archive of file '" << ex.what() << std::endl;
                    std::abort();
                }
        }

/*  This method does not seem to be ever used
************************************
*        void archive::abort() {
*            // Do not use a lock here, else deadlocking is really likly
*            for (std::map<std::string, std::pair<detail::archivecontext *, std::size_t> >::iterator it = ref_cnt_.begin(); it != ref_cnt_.end(); ++it) {
*                bool replace = it->second.first->replace_;
*                std::string filename = it->second.first->filename_;
*                it->second.first->replace_ = false;
*                delete it->second.first;
*                if (replace && boost::filesystem::exists(filename))
*                    boost::filesystem::remove(filename);
*            }
*            ref_cnt_.clear();
*        }
*************************************/

        void archive::close() {
            if (context_ == NULL)
                throw archive_closed("the archive is closed" + ALPS_STACKTRACE);
            ALPS_HDF5_LOCK_MUTEX
            H5Fflush(context_->file_id_, H5F_SCOPE_GLOBAL);
            if (!--ref_cnt_[file_key(context_->filename_, context_->memory_)].second) {
                ref_cnt_.erase(file_key(context_->filename_, context_->memory_));
                delete context_;
            }
            context_ = NULL;
        }

        void archive::open(const std::string & filename, const std::string &mode) {
            if(is_open())
                throw archive_opened("the archive '"+ filename + "' is already opened" + ALPS_STACKTRACE);
            if (mode.find_first_not_of("rwacm")!=std::string::npos)
                throw wrong_mode("Incorrect mode '"+mode+"' opening file '"+filename+"'" + ALPS_STACKTRACE);

            construct(filename,
                      (mode.find_last_of('w') == std::string::npos ? 0 : WRITE) //@todo FIXME_DEBOOST: "w" is equiv to "a"
                      | (mode.find_last_of('a') == std::string::npos ? 0 : WRITE)
                      | (mode.find_last_of('c') == std::string::npos ? 0 : COMPRESS)
                      | (mode.find_last_of('m') == std::string::npos ? 0 : MEMORY)
            );
        }

        bool archive::is_open() {
            return context_ != NULL;
        }

        std::string const & archive::get_filename() const {
            if (context_ == NULL)
                throw archive_closed("the archive is closed" + ALPS_STACKTRACE);
            return context_->filename_;
        }

        std::string archive::encode_segment(std::string segment) const {
            char chars[] = {'&', '/'};
            for (std::size_t i = 0; i < sizeof(chars); ++i)
                for (std::size_t pos = segment.find_first_of(chars[i]); pos < std::string::npos; pos = segment.find_first_of(chars[i], pos + 1))
                    segment = segment.substr(0, pos) + "&#" + cast<std::string>(static_cast<int>(chars[i])) + ";" + segment.substr(pos + 1);
            return segment;
        }

        std::string archive::decode_segment(std::string segment) const {
            for (std::size_t pos = segment.find_first_of('&'); pos < std::string::npos; pos = segment.find_first_of('&', pos + 1))
                segment = segment.substr(0, pos)
                        + static_cast<char>(cast<int>(segment.substr(pos + 2, segment.find_first_of(';', pos) - pos - 2)))
                        + segment.substr(segment.find_first_of(';', pos) + 1);
            return segment;
        }

        std::string archive::get_context() const {
            return current_;
        }

        void archive::set_context(std::string const & context) {
            ALPS_HDF5_LOCK_MUTEX
            current_ = complete_path(context);
        }

        std::string archive::complete_path(std::string path) const {
            if (path.size() > 1 && *path.rbegin() == '/')
                path = path.substr(0, path.size() - 1);
            if (path.size() && path[0] == '/')
                return path;
            else if (path.size() < 2 || path.substr(0, 2) != "..")
                return current_ + (current_.size() == 1 || !path.size() ? "" : "/") + path;
            else {
                std::string ctx = current_;
                while (ctx.size() && path.size() && path.substr(0, 2) == "..") {
                    ctx = ctx.substr(0, ctx.find_last_of('/'));
                    path = path.size() == 2 ? "" : path.substr(3);
                }
                return ctx + (ctx.size() == 1 || !path.size() ? "" : "/") + path;
            }
        }

        bool archive::is_data(std::string path) const {
            if (context_ == NULL)
                throw archive_closed("the archive is closed" + ALPS_STACKTRACE);
            if ((path = complete_path(path)).find_last_of('@') != std::string::npos)
                throw invalid_path("no data path: " + path + ALPS_STACKTRACE);
            ALPS_HDF5_FAKE_THREADSAFETY
            hid_t id = H5Dopen2(context_->file_id_, path.c_str(), H5P_DEFAULT);
            return id < 0 ? false : detail::check_data(id) != 0;
        }

        bool archive::is_attribute(std::string path) const {
            if (context_ == NULL)
                throw archive_closed("the archive is closed" + ALPS_STACKTRACE);
            if ((path = complete_path(path)).find_last_of('@') == std::string::npos)
                return false;
            ALPS_HDF5_FAKE_THREADSAFETY
            return detail::check_error(H5Aexists_by_name(context_->file_id_, path.substr(0, path.find_last_of('@')).c_str(), path.substr(path.find_last_of('@') + 1).c_str(), H5P_DEFAULT));
        }

        bool archive::is_group(std::string path) const {
            if (context_ == NULL)
                throw archive_closed("the archive is closed" + ALPS_STACKTRACE);
            if ((path = complete_path(path)).find_last_of('@') != std::string::npos)
                return false;
            ALPS_HDF5_FAKE_THREADSAFETY
            hid_t id = H5Gopen2(context_->file_id_, path.c_str(), H5P_DEFAULT);
            return id < 0 ? false : detail::check_group(id) != 0;
        }

        bool archive::is_scalar(std::string path) const {
            hid_t space_id;
            if (context_ == NULL)
                throw archive_closed("the archive is closed" + ALPS_STACKTRACE);
            ALPS_HDF5_FAKE_THREADSAFETY
            if ((path = complete_path(path)).find_last_of('@') != std::string::npos && is_attribute(path)) {
                detail::attribute_type attr_id(detail::open_attribute(*this, context_->file_id_, path));
                space_id = H5Aget_space(attr_id);
            } else if (path.find_last_of('@') == std::string::npos && is_data(path)) {
                detail::data_type data_id(H5Dopen2(context_->file_id_, path.c_str(), H5P_DEFAULT));
                space_id = H5Dget_space(data_id);
            } else
                #ifdef ALPS_HDF5_READ_GREEDY
                    return false;
                #else
                    throw path_not_found("error reading path " + path + ALPS_STACKTRACE);
                #endif
            H5S_class_t type = H5Sget_simple_extent_type(space_id);
            detail::check_space(space_id);
            if (type == H5S_NO_CLASS)
                throw archive_error("error reading class " + path + ALPS_STACKTRACE);
            return type == H5S_SCALAR;
        }

        bool archive::is_null(std::string path) const {
            hid_t space_id;
            if (context_ == NULL)
                throw archive_closed("the archive is closed" + ALPS_STACKTRACE);
            ALPS_HDF5_FAKE_THREADSAFETY
            if ((path = complete_path(path)).find_last_of('@') != std::string::npos) {
                detail::attribute_type attr_id(detail::open_attribute(*this, context_->file_id_, path));
                space_id = H5Aget_space(attr_id);
            } else {
                detail::data_type data_id(H5Dopen2(context_->file_id_, path.c_str(), H5P_DEFAULT));
                space_id = H5Dget_space(data_id);
            }
            H5S_class_t type = H5Sget_simple_extent_type(space_id);
            detail::check_space(space_id);
            if (type == H5S_NO_CLASS)
                throw archive_error("error reading class " + path + ALPS_STACKTRACE);
            return type == H5S_NULL;
        }

        bool archive::is_complex(std::string path) const {
            if (context_ == NULL)
                throw archive_closed("the archive is closed" + ALPS_STACKTRACE);
            ALPS_HDF5_FAKE_THREADSAFETY
            if ((path = complete_path(path)).find_last_of('@') != std::string::npos)
                return is_attribute(path.substr(0, path.find_last_of('@')) + "@__complex__:" + path.substr(path.find_last_of('@') + 1))
                    && is_scalar(path.substr(0, path.find_last_of('@')) + "@__complex__:" + path.substr(path.find_last_of('@') + 1));
            else if (is_group(path)) {
                std::vector<std::string> children = list_children(path);
                for (std::size_t i = 0; i < children.size(); ++i)
                    if (is_complex(path + "/" + children[i]))
                        return true;
                return false;
            } else
                return is_attribute(path + "/@__complex__") && is_scalar(path + "/@__complex__");
        }

        std::vector<std::string> archive::list_children(std::string path) const {
            if (context_ == NULL)
                throw archive_closed("the archive is closed" + ALPS_STACKTRACE);
            if ((path = complete_path(path)).find_last_of('@') != std::string::npos)
                throw invalid_path("no group path: " + path + ALPS_STACKTRACE);
            std::vector<std::string> list;
            ALPS_HDF5_FAKE_THREADSAFETY
            if (!is_group(path))
                throw path_not_found("The group '" + path + "' does not exist." + ALPS_STACKTRACE);
            detail::group_type group_id(H5Gopen2(context_->file_id_, path.c_str(), H5P_DEFAULT));
            detail::check_error(H5Literate(group_id, H5_INDEX_NAME, H5_ITER_NATIVE, NULL, detail::list_children_visitor, &list));
            return list;
        }

        std::vector<std::string> archive::list_attributes(std::string path) const {
            if (context_ == NULL)
                throw archive_closed("the archive is closed" + ALPS_STACKTRACE);
            if ((path = complete_path(path)).find_last_of('@') != std::string::npos)
                throw invalid_path("no group or data path: " + path + ALPS_STACKTRACE);
            std::vector<std::string> list;
            ALPS_HDF5_FAKE_THREADSAFETY
            if (is_group(path)) {
                detail::group_type id(H5Gopen2(context_->file_id_, path.c_str(), H5P_DEFAULT));
                detail::check_error(H5Aiterate2(id, H5_INDEX_CRT_ORDER, H5_ITER_NATIVE, NULL, detail::list_attributes_visitor, &list));
            } else if (is_data(path)) {
                detail::data_type id(H5Dopen2(context_->file_id_, path.c_str(), H5P_DEFAULT));
                detail::check_error(H5Aiterate2(id, H5_INDEX_CRT_ORDER, H5_ITER_NATIVE, NULL, detail::list_attributes_visitor, &list));
            } else
                throw path_not_found("The path '" + path + "' does not exist." + ALPS_STACKTRACE);
            return list;
        }

        std::vector<std::size_t> archive::extent(std::string path) const {
            if (context_ == NULL)
                throw archive_closed("the archive is closed" + ALPS_STACKTRACE);
            if (is_null(path = complete_path(path)))
                return std::vector<std::size_t>(1, 0);
            else if (is_scalar(path))
                return std::vector<std::size_t>(1, 1);
            std::vector<hsize_t> buffer(dimensions(path), 0);
            hid_t space_id;
            ALPS_HDF5_FAKE_THREADSAFETY
            if (path.find_last_of('@') != std::string::npos) {
                detail::attribute_type attr_id(detail::open_attribute(*this, context_->file_id_, path));
                space_id = H5Aget_space(attr_id);
            } else {
                detail::data_type data_id(H5Dopen2(context_->file_id_, path.c_str(), H5P_DEFAULT));
                space_id = H5Dget_space(data_id);
            }
            detail::check_error(H5Sget_simple_extent_dims(space_id, &buffer.front(), NULL));
            detail::check_space(space_id);
            std::vector<std::size_t> extent(buffer.begin(), buffer.end());
            return extent;
        }

        std::size_t archive::dimensions(std::string path) const {
            if (context_ == NULL)
                throw archive_closed("the archive is closed" + ALPS_STACKTRACE);
            ALPS_HDF5_FAKE_THREADSAFETY
            if ((path = complete_path(path)).find_last_of('@') != std::string::npos) {
                detail::attribute_type attr_id(detail::open_attribute(*this, context_->file_id_, path));
                return detail::check_error(H5Sget_simple_extent_dims(detail::space_type(H5Aget_space(attr_id)), NULL, NULL));
            } else {
                detail::data_type data_id(H5Dopen2(context_->file_id_, path.c_str(), H5P_DEFAULT));
                return detail::check_error(H5Sget_simple_extent_dims(detail::space_type(H5Dget_space(data_id)), NULL, NULL));
            }
        }

        void archive::create_group(std::string path) const {
            if (context_ == NULL)
                throw archive_closed("the archive is closed" + ALPS_STACKTRACE);
            if ((path = complete_path(path)).find_last_of('@') != std::string::npos)
                throw invalid_path("no group path: " + path + ALPS_STACKTRACE);
            ALPS_HDF5_FAKE_THREADSAFETY
            if (is_data(path))
                delete_data(path);
            if (!is_group(path)) {
                std::size_t pos;
                hid_t group_id = -1;
                for (pos = path.find_last_of('/'); group_id < 0 && pos > 0 && pos < std::string::npos; pos = path.find_last_of('/', pos - 1))
                    group_id = H5Gopen2(context_->file_id_, path.substr(0, pos).c_str(), H5P_DEFAULT);
                if (group_id < 0) {
                    if ((pos = path.find_first_of('/', 1)) != std::string::npos) {
                        detail::property_type prop_id(H5Pcreate(H5P_GROUP_CREATE));
                        detail::check_error(H5Pset_link_creation_order(prop_id, (H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED)));
                        detail::check_error(H5Pset_attr_creation_order(prop_id, (H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED)));
                        detail::check_group(H5Gcreate2(context_->file_id_, path.substr(0, pos).c_str(), H5P_DEFAULT, prop_id, H5P_DEFAULT));
                    }
                } else {
                    pos = path.find_first_of('/', pos + 1);
                    detail::check_group(group_id);
                }
                while (pos != std::string::npos && (pos = path.find_first_of('/', pos + 1)) != std::string::npos && pos > 0) {
                    detail::property_type prop_id(H5Pcreate(H5P_GROUP_CREATE));
                    detail::check_error(H5Pset_link_creation_order(prop_id, (H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED)));
                    detail::check_error(H5Pset_attr_creation_order(prop_id, (H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED)));
                    detail::check_group(H5Gcreate2(context_->file_id_, path.substr(0, pos).c_str(), H5P_DEFAULT, prop_id, H5P_DEFAULT));
                }
                detail::property_type prop_id(H5Pcreate(H5P_GROUP_CREATE));
                detail::check_error(H5Pset_link_creation_order(prop_id, (H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED)));
                detail::check_error(H5Pset_attr_creation_order(prop_id, (H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED)));
                detail::check_group(H5Gcreate2(context_->file_id_, path.c_str(), H5P_DEFAULT, prop_id, H5P_DEFAULT));
            }
        }

        void archive::delete_data(std::string path) const {
            if (context_ == NULL)
                throw archive_closed("the archive is closed" + ALPS_STACKTRACE);
            if ((path = complete_path(path)).find_last_of('@') != std::string::npos)
                throw invalid_path("no data path: " + path + ALPS_STACKTRACE);
            ALPS_HDF5_FAKE_THREADSAFETY
            if (is_data(path))
                detail::check_error(H5Ldelete(context_->file_id_, path.c_str(), H5P_DEFAULT));
            else if (is_group(path))
                throw invalid_path("the path contains a group: " + path + ALPS_STACKTRACE);
        }

        void archive::delete_group(std::string path) const  {
            if (context_ == NULL)
                throw archive_closed("the archive is closed" + ALPS_STACKTRACE);
            if ((path = complete_path(path)).find_last_of('@') != std::string::npos)
                throw invalid_path("no group path: " + path + ALPS_STACKTRACE);
            ALPS_HDF5_FAKE_THREADSAFETY
            if (is_group(path))
                detail::check_error(H5Ldelete(context_->file_id_, path.c_str(), H5P_DEFAULT));
            else if (is_data(path))
                throw invalid_path("the path contains a dataset: " + path + ALPS_STACKTRACE);
        }

        void archive::delete_attribute(std::string path) const {
            if (context_ == NULL)
                throw archive_closed("the archive is closed" + ALPS_STACKTRACE);
            if ((path = complete_path(path)).find_last_of('@') == std::string::npos)
                throw invalid_path("no attribute path: " + path + ALPS_STACKTRACE);
            // TODO: implement
            throw std::logic_error("Not implemented!" + ALPS_STACKTRACE);
        }

        void archive::set_complex(std::string path) {
            if (context_ == NULL)
                throw archive_closed("the archive is closed" + ALPS_STACKTRACE);
            ALPS_HDF5_FAKE_THREADSAFETY
            if (path.find_last_of('@') != std::string::npos)
                write(path.substr(0, path.find_last_of('@')) + "@__complex__:" + path.substr(path.find_last_of('@') + 1), true);
            else {
                if (is_group(path)) {
                    std::vector<std::string> children = list_children(path);
                    for (std::vector<std::string>::const_iterator it = children.begin(); it != children.end(); ++it)
                        set_complex(path + "/" + *it);
                } else
                    write(path + "/@__complex__", true);
            }
        }

        detail::archive_proxy<archive> archive::operator[](std::string const & path) {
            return detail::archive_proxy<archive>(path, *this);
        }

        void archive::construct(std::string const & filename, std::size_t props) {
            ALPS_HDF5_LOCK_MUTEX
            detail::check_error(H5Eset_auto2(H5E_DEFAULT, NULL, NULL));
            if (props & COMPRESS) {
                unsigned int flag;
                detail::check_error(H5Zget_filter_info(H5Z_FILTER_SZIP, &flag));
                props &= (flag & H5Z_FILTER_CONFIG_ENCODE_ENABLED ? ~0x00 : ~COMPRESS);
            }
            if (ref_cnt_.find(file_key(filename, props & MEMORY)) == ref_cnt_.end())
                ref_cnt_.insert(std::make_pair(
                      file_key(filename, props & MEMORY)
                    , std::make_pair(context_ = new detail::archivecontext(filename, props & WRITE, props & REPLACE, props & COMPRESS, props & MEMORY), 1)
                ));
            else {
                context_ = ref_cnt_.find(file_key(filename, props & MEMORY))->second.first;
                context_->grant(props & WRITE, props & REPLACE);
                ++ref_cnt_.find(file_key(filename, props & MEMORY))->second.second;
            }
        }

        std::string archive::file_key(std::string filename, bool memory) const {
            return (memory ? "m" : "_") + filename;
        }

#ifndef ALPS_SINGLE_THREAD
        boost::recursive_mutex archive::mutex_;
#endif
        std::map<std::string, std::pair<detail::archivecontext *, std::size_t> > archive::ref_cnt_;
    }
}

#undef ALPS_HDF5_FOREACH_NATIVE_TYPE_INTEGRAL
