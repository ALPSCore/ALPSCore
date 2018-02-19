/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <iostream>
#include <fstream>

#include <alps/utilities/signal.hpp>
#include <alps/utilities/stacktrace.hpp>
#include <alps/hdf5/errors.hpp>
#include <alps/utilities/cast.hpp>

#include "common.hpp"
#include "archivecontext.hpp"

namespace alps {
    namespace hdf5 {
        namespace detail {

            archivecontext::archivecontext(std::string const & filename, bool write, bool replace, bool compress, bool memory)
                : compress_(compress)
                , write_(write || replace)
                , replace_(!memory && replace)
                , memory_(memory)
                , filename_(filename)
                , filename_new_(filename)
            {
                construct();
            }

            archivecontext::~archivecontext() {
                destruct(true);
            }

            void archivecontext::grant(bool write, bool replace) {
                if (!write_ && (write || replace)) {
                    destruct(false);
                    write_ = write || replace;
                    replace_ = !memory_ && replace;
                    construct();
                }
            }

            void archivecontext::construct() {
                alps::signal::listen();
                if (memory_) {
                    property_type prop_id(H5Pcreate(H5P_FILE_ACCESS));
                    check_error(H5Pset_fapl_core(prop_id, 1 << 20, true));
                    #ifndef ALPS_HDF5_CLOSE_GREEDY
                        check_error(H5Pset_fclose_degree(prop_id, H5F_CLOSE_SEMI));
                    #endif
                    if (write_) {
                        if ((file_id_ = H5Fopen(filename_new_.c_str(), H5F_ACC_RDWR, prop_id)) < 0) {
                            property_type fcrt_id(H5Pcreate(H5P_FILE_CREATE));
                            check_error(H5Pset_link_creation_order(fcrt_id, (H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED)));
                            check_error(H5Pset_attr_creation_order(fcrt_id, (H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED)));
                            check_error(file_id_ = H5Fcreate(filename_new_.c_str(), H5F_ACC_TRUNC, fcrt_id, prop_id));
                        }
                    } else if ((file_id_ = H5Fopen(filename_new_.c_str(), H5F_ACC_RDONLY, prop_id)) < 0)
                        throw archive_not_found("file does not exists or is not a valid hdf5 archive: " + filename_new_ + ALPS_STACKTRACE);
                    else
                        check_error(file_id_);
                } else {
                    if (replace_) {
                        throw std::logic_error("'Replace' functionality is not yet implemented by hdf5::archive"
                                               +ALPS_STACKTRACE);
                        // @todo:FIXME_DEBOOST:insert proper tempname generation
                        // for (std::size_t i = 0; boost::filesystem::exists(filename_new_=(filename_ + ".tmp." + cast<std::string>(i))); ++i);
                    }
                    if (write_ && replace_ /* && boost::filesystem::exists(filename_)*/) { // @todo:FIXME_DEBOOST:verify exists() necessity
                        throw std::logic_error("'Replace' functionality is not yet implemented by hdf5::archive"
                                               +ALPS_STACKTRACE);
                        // @todo:FIXME_DEBOOST boost::filesystem::copy_file(filename_, filename_new_);
                    }
                    if (!write_) {
                        // pre-check that file is readable. HDF5 would complain anyway,
                        // but the pre-check makes the exception less scary.
                        // It is potentially time-consuming, so do not open archives in a tight loop!
                        if (!std::ifstream(filename_new_.c_str(),std::ios::in).good())
                            throw archive_not_found("file cannot be read or does not exist: " + filename_new_ + ALPS_STACKTRACE);
                        if (check_error(H5Fis_hdf5(filename_new_.c_str())) == 0)
                            throw archive_error("no valid hdf5 file: " + filename_new_ + ALPS_STACKTRACE);
                    }
                    #ifndef ALPS_HDF5_CLOSE_GREEDY
                        property_type ALPS_HDF5_FILE_ACCESS(H5Pcreate(H5P_FILE_ACCESS));
                        check_error(H5Pset_fclose_degree(ALPS_HDF5_FILE_ACCESS, H5F_CLOSE_SEMI));
                    #else
                        #define ALPS_HDF5_FILE_ACCESS H5P_DEFAULT
                    #endif
                    if (write_) {
                        if ((file_id_ = H5Fopen(filename_new_.c_str(), H5F_ACC_RDWR, ALPS_HDF5_FILE_ACCESS)) < 0) {
                            property_type fcrt_id(H5Pcreate(H5P_FILE_CREATE));
                            check_error(H5Pset_link_creation_order(fcrt_id, (H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED)));
                            check_error(H5Pset_attr_creation_order(fcrt_id, (H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED)));
                            check_error(file_id_ = H5Fcreate(filename_new_.c_str(), H5F_ACC_TRUNC, fcrt_id, ALPS_HDF5_FILE_ACCESS));
                        }
                    } else
                        check_error(file_id_ = H5Fopen(filename_new_.c_str(), H5F_ACC_RDONLY, ALPS_HDF5_FILE_ACCESS));
                    #ifdef ALPS_HDF5_CLOSE_GREEDY
                        #undef(ALPS_HDF5_FILE_ACCESS)
                    #endif
                }
            }

            void archivecontext::destruct(bool abort) {
                try {
                    H5Fflush(file_id_, H5F_SCOPE_GLOBAL);
                    #ifndef ALPS_HDF5_CLOSE_GREEDY
                        if (
                               H5Fget_obj_count(file_id_, H5F_OBJ_DATATYPE) > 0
                            || H5Fget_obj_count(file_id_, H5F_OBJ_ALL) - H5Fget_obj_count(file_id_, H5F_OBJ_FILE) > 0
                        ) {
                            std::cerr << "Not all resources closed in file '" << filename_new_ << "'" << std::endl;
                            std::abort();
                        }
                    #endif
                    if (H5Fclose(file_id_) < 0)
                        std::cerr << "Error in "
                                  << __FILE__
                                  << " on "
                                  << ALPS_STRINGIFY(__LINE__)
                                  << " in "
                                  << __FUNCTION__ // TODO: check for gcc and use __PRETTY_FUNCTION__
                                  << ":"
                                  << std::endl
                                  << error().invoke(file_id_)
                                  << std::endl;
                    if (replace_) {
                        throw std::logic_error("'Replace' functionality is not yet implemented by hdf5::archive"
                                               +ALPS_STACKTRACE);
                        //@todo:FIXME_DEBOOST if (boost::filesystem::exists(filename_))
                        //@todo:FIXME_DEBOOST     boost::filesystem::remove(filename_);
                        //@todo:FIXME_DEBOOST boost::filesystem::rename(filename_new_, filename_);
                    }
                } catch (std::exception & ex) {
                    if (abort) {
                        std::cerr << "Error destroying HDF5 context of file '" << filename_new_ << "'\n" << ex.what() << std::endl;
                        std::abort();
                    } else
                        throw ex;
                }
            }
        }
    }
}
