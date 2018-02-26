/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_HDF5_ERROR_HPP
#define ALPS_HDF5_ERROR_HPP

#include <string>
#include <stdexcept>

namespace alps {
    namespace hdf5 {

        class archive_error : public std::runtime_error {
            public:
                archive_error(std::string const & what)
                    : std::runtime_error(what) 
                {}
        };

        #define DEFINE_ALPS_HDF5_EXCEPTION(name)                                    \
            class name : public archive_error {                                     \
                public:                                                             \
                    name (std::string const & what)                                 \
                        : archive_error(what)                                       \
                    {}                                                              \
            };
        DEFINE_ALPS_HDF5_EXCEPTION(archive_not_found)
        DEFINE_ALPS_HDF5_EXCEPTION(archive_closed)
        DEFINE_ALPS_HDF5_EXCEPTION(archive_opened)
        DEFINE_ALPS_HDF5_EXCEPTION(invalid_path)
        DEFINE_ALPS_HDF5_EXCEPTION(path_not_found)
        DEFINE_ALPS_HDF5_EXCEPTION(wrong_type)
        DEFINE_ALPS_HDF5_EXCEPTION(wrong_mode)
        DEFINE_ALPS_HDF5_EXCEPTION(wrong_dimensions)
        #undef DEFINE_ALPS_HDF5_EXCEPTION
    }
};

#endif
