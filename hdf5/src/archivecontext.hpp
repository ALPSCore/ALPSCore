/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#pragma once

#include <string>

#include <boost/noncopyable.hpp>

#include <hdf5.h>

namespace alps {
    namespace hdf5 {
        namespace detail {

            struct archivecontext : boost::noncopyable {

                    archivecontext(std::string const & filename, bool write, bool replace, bool compress, bool memory);
                    ~archivecontext();

                    void grant(bool write, bool replace);

                    bool compress_;
                    bool write_;
                    bool replace_;
                    bool memory_;
                    std::string filename_;
                    std::string filename_new_;
                    hid_t file_id_;

                private:

                    void construct();
                    void destruct(bool abort);
            };
        }
    }
}
