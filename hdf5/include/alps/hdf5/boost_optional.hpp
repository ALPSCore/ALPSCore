/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_HDF5_BOOST_OPTIONAL_4ea49b8c8c814a52b0b3c6ac5bcc7839
#define ALPS_HDF5_BOOST_OPTIONAL_4ea49b8c8c814a52b0b3c6ac5bcc7839

#include <boost/optional.hpp>

namespace alps {
    namespace hdf5 {

        template <typename T>
        void save(alps::hdf5::archive& ar, const std::string& path,
                  const boost::optional<T>& value,
                  std::vector<std::size_t> /*size*/=std::vector<std::size_t>(),
                  std::vector<std::size_t> /*chunk*/=std::vector<std::size_t>(),
                  std::vector<std::size_t> /*offset*/=std::vector<std::size_t>())
        {
            if (ar.is_group(path)) ar.delete_group(path);
            if (!value) {
                ar.write(path,T()); // FIXME?: T must have a default constructor then.
                ar.write(path+"/@alps_hdf5_optional_empty", true);
            } else {
                ar.write(path,*value);
                ar.write(path+"/@alps_hdf5_optional_empty", false);
            }
        }
          
        template <typename T>
        void load(alps::hdf5::archive& ar, const std::string& path,
                  boost::optional<T>& value,
                  std::vector<std::size_t> /*size*/=std::vector<std::size_t>(),
                  std::vector<std::size_t> /*chunk*/=std::vector<std::size_t>(),
                  std::vector<std::size_t> /*offset*/=std::vector<std::size_t>())
        {
            bool is_empty=false;
            ar.read(path+"/@alps_hdf5_optional_empty", is_empty);
            if (is_empty) {
                value = boost::none;
            } else {
                ar.read(path, *value);
            }
        }

    } // hdf5::
} // alps::

#endif /* ALPS_HDF5_BOOST_OPTIONAL_4ea49b8c8c814a52b0b3c6ac5bcc7839 */
