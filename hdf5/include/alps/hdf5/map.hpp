/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_HDF5_STD_MAP
#define ALPS_HDF5_STD_MAP

#include <alps/hdf5/archive.hpp>
#include <alps/utilities/cast.hpp>

#include <map>

namespace alps {
    namespace hdf5 {

        template <typename K, typename T, typename C, typename A> void save(
              archive & ar
            , std::string const & path
            , std::map<K, T, C, A> const & value
            , std::vector<std::size_t> size = std::vector<std::size_t>()
            , std::vector<std::size_t> chunk = std::vector<std::size_t>()
            , std::vector<std::size_t> offset = std::vector<std::size_t>()
        ) {
            if (ar.is_group(path))
                ar.delete_group(path);
            ar.create_group(path);
            for(typename std::map<K, T, C, A>::const_iterator it = value.begin(); it != value.end(); ++it)
                save(ar, ar.complete_path(path) + "/" + ar.encode_segment(cast<std::string>(it->first)), it->second);
        }

        template <typename K, typename T, typename C, typename A> void load(
              archive & ar
            , std::string const & path
            , std::map<K, T, C, A> & value
            , std::vector<std::size_t> chunk = std::vector<std::size_t>()
            , std::vector<std::size_t> offset = std::vector<std::size_t>()
        ) {
            std::vector<std::string> children = ar.list_children(path);
            for (typename std::vector<std::string>::const_iterator it = children.begin(); it != children.end(); ++it)
                load(ar, ar.complete_path(path) + "/" +  *it, value[ar.decode_segment(cast<K>(*it))]);
        }
    }
}

#endif
