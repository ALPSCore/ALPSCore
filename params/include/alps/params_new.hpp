/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_PARAMS_HPP_INCLUDED_00f672a032d949a7aa0e760a6b6f0602
#define ALPS_PARAMS_HPP_INCLUDED_00f672a032d949a7aa0e760a6b6f0602

#include "alps/config.hpp"
// #include "alps/hdf5/archive.hpp"

#ifdef ALPS_HAVE_MPI
#include "alps/utilities/mpi.hpp"
#endif

#include <map>

#include "./params_new/dict_exceptions.hpp"
#include "./params_new/dict_value.hpp"


namespace alps {
    namespace params_new_ns {

        /// Python-like dictionary
        // FIXME: TODO: rewrite the whole thing using a proxy object 
        class dictionary {
          public:
            typedef dict_value value_type;

          private:
            typedef std::map<std::string, value_type> map_type;
            map_type map_;
            
          public:
            bool empty() const { return map_.empty(); }
            std::size_t size() const { return map_.size(); }

            /// Access with intent to assign
            value_type& operator[](const std::string& key) {
                map_type::iterator it=map_.lower_bound(key);
                if (it==map_.end() ||
                    map_.key_comp()(key, it->first)) {
                    // it's a new element, we have to construct it here
                    // and copy it to the map, returning the ref to the inserted element
                    it=map_.insert(it,map_type::value_type(key, value_type(key)));
                }
                // return reference to the existing or the newly-created element
                return it->second;
            }

            /// Read-only access
            const value_type& operator[](const std::string& key) const {
                map_type::const_iterator it=map_.find(key);
                if (it==map_.end()) throw exception::uninitialized_value(key, "Attempt to read uninitialized value");
                return it->second;
            }

          private:
            /// Check if the key exists and has a value; return the iterator
            map_type::const_iterator find_nonempty_(const std::string& key) const {
                map_type::const_iterator it=map_.find(key);
                if (it!=map_.end() && !(it->second).empty())
                    return it;
                else
                    return map_.end();
            }
          public:
            
            /// Check if a key exists and has a value (without creating the key)
            bool exists(const std::string& key) const {
                return find_nonempty_(key)!=map_.end();
            }
            
            /// Check if a key exists and has a value of a particular type (without creating the key)
            template <typename T>
            bool exists(const std::string& key) const {
                map_type::const_iterator it=find_nonempty_(key);
                return it!=map_.end() && (it->second).isType<T>();
            }

        };
        
    } // params_ns::
} // alps::


#endif /* ALPS_PARAMS_HPP_INCLUDED_00f672a032d949a7aa0e760a6b6f0602 */
