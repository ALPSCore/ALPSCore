/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file dictionary.cpp
    Contains implementation of some alps::params_ns::dictionary members */

#include <alps/dictionary.hpp>

#include <alps/hdf5/map.hpp>

#ifdef ALPS_HAVE_MPI
#include <alps/utilities/mpi_map.hpp>
#endif

namespace alps {
    namespace params_ns {

        dictionary::map_type::const_iterator dictionary::find_nonempty_(const std::string& key) const {
            map_type::const_iterator it=map_.find(key);
            if (it!=map_.end() && !(it->second).empty())
                return it;
            else
                return map_.end();
        }

        
        /// Access with intent to assign
        dictionary::value_type& dictionary::operator[](const std::string& key) {
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
        const dictionary::value_type& dictionary::operator[](const std::string& key) const {
            map_type::const_iterator it=map_.find(key);
            if (it==map_.end()) throw exception::uninitialized_value(key, "Attempt to read uninitialized value");
            return it->second;
        }

        namespace {

            template <typename M>
            struct compare {
                typedef typename M::value_type pair_type;
                bool operator()(const pair_type& lhs, const pair_type& rhs) const
                {
                    return (lhs.first==rhs.first) && lhs.second.equals(rhs.second);
                }
            };

        }
        
        bool dictionary::equals(const dictionary &rhs) const 
        {
            if (this->size()!=rhs.size()) return false;
            return std::equal(map_.begin(), map_.end(), rhs.map_.begin(), compare<map_type>());
        }


        void dictionary::save(alps::hdf5::archive& ar) const
        {
            ar[""] << map_;
        }

        void dictionary::load(alps::hdf5::archive& ar)
        {
            map_type new_map;
            ar[""] >> new_map;
            
            using std::swap;
            swap(map_,new_map);
        }

        std::ostream& operator<<(std::ostream& s, const dictionary& d)
        {
            for (dictionary::const_iterator it=d.begin(); it!=d.end(); ++it) {
                s << it->first << " = " << it->second << "\n";
            }
            return s;
        }

#ifdef ALPS_HAVE_MPI
        // Defined here to avoid including <mpi_map.hpp> inside user header
        void dictionary::broadcast(const alps::mpi::communicator& comm, int root) { 
            using alps::mpi::broadcast;
            broadcast(comm, map_, root);
        }
#endif

    } // params_ns::
} // alps::
