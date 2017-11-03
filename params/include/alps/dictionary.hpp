/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_PARAMS_DICTIONARY_HPP_e15039548f43464996cad06f9c8a3220
#define ALPS_PARAMS_DICTIONARY_HPP_e15039548f43464996cad06f9c8a3220

#include <alps/config.hpp>
#include <map>
#include "./params/dict_value.hpp"

namespace alps {
    namespace params_ns {

        /// Python-like dictionary
        class dictionary {
            public:
            typedef dict_value value_type;

            private:
            typedef std::map<std::string, value_type> map_type;
            map_type map_;
            
            public:
            /// Virtual destructor to make dictionary inheritable
            virtual ~dictionary() {}
            
            bool empty() const { return map_.empty(); }
            std::size_t size() const { return map_.size(); }

            /// Access with intent to assign
            value_type& operator[](const std::string& key);

            /// Read-only access
            const value_type& operator[](const std::string& key) const;

            private:
            /// Check if the key exists and has a value; return the iterator
            map_type::const_iterator find_nonempty_(const std::string& key) const;
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

            /// Compare two dictionaries (true if all entries are of the same type and value)
            bool equals(const dictionary& rhs) const;

            /// Save the dictionary to an archive
            void save(alps::hdf5::archive& ar) const;

            /// Load the dictionary from an archive
            void load(alps::hdf5::archive& ar);

#ifdef ALPS_HAVE_MPI
            /// Broadcast the dictionary
            void broadcast(const alps::mpi::communicator& comm, int root);
#endif
        };

        inline bool operator==(const dictionary& lhs, const dictionary& rhs) {
            return lhs.equals(rhs);
        }

        inline bool operator!=(const dictionary& lhs, const dictionary& rhs) {
            return !(lhs==rhs);
        }

    } // params_ns::
} // alps::


#endif /* ALPS_PARAMS_DICTIONARY_HPP_e15039548f43464996cad06f9c8a3220 */
