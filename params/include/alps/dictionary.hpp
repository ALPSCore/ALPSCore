/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
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
            typedef map_type::const_iterator const_iterator;

            /// Const-iterator to the beginning of the contained map
            const_iterator begin() const { return map_.begin(); }

            /// Const-iterator to the end of the contained map
            const_iterator end() const { return map_.end(); }

            /// Virtual destructor to make dictionary inheritable
            virtual ~dictionary() {}

            /// True if the cdictionary does not contain elements (even empty ones)
            bool empty() const { return map_.empty(); }

            /// Size of the dictionary (including empty elements)
            std::size_t size() const { return map_.size(); }

            /// Erase an element if it exists
            void erase(const std::string& key) { map_.erase(key); }

            /// Access with intent to assign
            value_type& operator[](const std::string& key);

            /// Read-only access
            const value_type& operator[](const std::string& key) const;

            /// Obtain read-only iterator to a name
            const_iterator find(const std::string& key) const {
                return map_.find(key);
            }

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

            /// Swap the dictionaries
            friend void swap(dictionary& d1, dictionary& d2) { using std::swap; swap(d1.map_, d2.map_); }

            /// Compare two dictionaries (true if all entries are of the same type and value)
            bool equals(const dictionary& rhs) const;

            /// Save the dictionary to an archive
            void save(alps::hdf5::archive& ar) const;

            /// Load the dictionary from an archive
            void load(alps::hdf5::archive& ar);

            friend std::ostream& operator<<(std::ostream&, const dictionary&);

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

        /// Const-access visitor to a value by an iterator
        /** @param visitor A functor that should be callable as `R result=visitor(bound_value_const_ref)`
            @param it Iterator to the dictionary
            @tparam F The functor type; must define typename `F::result_type`.
        */
        template <typename F>
        inline typename F::result_type apply_visitor(F& visitor, dictionary::const_iterator it) {
            return boost::apply_visitor(visitor, it->second);
        }

        /// Const-access visitor to a value by an iterator
        /** @param visitor A functor that should be callable as `R result=visitor(bound_value_const_ref)`
            @param it Iterator to the dictionary
            @tparam F The functor type; must define typename `F::result_type`.
        */
        template <typename F>
        inline typename F::result_type apply_visitor(const F& visitor, dictionary::const_iterator it) {
            return boost::apply_visitor(visitor, it->second);
        }

    } // params_ns::
} // alps::


#endif /* ALPS_PARAMS_DICTIONARY_HPP_e15039548f43464996cad06f9c8a3220 */
