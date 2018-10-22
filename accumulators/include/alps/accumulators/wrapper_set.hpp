/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#pragma once

#include <alps/config.hpp>
#include <alps/hdf5/archive.hpp>

#include <memory>
#include <mutex>

namespace alps {
    namespace accumulators {

        class accumulator_wrapper;
        class result_wrapper;

        namespace detail {
            template<typename T> struct serializable_type;
        }

        namespace impl {

            template <typename T> class wrapper_set {

                public:
                    typedef T value_type;

                    typedef typename std::map<std::string, std::shared_ptr<T> >::iterator iterator;
                    typedef typename std::map<std::string, std::shared_ptr<T> >::const_iterator const_iterator;

                    // TODO: make trait ... to disable for result_wrapper
                    template <typename U> wrapper_set(wrapper_set<U> const & arg) {
                        for (typename wrapper_set<U>::const_iterator it = arg.begin(); it != arg.end(); ++it)
                            insert(it->first, it->second->result());
                    }

                    wrapper_set();
                    wrapper_set(wrapper_set const &) {} // TODO: how do we handle that?

                    T & operator[](std::string const & name);
                    T const & operator[](std::string const & name) const;

                    bool has(std::string const & name) const;

                    void insert(std::string const & name, std::shared_ptr<T> ptr);

                    std::size_t size() const {
                        return m_storage.size();
                    }

                    void save(hdf5::archive & ar) const;
                    void load(hdf5::archive & ar);

                    /// Register a serializable type, without locking
                    template<typename A> static void register_serializable_type_nolock();
                    /// Register a user-defined serializable type
                    template<typename A> static void register_serializable_type();

                    void print(std::ostream & os) const;

                    iterator begin() { return m_storage.begin(); }
                    iterator end() { return m_storage.end(); }

                    const_iterator begin() const { return m_storage.begin(); }
                    const_iterator end() const { return m_storage.end(); }

                    void clear() { m_storage.clear(); }

                    //
                    // These methods are valid only for T = accumulator_wrapper
                    //

                    /// Merge another accumulator/result set into this one. @param rhs the set to merge.
                    template<typename U = T>
                    typename std::enable_if<std::is_same<U, accumulator_wrapper>::value>::type
                    merge(wrapper_set const &rhs) {
                        iterator it1 = this->begin();
                        const_iterator it2 = rhs.begin();
                        for(; it1 != end(); ++it1, ++it2) {
                            if (it1->first != it2 ->first) throw std::logic_error("Can't merge" + it1->first + " and " + it2->first);
                            it1->second->merge(*(it2->second));
                        }
                    }

                    template<typename U = T>
                    typename std::enable_if<std::is_same<U, accumulator_wrapper>::value>::type
                    reset() {
                        for(iterator it = begin(); it != end(); ++it)
                            it->second->reset();
                    }

                private:
                    std::map<std::string, std::shared_ptr<T> > m_storage;
                    static std::vector<std::shared_ptr<detail::serializable_type<T> > > m_types;
                    static std::mutex m_types_mutex;
            };
            template<typename T> std::vector<std::shared_ptr<detail::serializable_type<T> > > wrapper_set<T>::m_types;
            template<typename T> std::mutex wrapper_set<T>::m_types_mutex;

            template<typename T> inline std::ostream & operator<<(std::ostream & os, const wrapper_set<T> & arg) {
                arg.print(os);
                return os;
            }

            // Will be instantiated in wrapper_set.cpp and wrapper_set_hdf5.cpp
            extern template class wrapper_set<accumulator_wrapper>;
            extern template class wrapper_set<result_wrapper>;
        }
    }
}
