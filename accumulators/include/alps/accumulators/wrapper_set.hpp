/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#pragma once

#include <alps/config.hpp>
#include <alps/hdf5/archive.hpp>

#include <boost/shared_ptr.hpp>
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

                    typedef typename std::map<std::string, boost::shared_ptr<T> >::iterator iterator;
                    typedef typename std::map<std::string, boost::shared_ptr<T> >::const_iterator const_iterator;

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

                    void insert(std::string const & name, boost::shared_ptr<T> ptr);

                    std::size_t size() const {
                        return m_storage.size();
                    }

                    void save(hdf5::archive & ar) const;
                    void load(hdf5::archive & ar);

                    /// Register a serializable type, without locking
                    template<typename A> static void register_serializable_type_nolock();
                    /// Register a user-defined serializable type
                    template<typename A> static void register_serializable_type();

                    /// Merge another accumulator/result set into this one. @param rhs the set to merge.
                    void merge(wrapper_set const &rhs);

                    void print(std::ostream & os) const;

                    void reset();

                    iterator begin() { return m_storage.begin(); }
                    iterator end() { return m_storage.end(); }

                    const_iterator begin() const { return m_storage.begin(); }
                    const_iterator end() const { return m_storage.end(); }

                    void clear() { m_storage.clear(); }

                private:
                    std::map<std::string, boost::shared_ptr<T> > m_storage;
                    static std::vector<boost::shared_ptr<detail::serializable_type<T> > > m_types;
                    static std::mutex m_types_mutex;
            };
            template<typename T> std::vector<boost::shared_ptr<detail::serializable_type<T> > > wrapper_set<T>::m_types;
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
