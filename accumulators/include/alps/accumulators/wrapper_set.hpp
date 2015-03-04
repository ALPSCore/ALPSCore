/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_ACCUMULATOR_WRAPPER_SET_HPP
#define ALPS_ACCUMULATOR_WRAPPER_SET_HPP

#include <alps/config.hpp>
#include <alps/hdf5/vector.hpp>
#include <alps/hdf5/archive.hpp>

#include <boost/shared_ptr.hpp>

namespace alps {
    namespace accumulators {

         namespace detail {
            
            template<typename T> struct serializable_type {
                virtual ~serializable_type() {}
                virtual std::size_t rank() const = 0;
                virtual bool can_load(hdf5::archive & ar) const = 0;
                virtual T * create(hdf5::archive & ar) const = 0;
            };

            template<typename T, typename A> struct serializable_type_impl : public serializable_type<T> {
                std::size_t rank() const {
                    return A::rank();
                }
                bool can_load(hdf5::archive & ar) const {
                    return A::can_load(ar);
                }
                T * create(hdf5::archive & ar) const {
                    return new T(A());
                }
            };

            void register_predefined_serializable_type();

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
                    // template <typename U> wrapper_set(wrapper_set<U> const & arg, typename boost::disable_if<boost::is_same<result_wrapper, U>, void *>::type = NULL) {
                    //     for (typename wrapper_set<U>::const_iterator it = arg.begin(); it != arg.end(); ++it)
                    //         insert(it->first, it->second->result());
                    // }



                    wrapper_set() {
                        if (m_types.empty())
                            detail::register_predefined_serializable_type();
                    }
                    wrapper_set(wrapper_set const &) {} // TODO: how do we handle that?

                    T & operator[](std::string const & name) {
                        if (!has(name))
                            m_storage.insert(make_pair(name, boost::shared_ptr<T>(new T())));
                        return *(m_storage.find(name)->second);
                    }

                    T const & operator[](std::string const & name) const {
                        if (!has(name))
                            throw std::out_of_range("No observable found with the name: " + name + ALPS_STACKTRACE);
                        return *(m_storage.find(name)->second);
                    }

                    bool has(std::string const & name) const{
                        return m_storage.find(name) != m_storage.end();
                    }
                    
                    void insert(std::string const & name, boost::shared_ptr<T> ptr){
                        if (has(name))
                            throw std::out_of_range("There exists already an accumulator with the name: " + name + ALPS_STACKTRACE);
                        m_storage.insert(make_pair(name, ptr));
                    }

                    std::size_t size() const {
                        return m_storage.size();
                    }

                    void save(hdf5::archive & ar) const {
                        for(const_iterator it = begin(); it != end(); ++it)
                            ar[it->first] = *(it->second);
                    }

                    void load(hdf5::archive & ar) {
                        std::vector<std::string> list = ar.list_children("");
                        for (std::vector<std::string>::const_iterator it = list.begin(); it != list.end(); ++it) {
                            ar.set_context(*it);
                            for (typename std::vector<boost::shared_ptr<detail::serializable_type<T> > >::const_iterator jt = m_types.begin()
                                ; jt != m_types.end()
                                ; ++jt
                            )
                                if ((*jt)->can_load(ar)) {
                                    operator[](*it) = boost::shared_ptr<T>((*jt)->create(ar));
                                    break;
                                }
                            if (!has(*it))
                                throw std::logic_error("The Accumulator/Result " + *it + " cannot be unserilized" + ALPS_STACKTRACE);
                            operator[](*it).load(ar);
                            ar.set_context("..");
                        }
                    }

                    template<typename A> static void register_serializable_type(bool known = false) {
                        if (!known && m_types.empty())
                            detail::register_predefined_serializable_type();
                        m_types.push_back(boost::shared_ptr<detail::serializable_type<T> >(new detail::serializable_type_impl<T, A>));
                        for (std::size_t i = m_types.size(); i > 1 && m_types[i - 1]->rank() > m_types[i - 2]->rank(); --i)
                            m_types[i - 1].swap(m_types[i - 2]);
                    }

                    /// Merge another accumulator/result set into this one. @param rhs the set to merge.
                    void merge(wrapper_set const &rhs) {
                        iterator it1 = this->begin();
                        const_iterator it2 = rhs.begin();
                        for(; it1 != end(); ++it1, ++it2) { 
                            if (it1->first != it2 ->first) throw std::logic_error("Can't merge" + it1->first + " and " + it2->first);
                            it1->second->merge(*(it2->second));
                        }
                    }

                    void print(std::ostream & os) const {
                        for(const_iterator it = begin(); it != end(); ++it)
                            os << it->first << ": " << *(it->second) << std::endl;
                    }

                    void reset() {
                        for(iterator it = begin(); it != end(); ++it)
                            it->second->reset();
                    }
                    
                    iterator begin() { return m_storage.begin(); }
                    iterator end() { return m_storage.end(); }

                    const_iterator begin() const { return m_storage.begin(); }
                    const_iterator end() const { return m_storage.end(); }
                    
                    void clear() { m_storage.clear(); }

                private:
                    std::map<std::string, boost::shared_ptr<T> > m_storage;
                    static std::vector<boost::shared_ptr<detail::serializable_type<T> > > m_types;
            };
            template<typename T> std::vector<boost::shared_ptr<detail::serializable_type<T> > > wrapper_set<T>::m_types;

            template<typename T> inline std::ostream & operator<<(std::ostream & os, const wrapper_set<T> & arg) {
                arg.print(os);
                return os;
            }
        }

    }
}

 #endif
