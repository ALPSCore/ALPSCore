/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#pragma once

#include <alps/config.hpp>
#ifdef ALPS_HAVE_PYTHON_DEPRECATED
    // this must be first
    #include <alps/utilities/boost_python.hpp>
#endif

#include <alps/utilities/cast.hpp>
//#include <alps/utilities/config.hpp>

#include <boost/variant.hpp>

#ifdef ALPS_HAVE_PYTHON_DEPRECATED
	#include <alps/utilities/get_numpy_type.hpp>
	#include <alps/utilities/extract_from_pyobject.hpp>
#endif

namespace alps {
    namespace detail {

        template<typename T> struct paramvalue_reader_visitor {
            
            template <typename U> void operator()(U const & data) {
                value = cast<T>(data);
            }
            
            template <typename U> void operator()(U * const, std::vector<std::size_t>) {
                throw std::runtime_error(std::string("cannot cast from std::vector<") + typeid(U).name() + "> to " + typeid(T).name() + ALPS_STACKTRACE);
            }

            #if defined(ALPS_HAVE_PYTHON_DEPRECATED)
                void operator()(boost::python::list const &) {
                    throw std::runtime_error(std::string("cannot cast from boost::python::list ") + typeid(T).name() + ALPS_STACKTRACE);
                }

                void operator()(boost::python::dict const &) {
                    throw std::invalid_argument("python dict cannot be used in alps::params" + ALPS_STACKTRACE);
                }
            #endif

            T value;
        };

        template<typename T> struct paramvalue_reader_visitor<std::vector<T> > {
            
            template <typename U> void operator()(U const & data) {
                value.push_back(cast<T>(data));
            }

            template <typename U> void operator()(U * const ptr, std::vector<std::size_t> size) {
                if (size.size() != 1)
                    throw std::invalid_argument("only 1 D array are supported in alps::params" + ALPS_STACKTRACE);
                else
                    for (U const * it = ptr; it != ptr + size[0]; ++it)
                        (*this)(*it);
            }

            #if defined(ALPS_HAVE_PYTHON_DEPRECATED)
                void operator()(boost::python::list const & data) {
                    for(boost::python::ssize_t i = 0; i < boost::python::len(data); ++i) {
                        paramvalue_reader_visitor<T> scalar;
                        extract_from_pyobject(scalar, data[i]);
                        value.push_back(scalar.value);
                    }
                }

                void operator()(boost::python::dict const &) {
                    throw std::invalid_argument("python dict cannot be used in alps::params" + ALPS_STACKTRACE);
                }
            #endif

            std::vector<T> value;
        };

        template<> struct paramvalue_reader_visitor<std::string> {
            
            template <typename U> void operator()(U const & data) {
                value = cast<std::string>(data);
            }
            
            template <typename U> void operator()(U * const ptr, std::vector<std::size_t> size) {
                if (size.size() != 1)
                    throw std::invalid_argument("only 1 D array are supported in alps::params" + ALPS_STACKTRACE);
                else
                    for (U const * it = ptr; it != ptr + size[0]; ++it)
                        value += (it == ptr ? "," : "") + cast<std::string>(*it);
            }

            #if defined(ALPS_HAVE_PYTHON_DEPRECATED)
                void operator()(boost::python::list const & data) {
                    for(boost::python::ssize_t i = 0; i < boost::python::len(data); ++i)
                        value += (value.size() ? "," : "") + boost::python::call_method<std::string>(boost::python::object(data[i]).ptr(), "__str__");
                }

                void operator()(boost::python::dict const &) {
                    throw std::invalid_argument("python dict cannot be used in alps::params" + ALPS_STACKTRACE);
                }
            #endif

            std::string value;
        };

        template<typename T> struct paramvalue_reader 
            : public boost::static_visitor<> 
        {
            public:

                template <typename U> void operator()(U const & v) const {
                    visitor(v);
                }
                
                template <typename U> void operator()(std::vector<U> const & v) const {
                    visitor(&v.front(), std::vector<std::size_t>(1, v.size()));
                }

                void operator()(T const & v) const {
                    visitor.value = v; 
                }

                #if defined(ALPS_HAVE_PYTHON_DEPRECATED)
                    void operator()(boost::python::object const & v) const {
                        extract_from_pyobject(visitor, v);
                    }
                #endif

                T const & get_value() {
                    return visitor.value;
                }

            private:

                mutable paramvalue_reader_visitor<T> visitor;
        };

        #if defined(ALPS_HAVE_PYTHON_DEPRECATED)
            template<> struct paramvalue_reader<boost::python::object>
                : public boost::static_visitor<> 
            {
                public:

                    template <typename U> void operator()(U const & v) const {
                        value = boost::python::object(v);
                    }

                    template <typename U> void operator()(std::vector<U> const & v) const {
                        npy_intp npsize = v.size();
                        value = boost::python::object(boost::python::handle<>(PyArray_SimpleNew(1, &npsize, detail::get_numpy_type(U()))));
                        memcpy(PyArray_DATA(value.ptr()), &v.front(), PyArray_ITEMSIZE(value.ptr()) * PyArray_SIZE(value.ptr()));
                    }

                    void operator()(std::vector<std::string> const & v) const {
                        value = boost::python::list(v);
                    }

                    void operator()(boost::python::object const & v) const {
                        value = v; 
                    }

                    boost::python::object const & get_value() {
                        return value;
                    }

                private:

                    mutable boost::python::object value;
            };
        #endif
    }
}

