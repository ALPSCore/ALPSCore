/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/utilities/short_print.hpp>
#include <alps/hdf5/vector.hpp>
#include <alps/hdf5/complex.hpp>
#include <alps/hdf5/pointer.hpp>
#include <alps/params/paramvalue.hpp>
#include <alps/utilities/type_wrapper.hpp>

namespace alps {
    namespace detail {

        #if defined(ALPS_HAVE_PYTHON_DEPRECATED)
            struct paramvalue_save_python_visitor {
            
                paramvalue_save_python_visitor(hdf5::archive & a)
                    : ar(a) 
                {}

                template <typename U> void operator()(U const & data) {
                    ar[""] << data;
                }
                
                template <typename U> void operator()(U * const ptr, std::vector<std::size_t> const & size) {
                    ar << make_pvp("", ptr, size);
                }

                void operator()(boost::python::list const & raw) {
                    std::vector<std::string> data;
                    for(boost::python::ssize_t i = 0; i < boost::python::len(raw); ++i) {
                        // TODO: also consider other types than strings ...
                        paramvalue_reader_visitor<std::string> scalar;
                        extract_from_pyobject(scalar, raw[i]);
                        data.push_back(scalar.value);
                    }
                    ar[""] << data;
                }

                void operator()(boost::python::dict const &) {
                    throw std::invalid_argument("python dict cannot be used in alps::params" + ALPS_STACKTRACE);
                }

                hdf5::archive & ar;
            };
        #endif

        struct paramvalue_saver: public boost::static_visitor<> {

            paramvalue_saver(hdf5::archive & a)
                : ar(a) 
            {}

            template<typename T> void operator()(T const & v) const {
                ar[""] << v;
            }
            
            #if defined(ALPS_HAVE_PYTHON_DEPRECATED)
                void operator()(boost::python::object const & v) const {
                    paramvalue_save_python_visitor visitor(ar);
                    extract_from_pyobject(visitor, v);
                }
            #endif

            hdf5::archive & ar;
        };

        struct paramvalue_ostream : public boost::static_visitor<> {
            public:

                paramvalue_ostream(std::ostream & arg) : os(arg) {}

                template <typename U> void operator()(U const & v) const {
                    os << short_print(v);
                }
                
                #if defined(ALPS_HAVE_PYTHON_DEPRECATED)
                    void operator()(boost::python::object const & v) const {
                        os << boost::python::call_method<std::string>(v.ptr(), "__str__");
                    }
                #endif

            private:

                std::ostream & os;
        };

        #define ALPS_PARAMVALUE_OPERATOR_T_IMPL(T)                                   \
            paramvalue::operator T () const {                                        \
                paramvalue_reader< T > visitor;                                      \
                boost::apply_visitor(visitor, *this);                                \
                return visitor.get_value();                                          \
            }
        ALPS_FOREACH_PARAMETERVALUE_TYPE(ALPS_PARAMVALUE_OPERATOR_T_IMPL)
        #undef ALPS_PARAMVALUE_OPERATOR_T_IMPL

        #define ALPS_PARAMVALUE_OPERATOR_EQ_IMPL(T)                                  \
            paramvalue & paramvalue::operator=( T const & arg) {                     \
                paramvalue_base::operator=(arg);                                     \
                return *this;                                                        \
            }
        ALPS_FOREACH_PARAMETERVALUE_TYPE(ALPS_PARAMVALUE_OPERATOR_EQ_IMPL)
        #undef ALPS_PARAMVALUE_OPERATOR_EQ_IMPL

        void paramvalue::save(hdf5::archive & ar) const {
            boost::apply_visitor(
                paramvalue_saver(ar), static_cast<paramvalue_base const &>(*this)
            );
        }

        void paramvalue::load(hdf5::archive & ar) {
            #define ALPS_PARAMVALUE_LOAD_HDF5(T)                                     \
                {                                                                    \
                    T value;                                                         \
                    ar[""] >> value;                                                 \
                    operator=(value);                                                \
                }
            #define ALPS_PARAMVALUE_LOAD_HDF5_CHECK(T, U)                            \
                else if (ar.is_datatype< T >(""))                                    \
                    ALPS_PARAMVALUE_LOAD_HDF5(U)
            if (ar.is_scalar("")) {
                if (ar.is_complex(""))
                    ALPS_PARAMVALUE_LOAD_HDF5(std::complex<double>)
                ALPS_PARAMVALUE_LOAD_HDF5_CHECK(double, double)
                ALPS_PARAMVALUE_LOAD_HDF5_CHECK(int, int)
                ALPS_PARAMVALUE_LOAD_HDF5_CHECK(bool, bool)
                ALPS_PARAMVALUE_LOAD_HDF5_CHECK(std::string, std::string)
            } else {
                if (ar.is_complex(""))
                    ALPS_PARAMVALUE_LOAD_HDF5(
                        std::vector<std::complex<double> >
                    )
                ALPS_PARAMVALUE_LOAD_HDF5_CHECK(double, std::vector<double>)
                ALPS_PARAMVALUE_LOAD_HDF5_CHECK(int, std::vector<int>)
                ALPS_PARAMVALUE_LOAD_HDF5_CHECK(
                    std::string, std::vector<std::string>
                )
            }
            #undef ALPS_PARAMVALUE_LOAD_HDF5
            #undef ALPS_PARAMVALUE_LOAD_HDF5_CHECK
        }

        std::ostream & operator<<(std::ostream & os, paramvalue const & arg) {
            paramvalue_ostream visitor(os);
            boost::apply_visitor(visitor, arg);
            return os;
        }        
    }
}
