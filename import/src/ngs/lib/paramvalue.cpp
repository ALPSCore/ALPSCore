/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2012 by Lukas Gamper <gamperl@gmail.com>                   *
 *                                                                                 *
 * This software is part of the ALPS libraries, published under the ALPS           *
 * Library License; you can use, redistribute it and/or modify it under            *
 * the terms of the license, either version 1 or (at your option) any later        *
 * version.                                                                        *
 *                                                                                 *
 * You should have received a copy of the ALPS Library License along with          *
 * the ALPS Libraries; see the file LICENSE.txt. If not, the license is also       *
 * available from http://alps.comp-phys.org/.                                      *
 *                                                                                 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        *
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT       *
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE       *
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,     *
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER     *
 * DEALINGS IN THE SOFTWARE.                                                       *
 *                                                                                 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <alps/ngs/short_print.hpp>
#include <alps/hdf5/vector.hpp>
#include <alps/hdf5/complex.hpp>
#include <alps/hdf5/pointer.hpp>
#include <alps/ngs/detail/paramvalue.hpp>
#include <alps/ngs/detail/type_wrapper.hpp>

namespace alps {
    namespace detail {

        #if defined(ALPS_HAVE_PYTHON)
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
            
            #if defined(ALPS_HAVE_PYTHON)
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
                
                #if defined(ALPS_HAVE_PYTHON)
                    void operator()(boost::python::object const & v) const {
                        os << boost::python::call_method<std::string>(v.ptr(), "__str__");
                    }
                #endif

            private:

                std::ostream & os;
        };

        #define ALPS_NGS_PARAMVALUE_OPERATOR_T_IMPL(T)                                \
            paramvalue::operator T () const {                                        \
                paramvalue_reader< T > visitor;                                        \
                boost::apply_visitor(visitor, *this);                               \
                return visitor.get_value();                                            \
            }
        ALPS_NGS_FOREACH_PARAMETERVALUE_TYPE(ALPS_NGS_PARAMVALUE_OPERATOR_T_IMPL)
        #undef ALPS_NGS_PARAMVALUE_OPERATOR_T_IMPL

        #define ALPS_NGS_PARAMVALUE_OPERATOR_EQ_IMPL(T)                                \
            paramvalue & paramvalue::operator=( T const & arg) {                    \
                paramvalue_base::operator=(arg);                                    \
                return *this;                                                        \
            }
        ALPS_NGS_FOREACH_PARAMETERVALUE_TYPE(ALPS_NGS_PARAMVALUE_OPERATOR_EQ_IMPL)
        #undef ALPS_NGS_PARAMVALUE_OPERATOR_EQ_IMPL

        void paramvalue::save(hdf5::archive & ar) const {
            boost::apply_visitor(
                paramvalue_saver(ar), static_cast<paramvalue_base const &>(*this)
            );
        }

        void paramvalue::load(hdf5::archive & ar) {
            #define ALPS_NGS_PARAMVALUE_LOAD_HDF5(T)                                \
                {                                                                    \
                    T value;                                                        \
                    ar[""] >> value;                                        \
                    operator=(value);                                                \
                }
            #define ALPS_NGS_PARAMVALUE_LOAD_HDF5_CHECK(T, U)                        \
                else if (ar.is_datatype< T >(""))                                    \
                    ALPS_NGS_PARAMVALUE_LOAD_HDF5(U)
            if (ar.is_scalar("")) {
                if (ar.is_complex(""))
                    ALPS_NGS_PARAMVALUE_LOAD_HDF5(std::complex<double>)
                ALPS_NGS_PARAMVALUE_LOAD_HDF5_CHECK(double, double)
                ALPS_NGS_PARAMVALUE_LOAD_HDF5_CHECK(int, int)
                ALPS_NGS_PARAMVALUE_LOAD_HDF5_CHECK(bool, bool)
                ALPS_NGS_PARAMVALUE_LOAD_HDF5_CHECK(std::string, std::string)
            } else {
                if (ar.is_complex(""))
                    ALPS_NGS_PARAMVALUE_LOAD_HDF5(
                        std::vector<std::complex<double> >
                    )
                ALPS_NGS_PARAMVALUE_LOAD_HDF5_CHECK(double, std::vector<double>)
                ALPS_NGS_PARAMVALUE_LOAD_HDF5_CHECK(int, std::vector<int>)
                ALPS_NGS_PARAMVALUE_LOAD_HDF5_CHECK(
                    std::string, std::vector<std::string>
                )
            }
            #undef ALPS_NGS_PARAMVALUE_LOAD_HDF5
            #undef ALPS_NGS_PARAMVALUE_LOAD_HDF5_CHECK
        }

        std::ostream & operator<<(std::ostream & os, paramvalue const & arg) {
            paramvalue_ostream visitor(os);
            boost::apply_visitor(visitor, arg);
            return os;
        }        
    }
}
