/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2011 by Lukas Gamper <gamperl@gmail.com>                   *
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

#ifndef ALPS_NGS_DETAIL_PARAMVALUE_HPP
#define ALPS_NGS_DETAIL_PARAMVALUE_HPP

#include <alps/hdf5/archive.hpp>
#include <alps/ngs/config.hpp>
#include <alps/ngs/detail/remove_cvr.hpp>
#include <alps/ngs/detail/paramvalue_reader.hpp>

#if defined(ALPS_HAVE_PYTHON)
    #include <alps/ngs/boost_python.hpp>
#endif

#include <boost/variant.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/pop_back.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp> 
#include <boost/serialization/complex.hpp> 
#include <boost/serialization/split_member.hpp>

#include <string>
#include <complex>
#include <ostream>
#include <stdexcept>

#define ALPS_NGS_FOREACH_PARAMETERVALUE_TYPE_NO_PYTHON(CALLBACK)                    \
    CALLBACK(double)                                                                \
    CALLBACK(int)                                                                   \
    CALLBACK(bool)                                                                  \
    CALLBACK(std::string)                                                           \
    CALLBACK(std::complex<double>)                                                  \
    CALLBACK(std::vector<double>)                                                   \
    CALLBACK(std::vector<int>)                                                      \
    CALLBACK(std::vector<std::string>)                                              \
    CALLBACK(std::vector<std::complex<double> >)

#if defined(ALPS_HAVE_PYTHON)
    #define ALPS_NGS_FOREACH_PARAMETERVALUE_TYPE(CALLBACK)                          \
        ALPS_NGS_FOREACH_PARAMETERVALUE_TYPE_NO_PYTHON(CALLBACK)                    \
        CALLBACK(boost::python::object)
#else
    #define ALPS_NGS_FOREACH_PARAMETERVALUE_TYPE(CALLBACK)                          \
        ALPS_NGS_FOREACH_PARAMETERVALUE_TYPE_NO_PYTHON(CALLBACK)
#endif

namespace alps {

    namespace detail {

		template <typename T> struct paramvalue_index {};
		template <> struct paramvalue_index<double> {
			enum { value = 0 };
		};
		template <> struct paramvalue_index<int> {
			enum { value = 1 };
		};
		template <> struct paramvalue_index<bool> {
			enum { value = 2 };
		};
		template <> struct paramvalue_index<std::string> {
			enum { value = 3 };
		};
		template <> struct paramvalue_index<std::complex<double> > {
			enum { value = 4 };
		};
		template <> struct paramvalue_index<std::vector<double> > {
			enum { value = 5 };
		};
		template <> struct paramvalue_index<std::vector<int> > {
			enum { value = 6 };
		};
		template <> struct paramvalue_index<std::vector<std::string> > {
			enum { value = 7 };
		};
		template <> struct paramvalue_index<std::vector<std::complex<double> > > {
			enum { value = 8 };
		};
		#if defined(ALPS_HAVE_PYTHON)
			template <> struct paramvalue_index<boost::python::object> {
				enum { value = 9 };
			};
		#endif

        class paramvalue;

    }

    template<typename T> T extract (detail::paramvalue const & arg);

    namespace detail {

        template<class Archive> struct paramvalue_serializer 
            : public boost::static_visitor<> 
        {
            public:

                paramvalue_serializer(Archive & a)
                    : ar(a)
                {}

                template <typename U> void operator()(U & v) const {
                    std::size_t type = paramvalue_index<typename remove_cvr<U>::type>::value;
                    ar
                        << type
                        << v
                    ;
                }

            private:

                Archive & ar;
        };

        #define ALPS_NGS_PARAMVALUE_VARIANT_TYPE(T)    T,
        typedef boost::mpl::pop_back<boost::mpl::vector<
            ALPS_NGS_FOREACH_PARAMETERVALUE_TYPE(
                ALPS_NGS_PARAMVALUE_VARIANT_TYPE
            ) void
        >::type >::type paramvalue_types;
        #undef ALPS_NGS_PARAMVALUE_VARIANT_TYPE
        typedef boost::make_variant_over<paramvalue_types>::type paramvalue_base;

        class paramvalue : public paramvalue_base {

            public:

                paramvalue() {}

                paramvalue(paramvalue const & v)
                    : paramvalue_base(static_cast<paramvalue_base const &>(v))
                {}

                paramvalue const& operator=(paramvalue const& x)
                {
                  static_cast<paramvalue_base&>(*this) = static_cast<paramvalue_base const&>(x);
                  return *this;
                }
                template<typename T> T cast() const {
                    paramvalue_reader< T > visitor;
                    boost::apply_visitor(visitor, *this);
                    return visitor.get_value();
                }

                #define ALPS_NGS_PARAMVALUE_MEMBER_DECL(T)                          \
                    paramvalue( T const & v) : paramvalue_base(v) {}                \
                    operator T () const;                                            \
                    paramvalue & operator=( T const &);
                ALPS_NGS_FOREACH_PARAMETERVALUE_TYPE(ALPS_NGS_PARAMVALUE_MEMBER_DECL)
                #undef ALPS_NGS_PARAMVALUE_MEMBER_DECL

                
                #define ALPS_NGS_PARAMVALUE_MEMBER_DECL_CONVERT(T)                  \
                    paramvalue(T const & value)                                     \
                        : paramvalue_base(static_cast<int>(value))                  \
                    {}                                                              \
                    paramvalue & operator=(T const & value) {                       \
                        return *this = static_cast<int>(value);                     \
                    }
                ALPS_NGS_PARAMVALUE_MEMBER_DECL_CONVERT(char)
                ALPS_NGS_PARAMVALUE_MEMBER_DECL_CONVERT(unsigned char)
                ALPS_NGS_PARAMVALUE_MEMBER_DECL_CONVERT(short)
                ALPS_NGS_PARAMVALUE_MEMBER_DECL_CONVERT(unsigned short)
                ALPS_NGS_PARAMVALUE_MEMBER_DECL_CONVERT(unsigned)
                ALPS_NGS_PARAMVALUE_MEMBER_DECL_CONVERT(long)
                ALPS_NGS_PARAMVALUE_MEMBER_DECL_CONVERT(unsigned long)
                ALPS_NGS_PARAMVALUE_MEMBER_DECL_CONVERT(long long)
                ALPS_NGS_PARAMVALUE_MEMBER_DECL_CONVERT(unsigned long long)
                #undef ALPS_NGS_PARAMVALUE_MEMBER_DECL_CONVERT

                paramvalue(long double const & value)
                    : paramvalue_base(static_cast<double>(value))
                {}
                paramvalue & operator=(long double const & value) {
                    return *this = static_cast<double>(value);
                }

                paramvalue(char const * v) : paramvalue_base(std::string(v)) {}
                paramvalue & operator=(char const * value) { 
                    return *this = std::string(value); 
                }

                void save(hdf5::archive &) const;
                void load(hdf5::archive &);
                
            private:
            
                friend class boost::serialization::access;

                template<class Archive> void save(
                    Archive & ar, const unsigned int version
                ) const {
                    paramvalue_serializer<Archive> visitor(ar);
                    boost::apply_visitor(visitor, *this);
                }

                template<class Archive> void load(
                    Archive & ar, const unsigned int
                ) {
                    std::size_t type;
                    ar >> type;
                    if (false);
                    #define ALPS_NGS_PARAMVALUE_LOAD(T)                                 \
                        else if (type == paramvalue_index< T >::value) {                \
                            T value;                                                    \
                            ar >> value;                                                \
                            operator=(value);                                           \
                        }
                    // TODO: fix python serialization!
                    // ALPS_NGS_FOREACH_PARAMETERVALUE_TYPE(ALPS_NGS_PARAMVALUE_LOAD)
                    ALPS_NGS_FOREACH_PARAMETERVALUE_TYPE_NO_PYTHON(ALPS_NGS_PARAMVALUE_LOAD)
                    #undef ALPS_NGS_PARAMVALUE_LOAD
                    else
                        throw std::runtime_error("unknown type" + ALPS_STACKTRACE);
                }

                BOOST_SERIALIZATION_SPLIT_MEMBER()
        };

        ALPS_DECL std::ostream & operator<<(std::ostream & os, paramvalue const & arg);

        template<typename T> T extract_impl (paramvalue const & arg, T) {
            paramvalue_reader< T > visitor;
            boost::apply_visitor(visitor, arg);
            return visitor.get_value();
        }
    }

    template<typename T> struct cast_hook<T, detail::paramvalue> {
        static inline std::complex<T> apply(detail::paramvalue const & arg) {
            return extract<T>(arg);
        }
    };

    template<typename T> T extract (detail::paramvalue const & arg) {
        detail::paramvalue_reader< T > visitor;
        boost::apply_visitor(visitor, arg);
        return visitor.get_value();
    }
}

#endif
