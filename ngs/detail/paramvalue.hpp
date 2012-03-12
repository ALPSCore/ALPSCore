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

#ifndef ALPS_NGS_PARAMVALUE_HPP
#define ALPS_NGS_PARAMVALUE_HPP

#include <alps/ngs/hdf5.hpp>
#include <alps/ngs/config.hpp>
#include <alps/ngs/convert.hpp>

#if defined(ALPS_HAVE_PYTHON)
	#include <alps/ngs/boost_python.hpp>
#endif

#include <boost/variant.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/pop_back.hpp>

#include <string>
#include <complex>
#include <stdexcept>

#define ALPS_NGS_FOREACH_PARAMETERVALUE_TYPE_NO_PYTHON(CALLBACK)					\
	CALLBACK(double)																\
	CALLBACK(signed int)															\
	CALLBACK(bool)																	\
	CALLBACK(std::string)															\
	CALLBACK(std::complex<double>)													\
	CALLBACK(std::vector<double>)													\
	CALLBACK(std::vector<int>)														\
	CALLBACK(std::vector<std::string>)												\
	CALLBACK(std::vector<std::complex<double> >)

#if defined(ALPS_HAVE_PYTHON)
	#define ALPS_NGS_FOREACH_PARAMETERVALUE_TYPE(CALLBACK)							\
		ALPS_NGS_FOREACH_PARAMETERVALUE_TYPE_NO_PYTHON(CALLBACK)					\
		CALLBACK(boost::python::object)
#else
	#define ALPS_NGS_FOREACH_PARAMETERVALUE_TYPE(CALLBACK)							\
		ALPS_NGS_FOREACH_PARAMETERVALUE_TYPE_NO_PYTHON(CALLBACK)
#endif

namespace alps {

	namespace detail {

		class paramvalue;

	}

    #define ALPS_NGS_PARAMETERVALUE_CONVERT_DECL(T)									\
        template<> ALPS_DECL T convert(detail::paramvalue const &					\
		);
	ALPS_NGS_FOREACH_PARAMETERVALUE_TYPE(ALPS_NGS_PARAMETERVALUE_CONVERT_DECL)
	#undef ALPS_NGS_PARAMETERVALUE_CONVERT_DECL

    namespace detail {

		#define ALPS_NGS_PARAMVALUE_VARIANT_TYPE(T)	T,
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

				template<typename T> T cast() const {
					return convert<T, detail::paramvalue const &>(*this);
				}

				#define ALPS_NGS_PARAMVALUE_MEMBER_DECL(T)							\
					paramvalue( T const & v): paramvalue_base(v) {}					\
					operator T () const;											\
					paramvalue & operator=( T const &);
				ALPS_NGS_FOREACH_PARAMETERVALUE_TYPE(ALPS_NGS_PARAMVALUE_MEMBER_DECL)
				#undef ALPS_NGS_PARAMVALUE_MEMBER_DECL

				void save(hdf5::archive &) const;
				void load(hdf5::archive &);
		};

    }
}

#endif
