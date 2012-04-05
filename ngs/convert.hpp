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

#ifndef ALPS_NGS_CONVERT_HPP
#define ALPS_NGS_CONVERT_HPP

#include <alps/ngs/config.hpp>
#include <alps/ngs/stacktrace.hpp>

#include <boost/bind.hpp>
#include <boost/mpl/int.hpp>

#include <string>
#include <complex>
#include <typeinfo>
#include <algorithm>
#include <stdexcept>

namespace alps {

	namespace detail {
		template<typename T> class convert_helper;
	}

	template<typename T> inline T convert(detail::convert_helper<T> data);

	namespace detail {
		
		template<typename U, typename T> struct is_cast {
			static T t;
			static char check(U);
			static double check(...);
			enum { value = sizeof(check(t)) / sizeof(char) };
		};

		template<
			typename U, typename T, typename X
		> inline U convert_generic(T arg, X) {
			throw std::runtime_error(
				  std::string("cannot cast from ") 
				+ typeid(T).name() 
				+ " to " 
				+ typeid(U).name() + ALPS_STACKTRACE
			);
			return U();
		}

		template<typename U, typename T> inline U convert_generic(
			T arg, boost::mpl::int_<1>
		) {
			return arg;
		}

		template<typename U, typename T> struct convert_hook {
			static inline U apply(T arg) {
				return convert_generic<U, T>(
					arg, boost::mpl::int_<is_cast<U, T>::value>()
				);
			}
		};

		#define ALPS_NGS_CONVERT_STRING(T, p, c)										\
			template<> struct convert_hook<std::string, T > {							\
				static inline std::string apply( T arg) {								\
					char buffer[255];													\
					if (sprintf(buffer, "%" p "" c, arg) < 0)							\
						throw std::runtime_error(										\
							"error converting from " #T " to string" + ALPS_STACKTRACE	\
						);																\
					return buffer;														\
				}																		\
			};																			\
			template<> struct convert_hook< T, std::string> {							\
				static inline T apply(std::string arg) {								\
					T value = 0;														\
					if (arg.size() && sscanf(arg.c_str(), "%" c, &value) < 0)			\
						throw std::runtime_error(										\
							  "error converting from string to " #T ": "				\
							+ arg + ALPS_STACKTRACE										\
						);																\
					return value;														\
				}																		\
			};
		ALPS_NGS_CONVERT_STRING(short, "", "hd")
		ALPS_NGS_CONVERT_STRING(int, "", "d")
		ALPS_NGS_CONVERT_STRING(long, "", "ld")
		ALPS_NGS_CONVERT_STRING(unsigned short, "", "hu")
		ALPS_NGS_CONVERT_STRING(unsigned int, "", "u")
		ALPS_NGS_CONVERT_STRING(unsigned long, "", "lu")
		ALPS_NGS_CONVERT_STRING(float, ".8", "e")
		ALPS_NGS_CONVERT_STRING(double, ".16", "le")
		ALPS_NGS_CONVERT_STRING(long double, ".32", "Le")
		ALPS_NGS_CONVERT_STRING(long long, "", "lld")
		ALPS_NGS_CONVERT_STRING(unsigned long long, "", "llu")
		#undef ALPS_NGS_CONVERT_STRING

		#define ALPS_NGS_CONVERT_STRING_CHAR(T, U)										\
			template<> struct convert_hook<std::string, T > {							\
				static inline std::string apply( T arg) {								\
					return convert_hook<std::string, U>::apply(arg);					\
				}																		\
			};																			\
			template<> struct convert_hook<T, std::string> {							\
				static inline T apply(std::string arg) {								\
					return convert_hook< U , std::string>::apply(arg);					\
				}																		\
			};
		ALPS_NGS_CONVERT_STRING_CHAR(bool, short)
		ALPS_NGS_CONVERT_STRING_CHAR(char, short)
		ALPS_NGS_CONVERT_STRING_CHAR(signed char, short)
		ALPS_NGS_CONVERT_STRING_CHAR(unsigned char, unsigned short)
		#undef ALPS_NGS_CONVERT_STRING_CHAR

		template<typename U, typename T> struct convert_hook<U, std::complex<T> > {
			static inline U apply(std::complex<T> const & arg) {
				return static_cast<U>(arg.real());
			}
		};

		template<typename U, typename T> struct convert_hook<std::complex<U>, T> {
			static inline std::complex<U> apply(T const & arg) {
				return convert<U>(arg);
			}
		};

		template<typename U, typename T> struct convert_hook<std::complex<U>, std::complex<T> > {
			static inline std::complex<U> apply(std::complex<T> const & arg) {
				return std::complex<U>(arg.real(), arg.imag());
			}
		};

		template<typename T> struct convert_hook<std::string, std::complex<T> > {
			static inline std::string apply(std::complex<T> const & arg) {
				return convert<std::string>(arg.real()) + "+" + convert<std::string>(arg.imag()) + "i";
			}
		};

		// TODO: also parse a+bi
		template<typename T> struct convert_hook<std::complex<T>, std::string> {
			static inline std::complex<T> apply(std::string const & arg) {
				return convert<T>(arg);
			}
		};

		template<typename U> class convert_helper {

			public:

				template<typename T> convert_helper(T arg)
					: value(convert_hook<U, T>::apply(arg))
				{}
				
				U const & operator()() {
					return value;
				}

			private:

				U value;
		};
    }

	template<typename T> inline T convert(T const & data) {
		return data;
	}

	template<typename T> inline T convert(detail::convert_helper<T> data) {
		return data();
	}

    template<typename U, typename T> inline void convert(
		U const * src, U const * end, T * dest
	) {
		for (U const * it = src; it != end; ++it)
			dest[it - src] = convert<T>(*it);
    }
}

#endif
