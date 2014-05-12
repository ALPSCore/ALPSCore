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

#ifndef ALPS_NGS_CAST_HPP
#define ALPS_NGS_CAST_HPP

#include <alps/ngs/config.hpp>
#include <alps/ngs/stacktrace.hpp>

#include <boost/bind.hpp>
#include <boost/mpl/int.hpp>
#include <boost/filesystem/path.hpp>

#include <string>
#include <complex>
#include <typeinfo>
#include <algorithm>
#include <stdexcept>
#include <cstdio>

namespace alps {

    struct bad_cast : public std::runtime_error {
        bad_cast(std::string const & arg) : std::runtime_error(arg) {}
    };

    template<typename U, typename T> inline U cast(T const &);

    namespace detail {
    
        template<typename U, typename T> struct is_cast {
            static T t;
            static char check(U);
            static double check(...);
            enum { value = sizeof(check(t)) / sizeof(char) };
        };

        template<
            typename U, typename T, typename X
        > inline U cast_generic(T arg, X) {
            throw bad_cast(
                  std::string("cannot cast from ") 
                + typeid(T).name() 
                + " to " 
                + typeid(U).name() + ALPS_STACKTRACE
            );
            return U();
        }

        template<typename U, typename T> inline U cast_generic(
            T arg, boost::mpl::int_<1> const&
        ) {
            return arg;
        }

    }

    template<typename U, typename T> struct cast_hook {
        static inline U apply(T arg) {
            return detail::cast_generic<U, T>(
                arg, boost::mpl::int_<detail::is_cast<U, T>::value>()
            );
        }
    };

    #define ALPS_NGS_CAST_STRING(T, p, c)                                            \
        template<> struct cast_hook<std::string, T > {                                \
            static inline std::string apply( T arg) {                                \
                char buffer[255];                                                    \
                if (sprintf(buffer, "%" p "" c, arg) < 0)                            \
                    throw std::runtime_error(                                        \
                        "error casting from " #T " to string" + ALPS_STACKTRACE        \
                    );                                                                \
                return buffer;                                                        \
            }                                                                        \
        };                                                                            \
        template<> struct cast_hook< T, std::string> {                                \
            static inline T apply(std::string arg) {                                \
                T value = 0;                                                        \
                if (arg.size() && sscanf(arg.c_str(), "%" c, &value) < 0)            \
                    throw std::runtime_error(                                        \
                          "error casting from string to " #T ": "                    \
                        + arg + ALPS_STACKTRACE                                        \
                    );                                                                \
                return value;                                                        \
            }                                                                        \
        };
    ALPS_NGS_CAST_STRING(short, "", "hd")
    ALPS_NGS_CAST_STRING(int, "", "d")
    ALPS_NGS_CAST_STRING(long, "", "ld")
    ALPS_NGS_CAST_STRING(unsigned short, "", "hu")
    ALPS_NGS_CAST_STRING(unsigned int, "", "u")
    ALPS_NGS_CAST_STRING(unsigned long, "", "lu")
    ALPS_NGS_CAST_STRING(float, ".8", "e")
    ALPS_NGS_CAST_STRING(double, ".16", "le")
    ALPS_NGS_CAST_STRING(long double, ".32", "Le")
    ALPS_NGS_CAST_STRING(long long, "", "lld")
    ALPS_NGS_CAST_STRING(unsigned long long, "", "llu")
    #undef ALPS_NGS_CAST_STRING

    #define ALPS_NGS_CAST_STRING_CHAR(T, U)                                            \
        template<> struct cast_hook<std::string, T > {                                \
            static inline std::string apply( T arg) {                                \
                return cast_hook<std::string, U>::apply(arg);                        \
            }                                                                        \
        };                                                                            \
        template<> struct cast_hook<T, std::string> {                                \
            static inline T apply(std::string arg) {                                \
                return cast_hook< U , std::string>::apply(arg);                        \
            }                                                                        \
        };
    ALPS_NGS_CAST_STRING_CHAR(bool, short)
    ALPS_NGS_CAST_STRING_CHAR(char, short)
    ALPS_NGS_CAST_STRING_CHAR(signed char, short)
    ALPS_NGS_CAST_STRING_CHAR(unsigned char, unsigned short)
    #undef ALPS_NGS_CAST_STRING_CHAR

    template<typename U, typename T> struct cast_hook<U, std::complex<T> > {
        static inline U apply(std::complex<T> const & arg) {
            return static_cast<U>(arg.real());
        }
    };

    template<typename U, typename T> struct cast_hook<std::complex<U>, T> {
        static inline std::complex<U> apply(T const & arg) {
            return cast<U>(arg);
        }
    };

    template<typename U, typename T> struct cast_hook<std::complex<U>, std::complex<T> > {
        static inline std::complex<U> apply(std::complex<T> const & arg) {
            return std::complex<U>(arg.real(), arg.imag());
        }
    };

    template<typename T> struct cast_hook<std::string, std::complex<T> > {
        static inline std::string apply(std::complex<T> const & arg) {
            return cast<std::string>(arg.real()) + "+" + cast<std::string>(arg.imag()) + "i";
        }
    };

    // TODO: also parse a+bi
    template<typename T> struct cast_hook<std::complex<T>, std::string> {
        static inline std::complex<T> apply(std::string const & arg) {
            return cast<T>(arg);
        }
    };
    
    template<typename T> struct cast_hook<boost::filesystem::path, T> {
        static inline boost::filesystem::path apply(T const & arg) {
            return boost::filesystem::path(cast<std::string>(arg));
        }
    };

    template<typename T> struct cast_hook<boost::filesystem::path, std::complex<T> > {
        static inline boost::filesystem::path apply(std::complex<T> const & arg) {
            return boost::filesystem::path(cast<std::string>(arg));
        }
    };

    template<typename U, typename T> inline U cast(T const & arg) {
        return cast_hook<U, T>::apply(arg);
    }

    template<typename U, typename T> inline void cast(
        U const * src, U const * end, T * dest
    ) {
        for (U const * it = src; it != end; ++it)
            dest[it - src] = cast<T>(*it);
    }
}

#endif
