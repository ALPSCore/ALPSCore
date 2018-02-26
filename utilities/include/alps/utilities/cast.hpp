/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_UTILITY_CAST_HPP
#define ALPS_UTILITY_CAST_HPP

#include <alps/config.hpp>
#include <alps/utilities/stacktrace.hpp>

#include <boost/bind.hpp>

#include <string>
#include <complex>
#include <typeinfo>
#include <type_traits>
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
        > inline U cast_generic(T /*arg*/, X) {
            throw bad_cast(
                  std::string("cannot cast from ")
                + typeid(T).name()
                + " to "
                + typeid(U).name() + ALPS_STACKTRACE
            );
            return U();
        }

        template<typename U, typename T> inline U cast_generic(
            T arg, std::integral_constant<int, 1> const&
        ) {
            return arg;
        }

    }

    template<typename U, typename T> struct cast_hook {
        static inline U apply(T arg) {
            return detail::cast_generic<U, T>(
                arg, std::integral_constant<int, detail::is_cast<U, T>::value>()
            );
        }
    };

    #define ALPS_CAST_STRING(T, p, c)                                            \
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
    ALPS_CAST_STRING(short, "", "hd")
    ALPS_CAST_STRING(int, "", "d")
    ALPS_CAST_STRING(long, "", "ld")
    ALPS_CAST_STRING(unsigned short, "", "hu")
    ALPS_CAST_STRING(unsigned int, "", "u")
    ALPS_CAST_STRING(unsigned long, "", "lu")
    ALPS_CAST_STRING(float, ".8", "e")
    ALPS_CAST_STRING(double, ".16", "le")
    ALPS_CAST_STRING(long double, ".32", "Le")
    ALPS_CAST_STRING(long long, "", "lld")
    ALPS_CAST_STRING(unsigned long long, "", "llu")
    #undef ALPS_CAST_STRING

    #define ALPS_CAST_STRING_CHAR(T, U)                                            \
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
    ALPS_CAST_STRING_CHAR(bool, short)
    ALPS_CAST_STRING_CHAR(char, short)
    ALPS_CAST_STRING_CHAR(signed char, short)
    ALPS_CAST_STRING_CHAR(unsigned char, unsigned short)
    #undef ALPS_CAST_STRING_CHAR

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
