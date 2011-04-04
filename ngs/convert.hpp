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

#include <alps/ngs/macros.hpp>

#include <stdio.h>

namespace alps {

    template<typename U, typename T> inline U convert(T arg) {
        return static_cast<U>(arg);
    }

    #define ALPS_NGDS_CONVERT_STRING(T, c)                                                                                                     \
        template<> inline std::string convert<std::string, T >( T arg) {                                                                       \
            char buffer[255];                                                                                                                  \
            if (sprintf(buffer, "%" c, arg) < 0)                                                                                               \
                ALPS_NGS_THROW_RUNTIME_ERROR("error converting to string");                                                                    \
            return buffer;                                                                                                                     \
        }                                                                                                                                      \
        template<> inline T convert< T, std::string>(std::string arg) {                                                                        \
            T value;                                                                                                                           \
            if (sscanf(arg.c_str(), "%" c, &value) < 0)                                                                                        \
                ALPS_NGS_THROW_RUNTIME_ERROR("error converting from to string");                                                               \
            return value;                                                                                                                      \
        }
    ALPS_NGDS_CONVERT_STRING(short, "hd")
    ALPS_NGDS_CONVERT_STRING(int, "d")
    ALPS_NGDS_CONVERT_STRING(long, "ld")
    ALPS_NGDS_CONVERT_STRING(unsigned short, "hu")
    ALPS_NGDS_CONVERT_STRING(unsigned int, "u")
    ALPS_NGDS_CONVERT_STRING(unsigned long, "lu")
    ALPS_NGDS_CONVERT_STRING(float, "f")
    ALPS_NGDS_CONVERT_STRING(double, "lf")
    ALPS_NGDS_CONVERT_STRING(long double, "Lf")
    ALPS_NGDS_CONVERT_STRING(long long, "Ld")
    ALPS_NGDS_CONVERT_STRING(unsigned long long, "Lu")
    #undef ALPS_NGDS_CONVERT_STRING

    #define ALPS_NGDS_CONVERT_STRING_CHAR(T, U)                                                                                                \
        template<> inline std::string convert<std::string, T >( T arg) {                                                                       \
            return convert<std::string>(static_cast< U >(arg));                                                                                \
        }                                                                                                                                      \
        template<> inline T convert<T, std::string>(std::string arg) {                                                                         \
            return static_cast< T >(convert< U >(arg));                                                                                        \
        }
    ALPS_NGDS_CONVERT_STRING_CHAR(bool, short)
    ALPS_NGDS_CONVERT_STRING_CHAR(char, short)
    ALPS_NGDS_CONVERT_STRING_CHAR(signed char, short)
    ALPS_NGDS_CONVERT_STRING_CHAR(unsigned char, unsigned short)
    #undef ALPS_NGDS_CONVERT_STRING_CHAR

    template<typename U, typename T> inline void convert(U const * src, U const * end, T * dest) {
        std::copy(src, end, dest);
    }

    #define ALPS_NGDS_CONVERT_STRING_POINTER(T)                                                                                                \
        template<> inline void convert<std::string, T >(std::string const * src, std::string const * end, T * dest) {                          \
            for (std::string const * it = src; it != end; ++it)                                                                                \
                dest[it - src] = convert<T>(*it);                                                                                              \
        }                                                                                                                                      \
        template<> inline void convert<char *, T >(char * const * src, char * const * end, T * dest) {                                         \
            for (char * const * it = src; it != end; ++it)                                                                                     \
                dest[it - src] = convert<T>(std::string(*it));                                                                                 \
        }                                                                                                                                      \
        template<> inline void convert< T , std::string >(T const * src, T const * end, std::string * dest) {                                  \
            for (T const * it = src; it != end; ++it)                                                                                          \
                dest[it - src] = convert<std::string>(*it);                                                                                    \
        }
    ALPS_NGDS_CONVERT_STRING_POINTER(bool)
    ALPS_NGDS_CONVERT_STRING_POINTER(char)
    ALPS_NGDS_CONVERT_STRING_POINTER(signed char)
    ALPS_NGDS_CONVERT_STRING_POINTER(unsigned char)
    ALPS_NGDS_CONVERT_STRING_POINTER(short)
    ALPS_NGDS_CONVERT_STRING_POINTER(unsigned short)
    ALPS_NGDS_CONVERT_STRING_POINTER(int)
    ALPS_NGDS_CONVERT_STRING_POINTER(unsigned)
    ALPS_NGDS_CONVERT_STRING_POINTER(long)
    ALPS_NGDS_CONVERT_STRING_POINTER(unsigned long)
    ALPS_NGDS_CONVERT_STRING_POINTER(long long)
    ALPS_NGDS_CONVERT_STRING_POINTER(unsigned long long)
    ALPS_NGDS_CONVERT_STRING_POINTER(float)
    ALPS_NGDS_CONVERT_STRING_POINTER(double)
    ALPS_NGDS_CONVERT_STRING_POINTER(long double)
    #undef ALPS_NGDS_CONVERT_STRING_POINTER

}

#endif
