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

#include <alps/ngs/convert.hpp>
#include <alps/ngs/stacktrace.hpp>

#include <stdio.h>
#include <stdexcept>

namespace alps {

    #define ALPS_NGS_CONVERT_STRING(T, p, c)                                                                                                   \
        template<> ALPS_DECL std::string convert<std::string, T >( T arg) {                                                                    \
            char buffer[255];                                                                                                                  \
            if (sprintf(buffer, "%" p "" c, arg) < 0)                                                                                          \
                throw std::runtime_error("error converting from " #T " to string" + ALPS_STACKTRACE);                                                        \
            return buffer;                                                                                                                     \
        }                                                                                                                                      \
        template<> ALPS_DECL T convert< T, std::string>(std::string arg) {                                                                     \
            T value = 0;                                                                                                                       \
            if (arg.size() && sscanf(arg.c_str(), "%" c, &value) < 0)                                                                          \
                throw std::runtime_error("error converting from string to " #T ": " + arg + ALPS_STACKTRACE);                                                \
            return value;                                                                                                                      \
        }
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

    #define ALPS_NGS_CONVERT_STRING_CHAR(T, U)                                                                                                 \
        template<> ALPS_DECL std::string convert<std::string, T >( T arg) {                                                                              \
            return convert<std::string>(static_cast< U >(arg));                                                                                \
        }                                                                                                                                      \
        template<> ALPS_DECL T convert<T, std::string>(std::string arg) {                                                                                \
            return static_cast< T >(convert< U >(arg));                                                                                        \
        }
    ALPS_NGS_CONVERT_STRING_CHAR(bool, short)
    ALPS_NGS_CONVERT_STRING_CHAR(char, short)
    ALPS_NGS_CONVERT_STRING_CHAR(signed char, short)
    ALPS_NGS_CONVERT_STRING_CHAR(unsigned char, unsigned short)
    #undef ALPS_NGS_CONVERT_STRING_CHAR

    #define ALPS_NGS_CONVERT_STRING_POINTER(T)                                                                                                 \
        template<> ALPS_DECL void convert<std::string, T >(std::string const * src, std::string const * end, T * dest) {                                 \
            for (std::string const * it = src; it != end; ++it)                                                                                \
                dest[it - src] = convert<T>(*it);                                                                                              \
        }                                                                                                                                      \
        template<> ALPS_DECL void convert<char *, T >(char * const * src, char * const * end, T * dest) {                                                \
            for (char * const * it = src; it != end; ++it)                                                                                     \
                dest[it - src] = convert<T>(std::string(*it));                                                                                 \
        }                                                                                                                                      \
        template<> ALPS_DECL void convert< T , std::string >(T const * src, T const * end, std::string * dest) {                                         \
            for (T const * it = src; it != end; ++it)                                                                                          \
                dest[it - src] = convert<std::string>(*it);                                                                                    \
        }
    ALPS_NGS_CONVERT_STRING_POINTER(bool)
    ALPS_NGS_CONVERT_STRING_POINTER(char)
    ALPS_NGS_CONVERT_STRING_POINTER(signed char)
    ALPS_NGS_CONVERT_STRING_POINTER(unsigned char)
    ALPS_NGS_CONVERT_STRING_POINTER(short)
    ALPS_NGS_CONVERT_STRING_POINTER(unsigned short)
    ALPS_NGS_CONVERT_STRING_POINTER(int)
    ALPS_NGS_CONVERT_STRING_POINTER(unsigned)
    ALPS_NGS_CONVERT_STRING_POINTER(long)
    ALPS_NGS_CONVERT_STRING_POINTER(unsigned long)
    ALPS_NGS_CONVERT_STRING_POINTER(long long)
    ALPS_NGS_CONVERT_STRING_POINTER(unsigned long long)
    ALPS_NGS_CONVERT_STRING_POINTER(float)
    ALPS_NGS_CONVERT_STRING_POINTER(double)
    ALPS_NGS_CONVERT_STRING_POINTER(long double)
    #undef ALPS_NGS_CONVERT_STRING_POINTER

}
