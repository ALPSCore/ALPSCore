/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2010 by Lukas Gamper <gamperl -at- gmail.com>
*
* This software is part of the ALPS libraries, published under the ALPS
* Library License; you can use, redistribute it and/or modify it under
* the terms of the license, either version 1 or (at your option) any later
* version.
* 
* You should have received a copy of the ALPS Library License along with
* the ALPS Libraries; see the file LICENSE.txt. If not, the license is also
* available from http://alps.comp-phys.org/.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
* FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT 
* SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE 
* FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, 
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/

#include <alps/hdf5.hpp>

#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

#include <boost/filesystem.hpp>

/*
#include <boost/test/test_tools.hpp>
#include <boost/test/included/test_exec_monitor.hpp>

BOOST_AUTO_TEST_CASE( test_write_string ) {
    alps::hdf5::oarchive ar("scalar.h5");
    {
        std::string value = "Im a Test string";
        ar << alps::make_pvp("/scalar/std::string/value", value);
    }
    {
        std::string value;
        ar << alps::make_pvp("/scalar/std::string/empty", value);
    }
    {
        ar << alps::make_pvp("/scalar/c_string/value", "me 2");
    }
    {
        std::string value;
        ar << alps::make_pvp("/scalar/c_string/emtpy", "");
    }
}
int test_main() {
    test_write_string(oar);
    return 0;
}
*/

#define HDF5_WRITE(T)                                                                                                                                      \
    {                                                                                                                                                      \
        oar << alps::make_pvp("/scalar/" + oar.encode_segment(#T) + "/value1", static_cast<T>(1));                                                         \
        oar << alps::make_pvp("/scalar/" + oar.encode_segment(#T) + "/value2", static_cast<T>(-1));                                                        \
        oar << alps::make_pvp("/scalar/" + oar.encode_segment(#T) + "/value3", static_cast<T>(2.85));                                                      \
        oar << alps::make_pvp("/scalar/" + oar.encode_segment(#T) + "/value4", static_cast<T>(-38573.4));                                                  \
        oar << alps::make_pvp("/scalar/" + oar.encode_segment(#T) + "/value5", static_cast<T>(1));                                                         \
        oar << alps::make_pvp("/scalar/" + oar.encode_segment(#T) + "/value6", static_cast<T>(0));                                                         \
    }                                                                                                                                                      \
    {                                                                                                                                                      \
        oar << alps::make_pvp("/scalar/" + oar.encode_segment(#T) + "/complex1", std::complex<T>(1, 2));                                                   \
        oar << alps::make_pvp("/scalar/" + oar.encode_segment(#T) + "/complex2", std::complex<T>(static_cast<T>(-1), 2));                                  \
        oar << alps::make_pvp("/scalar/" + oar.encode_segment(#T) + "/complex3", std::complex<T>(static_cast<T>(1.2342), static_cast<T>(-2.93845)));       \
    }
#define HDF5_READ(T)                                                                                                                                       \
    {                                                                                                                                                      \
        std::cout << #T << "-values: ";                                                                                                                    \
        T value;                                                                                                                                           \
        iar >> alps::make_pvp("/scalar/" + iar.encode_segment(#T) + "/value1", value);                                                                     \
        std::cout << value << ", ";                                                                                                                        \
        iar >> alps::make_pvp("/scalar/" + iar.encode_segment(#T) + "/value2", value);                                                                     \
        std::cout << value << ", ";                                                                                                                        \
        iar >> alps::make_pvp("/scalar/" + iar.encode_segment(#T) + "/value3", value);                                                                     \
        std::cout << value << ", ";                                                                                                                        \
        iar >> alps::make_pvp("/scalar/" + iar.encode_segment(#T) + "/value4", value);                                                                     \
        std::cout << value << ", ";                                                                                                                        \
        iar >> alps::make_pvp("/scalar/" + iar.encode_segment(#T) + "/value5", value);                                                                     \
        std::cout << value << ", ";                                                                                                                        \
        iar >> alps::make_pvp("/scalar/" + iar.encode_segment(#T) + "/value6", value);                                                                     \
        std::cout << value << std::endl;                                                                                                                   \
    }                                                                                                                                                      \
    {                                                                                                                                                      \
        std::cout << #T << "-double: ";                                                                                                                    \
        long double value;                                                                                                                                 \
        iar >> alps::make_pvp("/scalar/" + iar.encode_segment(#T) + "/value1", value);                                                                     \
        std::cout << value << ", ";                                                                                                                        \
        iar >> alps::make_pvp("/scalar/" + iar.encode_segment(#T) + "/value2", value);                                                                     \
        std::cout << value << ", ";                                                                                                                        \
        iar >> alps::make_pvp("/scalar/" + iar.encode_segment(#T) + "/value3", value);                                                                     \
        std::cout << value << ", ";                                                                                                                        \
        iar >> alps::make_pvp("/scalar/" + iar.encode_segment(#T) + "/value4", value);                                                                     \
        std::cout << value << ", ";                                                                                                                        \
        iar >> alps::make_pvp("/scalar/" + iar.encode_segment(#T) + "/value5", value);                                                                     \
        std::cout << value << ", ";                                                                                                                        \
        iar >> alps::make_pvp("/scalar/" + iar.encode_segment(#T) + "/value6", value);                                                                     \
        std::cout << value << std::endl;                                                                                                                   \
    }                                                                                                                                                      \
    {                                                                                                                                                      \
        std::cout << #T << "-long: ";                                                                                                                      \
        long long value;                                                                                                                                   \
        iar >> alps::make_pvp("/scalar/" + iar.encode_segment(#T) + "/value1", value);                                                                     \
        std::cout << value << ", ";                                                                                                                        \
        iar >> alps::make_pvp("/scalar/" + iar.encode_segment(#T) + "/value2", value);                                                                     \
        std::cout << value << ", ";                                                                                                                        \
        iar >> alps::make_pvp("/scalar/" + iar.encode_segment(#T) + "/value3", value);                                                                     \
        std::cout << value << ", ";                                                                                                                        \
        iar >> alps::make_pvp("/scalar/" + iar.encode_segment(#T) + "/value4", value);                                                                     \
        std::cout << value << ", ";                                                                                                                        \
        iar >> alps::make_pvp("/scalar/" + iar.encode_segment(#T) + "/value5", value);                                                                     \
        std::cout << value << ", ";                                                                                                                        \
        iar >> alps::make_pvp("/scalar/" + iar.encode_segment(#T) + "/value6", value);                                                                     \
        std::cout << value << std::endl;                                                                                                                   \
    }                                                                                                                                                      \
    {                                                                                                                                                      \
        std::cout << #T << "-string: ";                                                                                                                    \
        std::string value;                                                                                                                                 \
        iar >> alps::make_pvp("/scalar/" + iar.encode_segment(#T) + "/value1", value);                                                                     \
        std::cout << value << ", ";                                                                                                                        \
        iar >> alps::make_pvp("/scalar/" + iar.encode_segment(#T) + "/value2", value);                                                                     \
        std::cout << value << ", ";                                                                                                                        \
        iar >> alps::make_pvp("/scalar/" + iar.encode_segment(#T) + "/value3", value);                                                                     \
        std::cout << value << ", ";                                                                                                                        \
        iar >> alps::make_pvp("/scalar/" + iar.encode_segment(#T) + "/value4", value);                                                                     \
        std::cout << value << ", ";                                                                                                                        \
        iar >> alps::make_pvp("/scalar/" + iar.encode_segment(#T) + "/value5", value);                                                                     \
        std::cout << value << ", ";                                                                                                                        \
        iar >> alps::make_pvp("/scalar/" + iar.encode_segment(#T) + "/value6", value);                                                                     \
        std::cout << value << std::endl;                                                                                                                   \
    }                                                                                                                                                      \
    {                                                                                                                                                      \
        std::cout << #T << "-complex: ";                                                                                                                   \
        std::complex<T> value;                                                                                                                             \
        iar >> alps::make_pvp("/scalar/" + iar.encode_segment(#T) + "/complex1", value);                                                                   \
        std::cout << "(" << value.real() << ", " << value.imag() << "), ";                                                                                 \
        iar >> alps::make_pvp("/scalar/" + iar.encode_segment(#T) + "/complex2", value);                                                                   \
        std::cout << "(" << value.real() << ", " << value.imag() << "), ";                                                                                 \
        iar >> alps::make_pvp("/scalar/" + iar.encode_segment(#T) + "/complex3", value);                                                                   \
        std::cout << "(" << value.real() << ", " << value.imag() << "), " << std::endl;                                                                    \
    }
#define HDF5_FOREACH(callback)                                                                                                                             \
    callback(bool)                                                                                                                                         \
    callback(char)                                                                                                                                         \
    callback(unsigned char)                                                                                                                                \
    callback(short)                                                                                                                                        \
    callback(unsigned short)                                                                                                                               \
    callback(int)                                                                                                                                          \
    callback(unsigned int)                                                                                                                                 \
    callback(long)                                                                                                                                         \
    callback(unsigned long)                                                                                                                                \
    callback(long long)                                                                                                                                    \
    callback(unsigned long long)                                                                                                                           \
    callback(float)                                                                                                                                        \
    callback(double)                                                                                                                                       \
    callback(long double)                                                                                                                                  \
    callback(boost::int8_t)                                                                                                                                \
    callback(boost::uint8_t)                                                                                                                               \
    callback(boost::int16_t)                                                                                                                               \
    callback(boost::uint16_t)                                                                                                                              \
    callback(boost::int32_t)                                                                                                                               \
    callback(boost::uint32_t)                                                                                                                              \
    callback(boost::int64_t)                                                                                                                               \
    callback(boost::uint64_t)
int main() {
    {
        alps::hdf5::oarchive oar("scalar.h5");
        HDF5_FOREACH(HDF5_WRITE)
        {
            std::string value = "Im a Test string";
            oar << alps::make_pvp("/scalar/std::string/value", value);
        }
        {
            std::string value;
            oar << alps::make_pvp("/scalar/std::string/empty", value);
        }
        {
            oar << alps::make_pvp("/scalar/c_string/value", "me 2");
        }
        {
            std::string value;
            oar << alps::make_pvp("/scalar/c_string/emtpy", "");
        }
    }
    {
        alps::hdf5::iarchive iar("scalar.h5");
        HDF5_FOREACH(HDF5_READ)
        {
            std::string value;
            iar >> alps::make_pvp("/scalar/std::string/value", value);
            std::cout << "std::string-value (" << value.size() << "): " << value << std::endl;
        }
        {
            std::string value;
            iar >> alps::make_pvp("/scalar/std::string/empty", value);
            std::cout << "std::string-empty (" << value.size() << ") " << value << std::endl;
        }
        {
            std::string value;
            iar >> alps::make_pvp("/scalar/c_string/value", value);
            std::cout << "c_string-value (" << value.size() << "): " << value << std::endl;
        }
        {
            std::string value;
            iar >> alps::make_pvp("/scalar/c_string/emtpy", value);
            std::cout << "c_string-empty (" << value.size() << ") " << value << std::endl;
        }
    }
    boost::filesystem::remove(boost::filesystem::path("scalar.h5"));
}
