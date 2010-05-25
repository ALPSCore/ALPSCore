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

#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

#include <boost/filesystem.hpp>

#include <alps/hdf5.hpp>

#define HDF5_WRITE(T)                                                                                                                                      \
    {                                                                                                                                                      \
        std::vector<std::vector< T > > value;                                                                                                              \
        for (std::size_t i = 0; i < 3; ++i) {                                                                                                              \
            value.push_back(std::vector< T >());                                                                                                           \
            for (std::size_t j = 0; j < 4; ++j)                                                                                                            \
                value[i].push_back(i * j);                                                                                                                 \
        }                                                                                                                                                  \
        oar << alps::make_pvp("/discontinous/" + oar.encode_segment(#T) + "/scalar", value);                                                               \
    }                                                                                                                                                      \
    {                                                                                                                                                      \
        std::vector< std::vector<T> > value;                                                                                                               \
        oar << alps::make_pvp("/discontinous/" + oar.encode_segment(#T) + "/null", value);                                                                 \
    }                                                                                                                                                      \
    {                                                                                                                                                      \
        std::vector<std::vector<std::complex< T > > > value;                                                                                               \
        value.push_back(std::vector<std::complex<T> >(4, std::complex<T>(1, 2)));                                                                          \
        value.push_back(std::vector<std::complex<T> >(4, std::complex<T>(static_cast<T>(-1), 2)));                                                         \
        value.push_back(std::vector<std::complex<T> >(4, std::complex<T>(static_cast<T>(1.2342), static_cast<T>(-2.93845))));                              \
        oar << alps::make_pvp("/discontinous/" + oar.encode_segment(#T) + "/complex", value);                                                              \
    }
#define HDF5_READ(T)                                                                                                                                       \
    {                                                                                                                                                      \
        std::vector<std::vector< T > > value;                                                                                                              \
        iar >> alps::make_pvp("/discontinous/" + iar.encode_segment(#T) + "/scalar", value);                                                               \
        std::cout << #T << "-scalar (" << value.size() << ", " << value[0].size() << "): [";                                                               \
        for (std::size_t i = 0; i < 3; ++i) {                                                                                                              \
            std::cout << "[";                                                                                                                              \
            for (std::size_t j = 0; j < 4; ++j)                                                                                                            \
                std::cout << value[i][j] << (j < 3 ? ", " : "");                                                                                           \
            std::cout << "]" << (i < 2 ? ", " : "");                                                                                                       \
        }                                                                                                                                                  \
        std::cout << "]" << std::endl;                                                                                                                     \
    }                                                                                                                                                      \
    {                                                                                                                                                      \
        std::vector< std::vector<T> > value;                                                                                                               \
        iar >> alps::make_pvp("/discontinous/" + iar.encode_segment(#T) + "/null", value);                                                                 \
        std::cout << #T << "-null (" << value.size() << ")" << std::endl;                                                                                  \
    }                                                                                                                                                      \
    {                                                                                                                                                      \
        std::vector<std::vector<std::complex< T > > > value;                                                                                               \
        iar >> alps::make_pvp("/discontinous/" + iar.encode_segment(#T) + "/complex", value);                                                              \
        std::cout << #T << "-complex (" << value.size() << ", " << value[0].size() << "): [";                                                              \
        for (std::size_t i = 0; i < 3; ++i) {                                                                                                              \
            std::cout << "[";                                                                                                                              \
            for (std::size_t j = 0; j < 4; ++j)                                                                                                            \
                std::cout << "(" << value[i][j].real() << ", " << value[i][j].imag() << ")" << (j < 3 ? ", " : "");                                        \
            std::cout << "]" << (i < 2 ? ", " : "");                                                                                                       \
        }                                                                                                                                                  \
        std::cout << "]" << std::endl;                                                                                                                     \
    }
#define HDF5_FOREACH(callback)                                                                                                                             \
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
    callback(long double)
int main() {
    {
        alps::hdf5::oarchive oar("discontinous.h5");
        HDF5_FOREACH(HDF5_WRITE)
        {
            std::vector<std::string> value;
            value.push_back("Im a Test string");
            value.push_back("me 2");
            oar << alps::make_pvp("/discontinous/std::string/scalar", std::vector<std::vector<std::string> >(1, value));
        }
        {
            std::vector<std::vector<std::string> > value;
            oar << alps::make_pvp("/discontinous/std::string/null", value);
        }
    }
    {
        alps::hdf5::iarchive iar("discontinous.h5");
        HDF5_FOREACH(HDF5_READ)
        {
            std::vector<std::vector<std::string> > value;
            iar >> alps::make_pvp("/discontinous/std::string/scalar", value);
            std::cout << "std::string-scalar (" << value.size() << "," << value[0].size() << "): [";
            for (std::vector<std::string>::const_iterator it = value[0].begin(); it != value[0].end(); ++it)
                std::cout << *it << (it + 1 != value[0].end() ? ", " : "");
            std::cout << "]" << std::endl;
        }
        {
            std::vector<std::vector<std::string> > value;
            iar >> alps::make_pvp("/discontinous/std::string/null", value);
            std::cout << "std::string-null (" << value.size() << ")" << std::endl;
        }
    }
    boost::filesystem::remove(boost::filesystem::path("discontinous.h5"));
}
