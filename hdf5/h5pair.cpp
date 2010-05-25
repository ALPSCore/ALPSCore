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

const std::size_t dim1 = 10;
const std::size_t dim2 = 7;
const std::size_t dim3 = 17;
const std::size_t length = dim1 * dim2 * dim3;

#define HDF5_WRITE(T)                                                                                                                                      \
    {                                                                                                                                                      \
        T data[length];                                                                                                                                    \
        std::pair<T*, std::vector<std::size_t> > value(data, size);                                                                                        \
        for (std::size_t i = 0; i < length; ++i)                                                                                                           \
            value.first[i] = i;                                                                                                                            \
        oar << alps::make_pvp("/pair/" + oar.encode_segment(#T) + "/scalar", value);                                                                       \
    }                                                                                                                                                      \
    {                                                                                                                                                      \
        T data[length];                                                                                                                                    \
        for (std::size_t i = 0; i < length; ++i)                                                                                                           \
            data[i] = i;                                                                                                                                   \
        std::pair<T const *, std::vector<std::size_t> > value(data, size);                                                                                 \
        oar << alps::make_pvp("/pair/" + oar.encode_segment(#T) + "/const-scalar", value);                                                                 \
    }                                                                                                                                                      \
    {                                                                                                                                                      \
        std::pair<T const *, std::vector<std::size_t> > value(NULL, std::vector<std::size_t>(3, 0));                                                       \
        oar << alps::make_pvp("/pair/" + oar.encode_segment(#T) + "/null", value);                                                                         \
    }                                                                                                                                                      \
    {                                                                                                                                                      \
        std::complex< T > data[1000];                                                                                                                      \
        std::pair<std::complex< T > *, std::vector<std::size_t> > value(data, size);                                                                       \
        value.first[0] = std::complex<T>(1, 2);                                                                                                            \
        value.first[1] = std::complex<T>(static_cast<T>(-1), 2);                                                                                           \
        value.first[2] = std::complex<T>(static_cast<T>(1.2342), static_cast<T>(-2.93845));                                                                \
        oar << alps::make_pvp("/pair/" + oar.encode_segment(#T) + "/complex", value);                                                                      \
    }
#define HDF5_READ(T)                                                                                                                                       \
    {                                                                                                                                                      \
        T data[length];                                                                                                                                    \
        std::pair<T*, std::vector<std::size_t> > value(data, size);                                                                                        \
        iar >> alps::make_pvp("/pair/" + iar.encode_segment(#T) + "/scalar", value);                                                                       \
        std::cout << #T << "-scalar: [";                                                                                                                   \
        for (std::size_t i = 0; i < 20;  ++i)                                                                                                              \
            std::cout << value.first[i] << ", ";                                                                                                           \
        std::cout << "...]" << std::endl;                                                                                                                  \
    }                                                                                                                                                      \
    {                                                                                                                                                      \
        T data[length];                                                                                                                                    \
        std::pair<T*, std::vector<std::size_t> > value(data, size);                                                                                        \
        iar >> alps::make_pvp("/pair/" + iar.encode_segment(#T) + "/const-scalar", value);                                                                 \
        std::cout << #T << "-scalar: [";                                                                                                                   \
        for (std::size_t i = 0; i < 20;  ++i)                                                                                                              \
            std::cout << value.first[i] << ", ";                                                                                                           \
        std::cout << "...]" << std::endl;                                                                                                                  \
    }                                                                                                                                                      \
    {                                                                                                                                                      \
        std::pair<T*, std::vector<std::size_t> > value(NULL, std::vector<std::size_t>(3, 0));                                                              \
        iar >> alps::make_pvp("/pair/" + iar.encode_segment(#T) + "/null", value);                                                                         \
        std::cout << #T << "-null" << std::endl;                                                                                                           \
    }                                                                                                                                                      \
    {                                                                                                                                                      \
        std::complex< T > data[length];                                                                                                                    \
        std::pair<std::complex< T > *, std::vector<std::size_t> > value(data, size);                                                                       \
        iar >> alps::make_pvp("/pair/" + iar.encode_segment(#T) + "/complex", value);                                                                      \
        std::cout << #T << "-complex: [";                                                                                                                  \
        for (std::size_t i = 0; i < 3;  ++i)                                                                                                               \
            std::cout << "(" << value.first[i].real() << ", " << value.first[i].imag() << "), ";                                                           \
        std::cout << "...]" << std::endl;                                                                                                                  \
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
    std::vector<std::size_t> size(3, dim1);
    size[1] = dim2;
    size[2] = dim3;
    {
        alps::hdf5::oarchive oar("pair.h5");
        HDF5_FOREACH(HDF5_WRITE)
    }
    {
        alps::hdf5::iarchive iar("pair.h5");
        HDF5_FOREACH(HDF5_READ)
    }
    boost::filesystem::remove(boost::filesystem::path("pair.h5"));
}
