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
#include <boost/random.hpp>

boost::mt19937 rng;
boost::uniform_int<> dist_int(0,128);
boost::variate_generator<boost::mt19937, boost::uniform_int<> > random_int(rng, dist_int);
boost::uniform_real<> dist_real(0.,1e12);
boost::variate_generator<boost::mt19937, boost::uniform_real<> > random_real(rng, dist_real);
template<typename T> typename boost::enable_if<typename boost::is_integral<T>::type, T>::type construct(T const &, std::vector<std::size_t> const & size = std::vector<std::size_t>()) {
    return T(random_int());
}
template<typename T> typename boost::enable_if<typename boost::is_floating_point<T>::type, T>::type construct(T const &, std::vector<std::size_t> const & size = std::vector<std::size_t>()) {
    return T(random_real());
}
template<typename T> std::complex<T> construct(std::complex<T> const &, std::vector<std::size_t> const & size = std::vector<std::size_t>()) {
    return std::complex<T>(construct(T()), construct(T()));
}
std::string construct(std::string const &, std::vector<std::size_t> const & size = std::vector<std::size_t>()) {
    return std::string(boost::lexical_cast<std::string>(random_real()) + boost::lexical_cast<std::string>(random_real()) + boost::lexical_cast<std::string>(random_real()));
}
template<typename T> bool equal(T & arg1, T & arg2) {
    return arg1 == arg2;
}
template<typename T> void test(std::string const & name) {
    T write_random(construct(T()));
    T write_empty;
    T read_random;
    T read_empty;
    {
        alps::hdf5::oarchive ar("test.h5");
        ar << alps::make_pvp("/random/" + name, write_random);
        ar << alps::make_pvp("/empty/" + name, write_empty);
    }
    {
        alps::hdf5::iarchive ar("test.h5");
        ar >> alps::make_pvp("/random/" + name, read_random);
        ar >> alps::make_pvp("/empty/" + name, read_empty);
    }
    if (!equal(write_random, read_empty)) {
        std::cerr << name << "-empty FAILED" << std::endl;
        std::abort();
    } else if (!equal(write_random, read_random)) {
        std::cerr << name << "-random FAILED" << std::endl;
        std::abort();
    } else
        std::cout << name << " SUCCESS" << std::endl;
}
#define HDF5_FOREACH_TYPE(type)                                                                                                                            \
    test<type>(#type);
#define HDF5_FOREACH_COMPLEX(type)                                                                                                                         \
    HDF5_FOREACH_TYPE(std::complex< type >)
#define HDF5_FOREACH_SCALAR(callback)                                                                                                                      \
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
#define HDF5_FOREACH_NATIVE()                                                                                                                              \
    HDF5_FOREACH_SCALAR(HDF5_FOREACH_TYPE)                                                                                                                 \
    HDF5_FOREACH_SCALAR(HDF5_FOREACH_COMPLEX)                                                                                                              \
    HDF5_FOREACH_TYPE(std::string)/*                                                                                                                         \
    HDF5_FOREACH_TYPE(class_type)                                                                                                                         \
    HDF5_FOREACH_TYPE(enum_type)                                                                                                                         \
*/
int main() {
    boost::filesystem::remove(boost::filesystem::path("test.h5"));
    HDF5_FOREACH_NATIVE()
    boost::filesystem::remove(boost::filesystem::path("test.h5"));
    return EXIT_SUCCESS;
}
