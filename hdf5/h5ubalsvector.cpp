#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

#include <boost/filesystem.hpp>

#include <alps/hdf5.hpp>

#define HDF5_WRITE(T)                                                                                                                                      \
    {                                                                                                                                                      \
        boost::numeric::ublas::vector< T > value(1000);                                                                                                    \
        for (std::size_t i = 0; i < 1000; ++i)                                                                                                             \
            value[i] = i;                                                                                                                                  \
            oar << alps::make_pvp("/vector/" + oar.encode_segment(#T) + "/scalar", value);                                                                 \
    }                                                                                                                                                      \
    {                                                                                                                                                      \
        boost::numeric::ublas::vector< T > value;                                                                                                          \
        oar << alps::make_pvp("/vector/" + oar.encode_segment(#T) + "/null", value);                                                                       \
    }                                                                                                                                                      \
    {                                                                                                                                                      \
        boost::numeric::ublas::vector<std::complex< T > > value(1000);                                                                                     \
        value[0] = std::complex<T>(1, 2);                                                                                                                  \
        value[1] = std::complex<T>(static_cast<T>(-1), 2);                                                                                                 \
        value[2] = std::complex<T>(static_cast<T>(1.2342), static_cast<T>(-2.93845));                                                                      \
        oar << alps::make_pvp("/vector/" + oar.encode_segment(#T) + "/complex", value);                                                                    \
    }
#define HDF5_READ(T)                                                                                                                                       \
    {                                                                                                                                                      \
        boost::numeric::ublas::vector< T > value;                                                                                                          \
        iar >> alps::make_pvp("/vector/" + iar.encode_segment(#T) + "/scalar", value);                                                                     \
        std::cout << #T << "-scalar (" << value.size() << "): [";                                                                                          \
        for (std::size_t i = 0; i < 20;  ++i)                                                                                                              \
            std::cout << value[i] << ", ";                                                                                                                 \
        std::cout << "...]" << std::endl;                                                                                                                  \
    }                                                                                                                                                      \
    {                                                                                                                                                      \
        boost::numeric::ublas::vector< T > value;                                                                                                          \
        iar >> alps::make_pvp("/vector/" + iar.encode_segment(#T) + "/null", value);                                                                       \
        std::cout << #T << "-null (" << value.size() << ")" << std::endl;                                                                                  \
    }                                                                                                                                                      \
    {                                                                                                                                                      \
        boost::numeric::ublas::vector<std::complex< T > > value;                                                                                           \
        iar >> alps::make_pvp("/vector/" + iar.encode_segment(#T) + "/complex", value);                                                                    \
        std::cout << #T << "-complex (" << value.size() << "): [";                                                                                         \
        for (std::size_t i = 0; i < 3;  ++i)                                                                                                               \
            std::cout << "(" << value[i].real() << ", " << value[i].imag() << "), ";                                                                       \
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
    {
        alps::hdf5::oarchive oar("vector.h5");
        HDF5_FOREACH(HDF5_WRITE)
    }
    {
        alps::hdf5::iarchive iar("vector.h5");
        HDF5_FOREACH(HDF5_READ)
    }
    boost::filesystem::remove(boost::filesystem::path("vector.h5"));
}
