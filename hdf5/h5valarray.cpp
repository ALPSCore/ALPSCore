#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

#include <boost/filesystem.hpp>

#include <alps/hdf5.hpp>

#define HDF5_WRITE(T)                                                                                                                                      \
    {                                                                                                                                                      \
        std::valarray< T > value(1000);                                                                                                                    \
        for (std::size_t i = 0; i < 1000; ++i)                                                                                                             \
            value[i] = i;                                                                                                                                  \
            oar << alps::make_pvp("/vector/" + oar.encode_segment(#T) + "/scalar", value);                                                                 \
    }                                                                                                                                                      \
    {                                                                                                                                                      \
        std::valarray< T > value;                                                                                                                          \
        oar << alps::make_pvp("/vector/" + oar.encode_segment(#T) + "/null", value);                                                                       \
    }                                                                                                                                                      \
    {                                                                                                                                                      \
        std::valarray<std::complex< T > > value(1000);                                                                                                     \
        value[0] = std::complex<T>(1, 2);                                                                                                                  \
        value[1] = std::complex<T>(static_cast<T>(-1), 2);                                                                                                 \
        value[2] = std::complex<T>(static_cast<T>(1.2342), static_cast<T>(-2.93845));                                                                      \
        oar << alps::make_pvp("/vector/" + oar.encode_segment(#T) + "/complex", value);                                                                    \
    }
#define HDF5_READ(T)                                                                                                                                       \
    {                                                                                                                                                      \
        std::valarray< T > value;                                                                                                                          \
        iar >> alps::make_pvp("/vector/" + iar.encode_segment(#T) + "/scalar", value);                                                                     \
        std::cout << #T << "-scalar (" << value.size() << "): [";                                                                                          \
        for (std::size_t i = 0; i < 20;  ++i)                                                                                                              \
            std::cout << value[i] << ", ";                                                                                                                 \
        std::cout << "...]" << std::endl;                                                                                                                  \
    }                                                                                                                                                      \
    {                                                                                                                                                      \
        std::valarray< T > value;                                                                                                                          \
        iar >> alps::make_pvp("/vector/" + iar.encode_segment(#T) + "/null", value);                                                                       \
        std::cout << #T << "-null (" << value.size() << ")" << std::endl;                                                                                  \
    }                                                                                                                                                      \
    {                                                                                                                                                      \
        std::valarray<std::complex< T > > value;                                                                                                           \
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
        alps::hdf5::oarchive oar("valarray.h5");
        HDF5_FOREACH(HDF5_WRITE)
        {
            std::valarray<std::string> value("Im a Test string", 2);
            value[1] = "me 2";
            oar << alps::make_pvp("/vector/std::string/scalar", value);
        }
        {
            std::valarray<std::string> value;
            oar << alps::make_pvp("/vector/std::string/null", value);
        }
    }
    {
        alps::hdf5::iarchive iar("valarray.h5");
        HDF5_FOREACH(HDF5_READ)
        {
            std::valarray<std::string> value;
            iar >> alps::make_pvp("/vector/std::string/scalar", value);
            std::cout << "std::string-scalar (" << value.size() << "): [";
            for (std::size_t i = 0; i < value.size(); ++i)
                std::cout << value[i] << (i + 1 < value.size() ? ", " : "");
            std::cout << "]" << std::endl;
        }
        {
            std::valarray<std::string> value;
            iar >> alps::make_pvp("/vector/std::string/null", value);
            std::cout << "std::string-null (" << value.size() << ")" << std::endl;
        }
    }
    boost::filesystem::remove(boost::filesystem::path("valarray.h5"));
}
