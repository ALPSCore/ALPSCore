#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

#include <boost/filesystem.hpp>

#include <alps/hdf5.hpp>

#define HDF5_WRITE(T)                                                                                                                                      \
    {                                                                                                                                                      \
        std::vector< T > value;                                                                                                                            \
        for (std::size_t i = 0; i < 1000; ++i)                                                                                                             \
            value.push_back(i);                                                                                                                            \
            oar << alps::make_pvp("/vector/" + oar.encode_segment(#T) + "/scalar", value);                                                                 \
    }                                                                                                                                                      \
    {                                                                                                                                                      \
        std::vector< T > value;                                                                                                                            \
        oar << alps::make_pvp("/vector/" + oar.encode_segment(#T) + "/null", value);                                                                       \
    }                                                                                                                                                      \
    {                                                                                                                                                      \
        std::vector<std::complex< T > > value(1000);                                                                                                       \
        value[0] = std::complex<T>(1, 2);                                                                                                                  \
        value[1] = std::complex<T>(static_cast<T>(-1), 2);                                                                                                 \
        value[2] = std::complex<T>(static_cast<T>(1.2342), static_cast<T>(-2.93845));                                                                      \
        oar << alps::make_pvp("/vector/" + oar.encode_segment(#T) + "/complex", value);                                                                    \
    }
#define HDF5_READ(T)                                                                                                                                       \
    {                                                                                                                                                      \
        std::vector< T > value;                                                                                                                            \
        iar >> alps::make_pvp("/vector/" + iar.encode_segment(#T) + "/scalar", value);                                                                     \
        std::cout << #T << "-scalar (" << value.size() << "): [";                                                                                          \
        for (std::size_t i = 0; i < 20;  ++i)                                                                                                              \
            std::cout << value[i] << ", ";                                                                                                                 \
        std::cout << "...]" << std::endl;                                                                                                                  \
    }                                                                                                                                                      \
    {                                                                                                                                                      \
        std::vector< T > value;                                                                                                                            \
        iar >> alps::make_pvp("/vector/" + iar.encode_segment(#T) + "/null", value);                                                                       \
        std::cout << #T << "-null (" << value.size() << ")" << std::endl;                                                                                  \
    }                                                                                                                                                      \
    {                                                                                                                                                      \
        std::vector<std::complex< T > > value;                                                                                                             \
        iar >> alps::make_pvp("/vector/" + iar.encode_segment(#T) + "/complex", value);                                                                    \
        std::cout << #T << "-complex (" << value.size() << "): [";                                                                                         \
        for (std::size_t i = 0; i < 3;  ++i)                                                                                                               \
            std::cout << "(" << value[i].real() << ", " << value[i].imag() << "), ";                                                                       \
        std::cout << "...]" << std::endl;                                                                                                                  \
    }
#define HDF5_FOREACH(callback)                                                                                                                             \
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
	callback(boost::int8_t)                                                                                                                                       \
    callback(boost::uint8_t)                                                                                                                                      \
    callback(boost::int16_t)                                                                                                                                      \
    callback(boost::uint16_t)                                                                                                                                     \
    callback(boost::int32_t)                                                                                                                                      \
    callback(boost::uint32_t)                                                                                                                                     \
    callback(boost::int64_t)                                                                                                                                      \
    callback(boost::uint64_t)
int main() {
    {
        alps::hdf5::oarchive oar("vector.h5");
        HDF5_FOREACH(HDF5_WRITE)
        {
            std::vector<std::string> value(1, "Im a Test string");
            value.push_back("me 2");
            oar << alps::make_pvp("/vector/std::string/scalar", value);
        }
        {
            std::vector<std::string> value;
            oar << alps::make_pvp("/vector/std::string/null", value);
        }
    }
    {
        alps::hdf5::iarchive iar("vector.h5");
        HDF5_FOREACH(HDF5_READ)
        {
            std::vector<std::string> value;
            iar >> alps::make_pvp("/vector/std::string/scalar", value);
            std::cout << "std::string-scalar (" << value.size() << "): [";
            for (std::vector<std::string>::const_iterator it = value.begin(); it != value.end(); ++it)
                std::cout << *it << (it + 1 != value.end() ? ", " : "");
            std::cout << "]" << std::endl;
        }
        {
            std::vector<std::string> value;
            iar >> alps::make_pvp("/vector/std::string/null", value);
            std::cout << "std::string-null (" << value.size() << ")" << std::endl;
        }
    }
    boost::filesystem::remove(boost::filesystem::path("vector.h5"));
}
