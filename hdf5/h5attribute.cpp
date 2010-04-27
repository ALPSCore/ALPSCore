#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

#include <boost/filesystem.hpp>

#include <alps/hdf5.hpp>

#define HDF5_WRITE(T)                                                                                                                                      \
    {                                                                                                                                                      \
        oar.serialize("/attribute/" + oar.encode_segment(#T));                                                                                             \
        oar << alps::make_pvp("/attribute/" + oar.encode_segment(#T) + "/@value1", static_cast<T>(1));                                                     \
        oar << alps::make_pvp("/attribute/" + oar.encode_segment(#T) + "/@value2", static_cast<T>(-1));                                                    \
        oar << alps::make_pvp("/attribute/" + oar.encode_segment(#T) + "/@value3", static_cast<T>(2.85));                                                  \
        oar << alps::make_pvp("/attribute/" + oar.encode_segment(#T) + "/@value4", static_cast<T>(-38573.4));                                              \
        oar << alps::make_pvp("/attribute/" + oar.encode_segment(#T) + "/@value5", static_cast<T>(1));                                                     \
        oar << alps::make_pvp("/attribute/" + oar.encode_segment(#T) + "/@value6", static_cast<T>(0));                                                     \
    }                                                                                                                                                      \
    {                                                                                                                                                      \
        oar << alps::make_pvp("/attribute/" + oar.encode_segment(#T) + "/@complex1", std::complex<T>(1, 2));                                               \
        oar << alps::make_pvp("/attribute/" + oar.encode_segment(#T) + "/@complex2", std::complex<T>(static_cast<T>(-1), 2));                              \
        oar << alps::make_pvp("/attribute/" + oar.encode_segment(#T) + "/@complex3", std::complex<T>(static_cast<T>(1.2342), static_cast<T>(-2.93845)));   \
    }
#define HDF5_READ(T)                                                                                                                                       \
    {                                                                                                                                                      \
        std::cout << #T << "-values: ";                                                                                                                    \
        T value;                                                                                                                                           \
        iar >> alps::make_pvp("/attribute/" + iar.encode_segment(#T) + "/@value1", value);                                                                 \
        std::cout << value << ", ";                                                                                                                        \
        iar >> alps::make_pvp("/attribute/" + iar.encode_segment(#T) + "/@value2", value);                                                                 \
        std::cout << value << ", ";                                                                                                                        \
        iar >> alps::make_pvp("/attribute/" + iar.encode_segment(#T) + "/@value3", value);                                                                 \
        std::cout << value << ", ";                                                                                                                        \
        iar >> alps::make_pvp("/attribute/" + iar.encode_segment(#T) + "/@value4", value);                                                                 \
        std::cout << value << ", ";                                                                                                                        \
        iar >> alps::make_pvp("/attribute/" + iar.encode_segment(#T) + "/@value5", value);                                                                 \
        std::cout << value << ", ";                                                                                                                        \
        iar >> alps::make_pvp("/attribute/" + iar.encode_segment(#T) + "/@value6", value);                                                                 \
        std::cout << value << std::endl;                                                                                                                   \
    }                                                                                                                                                      \
    {                                                                                                                                                      \
        std::cout << #T << "-double: ";                                                                                                                    \
        long double value;                                                                                                                                 \
        iar >> alps::make_pvp("/attribute/" + iar.encode_segment(#T) + "/@value1", value);                                                                 \
        std::cout << value << ", ";                                                                                                                        \
        iar >> alps::make_pvp("/attribute/" + iar.encode_segment(#T) + "/@value2", value);                                                                 \
        std::cout << value << ", ";                                                                                                                        \
        iar >> alps::make_pvp("/attribute/" + iar.encode_segment(#T) + "/@value3", value);                                                                 \
        std::cout << value << ", ";                                                                                                                        \
        iar >> alps::make_pvp("/attribute/" + iar.encode_segment(#T) + "/@value4", value);                                                                 \
        std::cout << value << ", ";                                                                                                                        \
        iar >> alps::make_pvp("/attribute/" + iar.encode_segment(#T) + "/@value5", value);                                                                 \
        std::cout << value << ", ";                                                                                                                        \
        iar >> alps::make_pvp("/attribute/" + iar.encode_segment(#T) + "/@value6", value);                                                                 \
        std::cout << value << std::endl;                                                                                                                   \
    }                                                                                                                                                      \
    {                                                                                                                                                      \
        std::cout << #T << "-long: ";                                                                                                                      \
        long long value;                                                                                                                                   \
        iar >> alps::make_pvp("/attribute/" + iar.encode_segment(#T) + "/@value1", value);                                                                 \
        std::cout << value << ", ";                                                                                                                        \
        iar >> alps::make_pvp("/attribute/" + iar.encode_segment(#T) + "/@value2", value);                                                                 \
        std::cout << value << ", ";                                                                                                                        \
        iar >> alps::make_pvp("/attribute/" + iar.encode_segment(#T) + "/@value3", value);                                                                 \
        std::cout << value << ", ";                                                                                                                        \
        iar >> alps::make_pvp("/attribute/" + iar.encode_segment(#T) + "/@value4", value);                                                                 \
        std::cout << value << ", ";                                                                                                                        \
        iar >> alps::make_pvp("/attribute/" + iar.encode_segment(#T) + "/@value5", value);                                                                 \
        std::cout << value << ", ";                                                                                                                        \
        iar >> alps::make_pvp("/attribute/" + iar.encode_segment(#T) + "/@value6", value);                                                                 \
        std::cout << value << std::endl;                                                                                                                   \
    }                                                                                                                                                      \
    {                                                                                                                                                      \
        std::cout << #T << "-string: ";                                                                                                                    \
        std::string value;                                                                                                                                 \
        iar >> alps::make_pvp("/attribute/" + iar.encode_segment(#T) + "/@value1", value);                                                                 \
        std::cout << value << ", ";                                                                                                                        \
        iar >> alps::make_pvp("/attribute/" + iar.encode_segment(#T) + "/@value2", value);                                                                 \
        std::cout << value << ", ";                                                                                                                        \
        iar >> alps::make_pvp("/attribute/" + iar.encode_segment(#T) + "/@value3", value);                                                                 \
        std::cout << value << ", ";                                                                                                                        \
        iar >> alps::make_pvp("/attribute/" + iar.encode_segment(#T) + "/@value4", value);                                                                 \
        std::cout << value << ", ";                                                                                                                        \
        iar >> alps::make_pvp("/attribute/" + iar.encode_segment(#T) + "/@value5", value);                                                                 \
        std::cout << value << ", ";                                                                                                                        \
        iar >> alps::make_pvp("/attribute/" + iar.encode_segment(#T) + "/@value6", value);                                                                 \
        std::cout << value << std::endl;                                                                                                                   \
    }                                                                                                                                                      \
    {                                                                                                                                                      \
        std::cout << #T << "-complex: ";                                                                                                                   \
        std::complex<T> value;                                                                                                                             \
        iar >> alps::make_pvp("/attribute/" + iar.encode_segment(#T) + "/@complex1", value);                                                               \
        std::cout << "(" << value.real() << ", " << value.imag() << "), ";                                                                                 \
        iar >> alps::make_pvp("/attribute/" + iar.encode_segment(#T) + "/@complex2", value);                                                               \
        std::cout << "(" << value.real() << ", " << value.imag() << "), ";                                                                                 \
        iar >> alps::make_pvp("/attribute/" + iar.encode_segment(#T) + "/@complex3", value);                                                               \
        std::cout << "(" << value.real() << ", " << value.imag() << "), " << std::endl;                                                                    \
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
        alps::hdf5::oarchive oar("attribute.h5");
        HDF5_FOREACH(HDF5_WRITE)
        oar.serialize("/attribute/std::string");
        {
            std::string value = "Im a Test string";
            oar << alps::make_pvp("/attribute/std::string/@value", value);
        }
        {
            std::string value;
            oar << alps::make_pvp("/attribute/std::string/@empty", value);
        }
        oar.serialize("/attribute/c_string");
        {
            oar << alps::make_pvp("/attribute/c_string/@value", "me 2");
        }
        {
            std::string value;
            oar << alps::make_pvp("/attribute/c_string/@emtpy", "");
        }
    }
    {
        alps::hdf5::iarchive iar("attribute.h5");
        HDF5_FOREACH(HDF5_READ)
        {
            std::string value;
            iar >> alps::make_pvp("/attribute/std::string/@value", value);
            std::cout << "std::string-value (" << value.size() << "): " << value << std::endl;
        }
        {
            std::string value;
            iar >> alps::make_pvp("/attribute/std::string/@empty", value);
            std::cout << "std::string-empty (" << value.size() << "): " << value << std::endl;
        }
        {
            std::string value;
            iar >> alps::make_pvp("/attribute/c_string/@value", value);
            std::cout << "c_string-value (" << value.size() << "): " << value << std::endl;
        }
        {
            std::string value;
            iar >> alps::make_pvp("/attribute/c_string/@emtpy", value);
            std::cout << "c_string-empty (" << value.size() << ") " << value << std::endl;
        }
    }
    boost::filesystem::remove(boost::filesystem::path("attribute.h5"));
}
