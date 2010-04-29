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
        T value[length];                                                                                                                                   \
        for (std::size_t i = 0; i < length; ++i)                                                                                                           \
            value[i] = i;                                                                                                                                  \
        oar << alps::make_pvp("/ptr-1/" + oar.encode_segment(#T) + "/scalar", value, length);                                                              \
        oar << alps::make_pvp("/ptr-n/" + oar.encode_segment(#T) + "/scalar", value, size);                                                                \
    }                                                                                                                                                      \
    {                                                                                                                                                      \
        T* value=0;                                                                                                                                          \
        oar << alps::make_pvp("/ptr-1/" + oar.encode_segment(#T) + "/null", value, 0);                                                                     \
        oar << alps::make_pvp("/ptr-n/" + oar.encode_segment(#T) + "/null", value, std::vector<std::size_t>(3, 0));                                        \
    }                                                                                                                                                      \
    {                                                                                                                                                      \
        std::complex< T > value[length];                                                                                                                     \
        value[0] = std::complex<T>(1, 2);                                                                                                                  \
        value[1] = std::complex<T>(static_cast<T>(-1), 2);                                                                                                 \
        value[2] = std::complex<T>(static_cast<T>(1.2342), static_cast<T>(-2.93845));                                                                      \
        oar << alps::make_pvp("/ptr-1/" + oar.encode_segment(#T) + "/complex", value, length);                                                             \
        oar << alps::make_pvp("/ptr-n/" + oar.encode_segment(#T) + "/complex", value, size);                                                               \
    }
#define HDF5_READ(T)                                                                                                                                       \
    {                                                                                                                                                      \
        T value[length];                                                                                                                                   \
        iar >> alps::make_pvp("/ptr-1/" + iar.encode_segment(#T) + "/scalar", value, length);                                                              \
        std::cout << #T << "-1-scalar: [";                                                                                                                 \
        for (std::size_t i = 0; i < 20;  ++i)                                                                                                              \
            std::cout << value[i] << ", ";                                                                                                                 \
        std::cout << "...]" << std::endl;                                                                                                                  \
    }                                                                                                                                                      \
    {                                                                                                                                                      \
        T value[length];                                                                                                                                   \
        iar >> alps::make_pvp("/ptr-n/" + iar.encode_segment(#T) + "/scalar", value, size);                                                                \
        std::cout << #T << "-n-scalar: [";                                                                                                                 \
        for (std::size_t i = 0; i < 20;  ++i)                                                                                                              \
            std::cout << value[i] << ", ";                                                                                                                 \
        std::cout << "...]" << std::endl;                                                                                                                  \
    }                                                                                                                                                      \
    {                                                                                                                                                      \
        T * value=0;                                                                                                                                         \
        iar >> alps::make_pvp("/ptr-1/" + iar.encode_segment(#T) + "/null", value, 0);                                                                     \
        std::cout << #T << "-1-null" << std::endl;                                                                                                         \
    }                                                                                                                                                      \
    {                                                                                                                                                      \
        T * value=0;                                                                                                                                         \
        iar >> alps::make_pvp("/ptr-n/" + iar.encode_segment(#T) + "/null", value,  std::vector<std::size_t>(3, 0));                                       \
        std::cout << #T << "-n-null" << std::endl;                                                                                                         \
    }                                                                                                                                                      \
    {                                                                                                                                                      \
        std::complex< T > value[length];                                                                                                                   \
        iar >> alps::make_pvp("/ptr-1/" + iar.encode_segment(#T) + "/complex", value, length);                                                             \
        std::cout << #T << "-1-complex: [";                                                                                                                \
        for (std::size_t i = 0; i < 3;  ++i)                                                                                                               \
            std::cout << "(" << value[i].real() << ", " << value[i].imag() << "), ";                                                                       \
        std::cout << "...]" << std::endl;                                                                                                                  \
    }                                                                                                                                                      \
    {                                                                                                                                                      \
        std::complex< T > value[length];                                                                                                                   \
        iar >> alps::make_pvp("/ptr-n/" + iar.encode_segment(#T) + "/complex", value, size);                                                               \
        std::cout << #T << "-n-complex: [";                                                                                                                \
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

	try {
    std::vector<std::size_t> size(3, dim1);
    size[1] = dim2;
    size[2] = dim3;
    {
        alps::hdf5::oarchive oar("ptr.h5");
        HDF5_FOREACH(HDF5_WRITE)
        {
			std::vector<std::string> value(2);
            value[0] = "Im a Test string";
            value[1] = "me 2";
            oar << alps::make_pvp("/ptr/std::string/scalar", &value[0], 2);
        }
        {
            std::string * value=0;
            oar << alps::make_pvp("/ptr/std::string/null", value, 0);
        }
    }
    {
        alps::hdf5::iarchive iar("ptr.h5");
        HDF5_FOREACH(HDF5_READ)
        {
            std::string value[2];
            iar >> alps::make_pvp("/ptr/std::string/scalar", value, 2);
            std::cout << "std::string-scalar: [" << value[0] << ", " << value[1] << "]" << std::endl;
        }
        {
            std::string * value;
            iar >> alps::make_pvp("/ptr/std::string/null", value, 0);
            std::cout << "std::string-null" << std::endl;
        }
    }
    boost::filesystem::remove(boost::filesystem::path("ptr.h5"));
	} 
	catch (std::exception const& e)
	{
		std::cerr << e.what() << "\n";
		return -1;
	}
}
