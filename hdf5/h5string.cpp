#include <alps/hdf5.hpp>
#include <alps/utility/encode.hpp>

#include <boost/filesystem.hpp>

#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

int main() {
    {
        alps::hdf5::oarchive oar("string.h5");
        std::vector<alps::uint8_t> vec(10, 2);
        oar << alps::make_pvp("/string/vector", vec);
        oar << alps::make_pvp("/string/c_str", "...");
    }
    {
        alps::hdf5::iarchive iar("string.h5");
    }
    boost::filesystem::remove(boost::filesystem::path("string.h5"));
}
