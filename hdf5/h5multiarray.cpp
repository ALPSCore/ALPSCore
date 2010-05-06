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
    }
    {
        alps::hdf5::iarchive iar("string.h5");
    }
    boost::filesystem::remove(boost::filesystem::path("string.h5"));
}
