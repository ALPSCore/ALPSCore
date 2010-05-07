#include <alps/hdf5.hpp>
#include <alps/utility/encode.hpp>

#include <boost/filesystem.hpp>

#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

int main() {
    {
        alps::hdf5::oarchive oar("overwrite.h5");
    }
    {
        alps::hdf5::iarchive iar("overwrite.h5");
    }
    boost::filesystem::remove(boost::filesystem::path("overwrite.h5"));
}