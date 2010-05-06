#include <alps/hdf5.hpp>
#include <alps/utility/encode.hpp>

#include <boost/filesystem.hpp>

#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

int main() {
    {
        alps::hdf5::oarchive oar("carray.h5");
    }
    {
        alps::hdf5::iarchive iar("carray.h5");
    }
    boost::filesystem::remove(boost::filesystem::path("carray.h5"));
}
