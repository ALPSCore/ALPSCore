#include <string>
#include <iostream>

#include <alps/hdf5.hpp>

int main() {
       alps::hdf5::iarchive iar("classical1a.task3.out.run1.h5");
       {
           std::string value;
           iar >> alps::make_pvp("/parameters/MODEL", value);
           std::cout << "std::string-value (" << value.size() << "): " << value << std::endl;
       }
}

/*
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
*/
