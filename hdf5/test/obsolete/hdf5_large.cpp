/*
 * Copyright (C) 1998-2015 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/hdf5/archive.hpp>
#include <alps/hdf5/vector.hpp>
#include <alps/utilities/cast.hpp>

#include <boost/filesystem.hpp>

#include <vector>

using namespace std;
using namespace alps;

int main () {

    for (std::size_t i = 0; i < 100; ++i)
        if (boost::filesystem::exists(boost::filesystem::path("large" + alps::cast<std::string>(i) + ".h5")))
            boost::filesystem::remove(boost::filesystem::path("large" + alps::cast<std::string>(i) + ".h5"));

    hdf5::archive ar("large%d.h5", "al");
    for (unsigned long long s = 1; s < (1ULL << 29); s <<= 1) {
        std::cout << s << std::endl;
        vector<double> vec(s, 10.);
        ar << make_pvp("/" + cast<std::string>(s), vec);
    }

    for (std::size_t i = 0; i < 100; ++i)
        if (boost::filesystem::exists(boost::filesystem::path("large" + alps::cast<std::string>(i) + ".h5")))
            boost::filesystem::remove(boost::filesystem::path("large" + alps::cast<std::string>(i) + ".h5"));
    return 0;
}
