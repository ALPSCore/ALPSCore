/*
 * Copyright (C) 1998-2015 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <iostream>
#include <sstream>
#include <vector>
#include <alps/hdf5/archive.hpp>
#include <alps/hdf5/vector.hpp>

int main() {

    for (int i=0; i<100; ++i) {
        std::vector<double> vec(10,2.);
        alps::hdf5::archive ar("test_hdf5_valgrind.h5", "w");
        std::ostringstream ss;
        ss << "/vec" << i;
        ar << alps::make_pvp(ss.str(), vec);
    }
    return 0;
}
