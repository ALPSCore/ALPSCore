/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/hdf5/archive.hpp>
#include <alps/utilities/encode.hpp>
#include <alps/alea.h>

#include <string>
#include <vector>
#include <iostream>
#include <algorithm>


int main() {
    alps::hdf5::archive iar("loadobs.h5");

    std::vector<std::string> list = iar.list_children("/simulation/results");
    for (std::vector<std::string>::const_iterator it = list.begin(); it != list.end(); ++it) {
        iar.set_context("/simulation/results/" + iar.encode_segment(*it));
        if (iar.is_scalar("/simulation/results/" + iar.encode_segment(*it) + "/mean/value")) {
            alps::alea::mcdata<double> obs;
            obs.load(iar);
            std::cout << *it << " " << obs << std::endl;
        } else {
            alps::alea::mcdata<std::vector<double> > obs;
            obs.load(iar);
            std::cout << *it << " " << obs << std::endl;
        }
    }
}
