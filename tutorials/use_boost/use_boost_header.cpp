/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <iostream>
// Use ALPSCore for something
#include <alps/config.hpp>
#include <alps/utilities/stringify.hpp>

// Let's use, e.g.,  boost::multi_array
#include <boost/multi_array.hpp>

int main()
{
    std::cout << "Using ALPSCore version " << ALPS_STRINGIFY(ALPSCORE_VERSION);
    // make a 2D array 3x4
    boost::multi_array<double,2> my_array(boost::extents[3][4]);
    std::cout << "\nSuccessfully created an array of dimension " << my_array.num_dimensions();
    for (auto i=0; i<my_array.num_dimensions(); ++i) {
        std::cout << "\nDimension " << i << " size is " << my_array.shape()[i];
    }
    endl(std::cout);
    return 0;
}
