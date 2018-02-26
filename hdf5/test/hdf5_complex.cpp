/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <complex>
#include <alps/hdf5/archive.hpp>
#include <alps/hdf5/complex.hpp>
#include <alps/hdf5/vector.hpp>
#include <vector>
#include <iostream>
#include <algorithm>

#include "gtest/gtest.h"

template<class T>
std::ostream& operator<<(std::ostream& os, std::vector<T> const & v)
{
    std::copy(v.begin(), v.end(), std::ostream_iterator<T>(os, " "));
    return os;
}


struct foo {
  
    std::complex<double> scalar;
    std::vector<std::complex<double> > vec;
    
    void load(alps::hdf5::archive & ar)
    {
        ar >> alps::make_pvp("scalar", scalar);
        ar >> alps::make_pvp("vector", vec);
    }
    void save(alps::hdf5::archive & ar) const
    {
        ar << alps::make_pvp("scalar", scalar);
        ar << alps::make_pvp("vector", vec);
    }
    
};
TEST(hdf5_complex, TestingIoOfComplexVars){
    foo b;
    b.scalar = std::complex<double>(3,4);
    b.vec = std::vector<std::complex<double> >(5, std::complex<double>(0,7));
    {
        alps::hdf5::archive ar("test_hdf5_complex.h5", "w");
        ar << alps::make_pvp("/test/foo", b);
    }
    
    // check
    {
        foo t_b;
        alps::hdf5::archive ar("test_hdf5_complex.h5", "r");
        ar >> alps::make_pvp("/test/foo", t_b);
        std::cout << "scalar (write): " << b.scalar << std::endl;
        std::cout << "scalar (read): " << t_b.scalar << std::endl;
        EXPECT_NEAR(std::abs(b.scalar-t_b.scalar), 0, 1e-12);
        std::cout << "vector (write): " << b.vec << std::endl;
        std::cout << "vector (read): " << t_b.vec << std::endl;
    }
    
}
