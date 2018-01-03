/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <vector>
#include <complex>

#include <alps/hdf5.hpp>
#include <alps/hdf5/vector.hpp>
#include "gtest/gtest.h"

TEST(hdf5, TestingIoOfPair){
    alps::hdf5::archive ar("creal.h5", "a");
    {
        std::vector<double> a(1e6);
        ar << alps::make_pvp("a",
            std::make_pair(
                static_cast< double const *>(&a.front())
                , std::vector<std::size_t>(1,a.size())
            )
        );
    }
    {
        std::vector<std::complex<double> > a(1e6);
        ar << alps::make_pvp("a",
            std::make_pair(
                static_cast<std::complex<double> const *>(&a.front())
                , std::vector<std::size_t>(1,a.size())
            )
        );
    }
}
