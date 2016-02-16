/*
 * Copyright (C) 1998-2016 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/hdf5/archive.hpp>
#include <alps/hdf5/vector.hpp>
#include <alps/hdf5/complex.hpp>

#include <boost/filesystem.hpp>

#include <vector>
#include <complex>

#include "gtest/gtest.h"

using namespace std;

TEST(hdf5, TestingIoOfComplexVectors){
    if (boost::filesystem::exists(boost::filesystem::path("vvcplx.h5")))
        boost::filesystem::remove(boost::filesystem::path("vvcplx.h5"));
	{
      vector< vector< complex<double> > > v;
      for( int i = 0; i < 3; ++i )
        v.push_back(vector< complex<double> >(i+1, complex<double>(i,2*i)));
      alps::hdf5::archive ar("vvcplx.h5",alps::hdf5::archive::WRITE);
      ar << alps::make_pvp("v",v);
	}
    boost::filesystem::remove(boost::filesystem::path("vvcplx.h5"));
}

