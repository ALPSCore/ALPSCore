/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/hdf5/tensor.hpp>
#include <alps/hdf5/complex.hpp>
#include <boost/multi_array.hpp>

#include <alps/hdf5/multi_array.hpp>
#include <alps/hdf5/archive.hpp>
#include <alps/utilities/short_print.hpp>

#include <alps/testing/unique_file.hpp>
#include <vector>
#include <complex>
#include <iostream>

#include "gtest/gtest.h"

TEST(hdf5, TestingTensorBoost){
  alps::testing::unique_file ufile("real_complex_vec.h5.", alps::testing::unique_file::KEEP_AFTER);
  const std::string&  filename = ufile.name();

  boost::multi_array<std::complex<double>, 2> v(boost::extents[1][2]);
  boost::multi_array<double, 2> w(boost::extents[1][2]);
  {
    alps::hdf5::archive ar(filename, "w");
    ar["/vec"] << v;
    ar["/vec2"] << w;
  }
  {
    alps::hdf5::archive ar(filename);
    alps::hdf5::load(ar, "/vec", v);
  }
}


TEST(hdf5, TestingTensor){
  alps::testing::unique_file ufile("real_complex_vec.h5.", alps::testing::unique_file::KEEP_AFTER);
  const std::string&  filename = ufile.name();

  Eigen::MatrixXd M(Eigen::MatrixXd::Random(5, 6));

  alps::numerics::tensor<double, 2> v(5,6);
  for(int i = 0; i< 5; ++i){
    for (int j = 0; j < 6; ++j) {
      v(i, j) = M(i, j);
    }
  }
  {
    alps::hdf5::archive ar(filename, "w");
    ar["/vec"] << v;
//    v.save(ar, "/vec");
    alps::hdf5::save<double, 2>(ar, "/vec", v);
  }

  alps::numerics::tensor<double, 2> w(5,6);
  {
    alps::hdf5::archive ar(filename, "r");
    ar["/vec"] >> w;
//    alps::hdf5::load<double, 2>(ar, "/vec", w);
  }

  ASSERT_EQ(v.shape(), w.shape());

  for(int i = 0; i< 5; ++i) {
    for (int j = 0; j < 6; ++j) {
      EXPECT_TRUE(v(i,j) == w(i,j));
    }
  }
}

TEST(hdf5, TestingTensorComplex){
  alps::testing::unique_file ufile("real_complex_vec.h5.", alps::testing::unique_file::KEEP_AFTER);
  const std::string&  filename = ufile.name();

  Eigen::MatrixXd M(Eigen::MatrixXd::Random(5, 6));

  alps::numerics::tensor<std::complex<double>, 2> v(5,6);
  for(int i = 0; i< 5; ++i){
    for (int j = 0; j < 6; ++j) {
      v(i, j) = M(i, j);
    }
  }
  {
    alps::hdf5::archive ar(filename, "w");
    ar["/vec"] << v;
  }

  alps::numerics::tensor<std::complex<double>, 2> w(5,6);
  {
    alps::hdf5::archive ar(filename, "r");
    ar["/vec"] >> w;
  }

  ASSERT_EQ(v.shape(), w.shape());

  for(int i = 0; i< 5; ++i) {
    for (int j = 0; j < 6; ++j) {
      EXPECT_TRUE(v(i,j) == w(i,j));
    }
  }
}
