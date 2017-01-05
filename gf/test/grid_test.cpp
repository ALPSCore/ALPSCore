/*
 * Copyright (C) 1998-2016 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include "gtest/gtest.h"
#include "alps/gf/grid.hpp"
#include "gf_test.hpp"

TEST(Grid, Logarithmic) {
  double tmax = 5, tmin = 0.001;
  double c = 2.0;
  int nfreq = 101;
  alps::gf::grid::logarithmic_real_frequency_grid grid(tmin, tmax, c, nfreq);
  std::vector<double> points;
  grid.compute_points(points);
  double ratio1 = (points[0] - points[1])/(points[1] - points[2]);
  double ratio2 = (points[2] - points[3])/(points[3] - points[4]);
  EXPECT_NEAR(ratio1, ratio2, 1e-10);
}

TEST(Grid, Linear) {
  double Emin = -5;
  double Emax = 5;
  int nfreq = 20;
  alps::gf::grid::linear_real_frequency_grid grid(Emin, Emax, nfreq);
  std::vector<double> points;
  grid.compute_points(points);
  double diff1 = points[0] - points[1];
  double diff2 = points[2] - points[3];
  EXPECT_NEAR(diff1, diff2, 1e-10);
  EXPECT_NEAR(Emin, points[0], 1e-10);
  EXPECT_NEAR(Emax, points[points.size()-1], 1e-10);
}
