/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include "gtest/gtest.h"
#include "alps/gf/grid.hpp"
#include "gf_test.hpp"

TEST(LogarithmicGrid, CheckRatios) {
  double tmax = 5, tmin = 0.001;
  double c = 2.0;
  int nfreq = 100;
  alps::gf::grid::logarithmic_real_frequency_grid grid(tmin, tmax, c, nfreq);
  std::vector<double> points;
  grid.compute_points(points);
  double ratio1 = (points[0] - points[1])/(points[1] - points[2]);
  double ratio2 = (points[2] - points[3])/(points[3] - points[4]);
  EXPECT_NEAR(ratio1, ratio2, 1e-10);
}

TEST(LogarithmicGrid, CheckBoundaries) {
    double tmax = 5, tmin = 0.001;
    double c = 2.0;
    int odd_nfreq = 101;
    int even_nfreq = 100;
    alps::gf::grid::logarithmic_real_frequency_grid odd_grid(tmin, tmax, c, odd_nfreq);
    alps::gf::grid::logarithmic_real_frequency_grid even_grid(tmin, tmax, c, even_nfreq);
    std::vector<double> points;
    odd_grid.compute_points(points);
    EXPECT_NEAR(c - points[0], points[points.size()-1] - c, 1e-10);
    even_grid.compute_points(points);
    EXPECT_NEAR(c - points[1], points[points.size()-1] - c, 1e-10);
}

TEST(LogarithmicGrid, CheckCenter) {
    double tmax = 5, tmin = 0.001;
    double c = 2.0;
    int odd_nfreq = 101;
    int even_nfreq = 100;
    alps::gf::grid::logarithmic_real_frequency_grid odd_grid(tmin, tmax, c, odd_nfreq);
    alps::gf::grid::logarithmic_real_frequency_grid even_grid(tmin, tmax, c, even_nfreq);
    std::vector<double> points;
    odd_grid.compute_points(points);
    EXPECT_NEAR(points[odd_nfreq/2], c, 1e-10);
    even_grid.compute_points(points);
    EXPECT_NEAR(points[even_nfreq/2], c, 1e-10);
}
TEST(LogarithmicGrid, CheckMinMax) {
    double tmax = 5, tmin = 0.001;
    double c = 2.0;
    int odd_nfreq = 101;
    int even_nfreq = 100;
    alps::gf::grid::logarithmic_real_frequency_grid odd_grid(tmin, tmax, c, odd_nfreq);
    alps::gf::grid::logarithmic_real_frequency_grid even_grid(tmin, tmax, c, even_nfreq);
    std::vector<double> points;
    odd_grid.compute_points(points);
    EXPECT_NEAR(c - points[odd_nfreq/2-1], tmin, 1e-10);
    EXPECT_NEAR(points[odd_nfreq/2+1] - c, tmin, 1e-10);
    EXPECT_NEAR(points[odd_nfreq-1] - points[odd_nfreq/2], tmax, 1e-10);
    EXPECT_NEAR(points[odd_nfreq/2] - points[0], tmax, 1e-10);
    even_grid.compute_points(points);
    EXPECT_NEAR(c - points[even_nfreq/2-1], tmin, 1e-10);
    EXPECT_NEAR(points[even_nfreq/2+1] - c, tmin, 1e-10);
    EXPECT_NEAR(points[even_nfreq/2] - points[0], tmax, 1e-10);
}

TEST(LinearGrid, CheckEquidistance) {
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

TEST(LinearGrid, CheckMinMax) {
    double Emin = -5;
    double Emax = 5;
    int nfreq = 23;
    alps::gf::grid::linear_real_frequency_grid grid(Emin, Emax, nfreq);
    std::vector<double> points;
    grid.compute_points(points);
    EXPECT_NEAR(Emin, points[0], 1e-10);
    EXPECT_NEAR(Emax, points[points.size()-1], 1e-10);
}