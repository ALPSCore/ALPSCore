/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#include <alps/alea/fisher_f.hpp>

#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <limits>

using alps::alea::fisher_f_distribution;

// d1, d2, f, P(F <= f), P(F > f); reference values from Boost.Math 1.88
static const double reference[][5] = {
    {1, 1, 0.5, 0.39182655203060723, 0.60817344796939277},
    {2, 5, 1.3, 0.64893217379942314, 0.35106782620057686},
    {4, 10, 0.2, 0.067348952213004909, 0.93265104778699515},
    {4, 10, 3.5, 0.95081185967506854, 0.04918814032493142},
    {10, 3, 2, 0.69062224553355767, 0.30937775446644245},
    {3, 100, 0.05, 0.014864882720599607, 0.9851351172794004},
    {7, 50, 12, 0.99999999274021389, 7.2597861147641365e-09},
    {20, 1000, 1.1, 0.65713157722744875, 0.34286842277255125},
    {5.5, 17.25, 0.8, 0.42571130035119681, 0.57428869964880325},
    {1.4217500000000001, 1.0270999999999999, 1758938688932100, 0.9999999899329578, 1.0067042170390192e-08},
    {0.76127299999999998, 0.97565599999999997, 4.1425378079875538e-16, 8.6213322622465496e-07, 0.99999913786677375},
    {12, 100000, 4, 0.99999686289237122, 3.1371076287823469e-06},
    {50, 200000, 1, 0.52659159250677434, 0.4734084074932256},
    {1, 2e+06, 30, 0.99999995679018883, 4.320981114086151e-08},
    {100000000, 100000000, 1.0001, 0.69145365940259818, 0.30854634059740177},
    {3, 500000000, 1.5, 0.78770971123500266, 0.21229028876499736},
    {800000000, 40, 0.90000000000000002, 0.28993750555193187, 0.71006249444806813},
};

// Documented relative accuracy (see fisher_f.hpp), by largest degree of freedom
static double tolerance(double d1, double d2)
{
    double dmax = std::max(d1, d2);
    return dmax <= 1e3 ? 1e-12 : dmax <= 1e6 ? 1e-9 : 1e-6;
}

TEST(fisher_f, reference)
{
    for (const auto &r : reference) {
        fisher_f_distribution dist(r[0], r[1]);
        double tol = tolerance(r[0], r[1]);
        EXPECT_NEAR(dist.cdf(r[2]), r[3], tol * r[3])
            << "d1=" << r[0] << " d2=" << r[1] << " f=" << r[2];
        EXPECT_NEAR(dist.ccdf(r[2]), r[4], tol * r[4])
            << "d1=" << r[0] << " d2=" << r[1] << " f=" << r[2];
    }
}

TEST(fisher_f, no_overflow)
{
    // Distribution is concentrated at f = 1 for huge degrees of freedom
    for (double d : {1e20, 1e200, 1e308}) {
        fisher_f_distribution dist(d, d);
        EXPECT_EQ(dist.cdf(2), 1) << "d=" << d;
        EXPECT_EQ(dist.ccdf(2), 0) << "d=" << d;
        EXPECT_EQ(dist.cdf(0.5), 0) << "d=" << d;
    }
    // f d1 overflows, but the result must still be finite and sensible
    fisher_f_distribution dist(1e10, 5);
    EXPECT_NEAR(dist.cdf(1e300), 1, 1e-12);

    // d1/d2 overflows although f d1/d2 is finite; Boost.Math: cdf ~ 0
    fisher_f_distribution tiny_d2(1, 1e-309);
    EXPECT_NEAR(tiny_d2.cdf(0.1), 0, 1e-12);
    EXPECT_NEAR(tiny_d2.ccdf(0.1), 1, 1e-12);
    fisher_f_distribution tiny_d1(1e-305, 3);
    EXPECT_NEAR(tiny_d1.cdf(1e5), 1, 1e-12);
    EXPECT_NEAR(tiny_d1.ccdf(1e5), 0, 1e-12);
}

TEST(fisher_f, edge_cases)
{
    const double inf = std::numeric_limits<double>::infinity();
    fisher_f_distribution dist(3, 7);
    EXPECT_EQ(dist.cdf(0), 0);
    EXPECT_EQ(dist.ccdf(0), 1);
    EXPECT_EQ(dist.cdf(-2), 0);
    EXPECT_EQ(dist.ccdf(-2), 1);
    EXPECT_EQ(dist.cdf(inf), 1);
    EXPECT_EQ(dist.ccdf(inf), 0);
    EXPECT_TRUE(std::isnan(dist.cdf(NAN)));
    EXPECT_TRUE(std::isnan(dist.ccdf(NAN)));
    EXPECT_TRUE(std::isnan(fisher_f_distribution(3, 0).cdf(1)));
    EXPECT_TRUE(std::isnan(fisher_f_distribution(-1, 5).ccdf(1)));
}
