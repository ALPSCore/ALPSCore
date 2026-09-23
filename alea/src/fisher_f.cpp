/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/alea/fisher_f.hpp>

#include <cmath>
#include <limits>

namespace alps { namespace alea {

namespace {

// Continued fraction for the incomplete beta function (modified Lentz),
// converges quickly for x < (a + 1)/(a + b + 2).
double beta_cf(double a, double b, double x)
{
    const double eps = 1e-16, tiny = 1e-300;
    double c = 1, d = 1 / std::fmax(std::abs(1 - (a + b) * x / (a + 1)), tiny);
    double h = d;
    for (int m = 1; m <= 100000; ++m) {
        for (int odd = 0; odd != 2; ++odd) {
            double coeff = odd
                ? -(a + m) * (a + b + m) * x / ((a + 2*m) * (a + 2*m + 1))
                : m * (b - m) * x / ((a + 2*m - 1) * (a + 2*m));
            d = 1 + coeff * d;
            d = 1 / (std::abs(d) < tiny ? tiny : d);
            c = 1 + coeff / c;
            c = std::abs(c) < tiny ? tiny : c;
            h *= d * c;
        }
        if (std::abs(d * c - 1) < eps)
            break;
    }
    return h;
}

// Regularized incomplete beta function I_x(a, b), where y = 1 - x is passed
// separately to avoid cancellation.
double beta_inc(double a, double b, double x, double y)
{
    if (x <= 0) return 0;
    if (y <= 0) return 1;
    double front = std::exp(std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b)
                            + a * std::log(x) + b * std::log(y));
    if (x < (a + 1) / (a + b + 2))
        return front * beta_cf(a, b, x) / a;
    return 1 - front * beta_cf(b, a, y) / b;
}

}

double fisher_f_distribution::cdf(double f) const
{
    if (!(d1_ > 0 && d2_ > 0) || std::isnan(f))
        return std::numeric_limits<double>::quiet_NaN();
    if (f <= 0)
        return 0;
    if (std::isinf(f))
        return 1;
    double denom = d1_ * f + d2_;
    return beta_inc(d1_ / 2, d2_ / 2, d1_ * f / denom, d2_ / denom);
}

double fisher_f_distribution::ccdf(double f) const
{
    if (!(d1_ > 0 && d2_ > 0) || std::isnan(f))
        return std::numeric_limits<double>::quiet_NaN();
    if (f <= 0)
        return 1;
    if (std::isinf(f))
        return 0;
    double denom = d1_ * f + d2_;
    return beta_inc(d2_ / 2, d1_ / 2, d2_ / denom, d1_ * f / denom);
}

}} /* namespace alps::alea */
