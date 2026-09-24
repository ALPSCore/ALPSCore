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

const double nan = std::numeric_limits<double>::quiet_NaN();

// Continued fraction for the incomplete beta function (modified Lentz),
// converges quickly for x < (a + 1)/(a + b + 2).  Returns NaN if it does not
// converge, which happens only for extremely many degrees of freedom.
double beta_cf(double a, double b, double x)
{
    const double eps = 1e-15, tiny = 1e-300;
    double c = 1, d = 1 / std::fmax(std::abs(1 - (a + b) * x / (a + 1)), tiny);
    double h = d;
    for (int m = 1; m <= 100000; ++m) {
        for (int odd = 0; odd != 2; ++odd) {
            // written as products of ratios to avoid overflow
            double coeff = odd
                ? -(a + m) / (a + 2*m) * ((a + b + m) / (a + 2*m + 1)) * x
                : m / (a + 2*m - 1) * ((b - m) / (a + 2*m)) * x;
            d = 1 + coeff * d;
            d = 1 / (std::abs(d) < tiny ? tiny : d);
            c = 1 + coeff / c;
            c = std::abs(c) < tiny ? tiny : c;
            h *= d * c;
        }
        if (std::abs(d * c - 1) < eps)
            return h;
    }
    return nan;
}

// Remainder of Stirling's series, lgamma(x) - [(x - 1/2) log x - x + log(2 pi)/2],
// accurate to double precision for x >= 15
double stirling_corr(double x)
{
    double x2 = 1 / (x * x);
    return (1./12 - x2 * (1./360 - x2 * (1./1260 - x2 / 1680))) / x;
}

// log of the beta function B(a, b), avoiding the cancellation between large
// lgamma values
double log_beta(double a, double b)
{
    double p = std::fmin(a, b), q = std::fmax(a, b);
    if (q < 15)
        return std::lgamma(p) + std::lgamma(q) - std::lgamma(p + q);
    double corr = stirling_corr(q) - stirling_corr(p + q);
    double r = p / (p + q);
    if (p < 15)
        return std::lgamma(p) + corr + p - p * std::log(p + q)
               + (q - 0.5) * std::log1p(-r);
    return 0.91893853320467274178 /* log(2 pi)/2 */ - 0.5 * std::log(q) + stirling_corr(p) + corr
           + (p - 0.5) * std::log(r) + q * std::log1p(-r);
}

// Regularized incomplete beta function I_x(a, b), with y = 1 - x and the
// logarithms of x and y passed separately to avoid cancellation (x or y may
// underflow to zero while their logarithms are still finite).
double beta_inc(double a, double b, double x, double y, double log_x, double log_y)
{
    if (std::isinf(log_x)) return 0;
    if (std::isinf(log_y)) return 1;
    double front = std::exp(a * log_x + b * log_y - log_beta(a, b));
    double result = x < (a + 1) / (a + b + 2)
        ? (front == 0 ? 0 : front * beta_cf(a, b, x) / a)
        : (front == 0 ? 1 : 1 - front * beta_cf(b, a, y) / b);
    return std::fmin(std::fmax(result, 0.), 1.);
}

// P(F <= f) or P(F > f) for f > 0, in terms of q = f d1/d2:
// x = q/(1 + q), y = 1/(1 + q).
double f_tail(double d1, double d2, double f, bool upper)
{
    double q = d1 / d2 * f;
    double log_q = std::isnormal(q) ? std::log(q)
                   // d1/d2 over- or underflowed: rescale in log space
                   : std::log(d1) - std::log(d2) + std::log(f);
    double x, y, log_x, log_y;
    if (log_q > 0) {
        double inv_q = std::isnormal(q) ? 1 / q : std::exp(-log_q);
        x = 1 / (1 + inv_q);
        y = inv_q / (1 + inv_q);
        log_x = -std::log1p(inv_q);
        log_y = -log_q + log_x;
    } else {
        if (!std::isnormal(q))
            q = std::exp(log_q);
        x = q / (1 + q);
        y = 1 / (1 + q);
        log_y = -std::log1p(q);
        log_x = log_q + log_y;
    }
    return upper ? beta_inc(d2 / 2, d1 / 2, y, x, log_y, log_x)
                 : beta_inc(d1 / 2, d2 / 2, x, y, log_x, log_y);
}

}

double fisher_f_distribution::cdf(double f) const
{
    if (!(d1_ > 0 && d2_ > 0) || std::isnan(f))
        return nan;
    if (f <= 0)
        return 0;
    return f_tail(d1_, d2_, f, false);
}

double fisher_f_distribution::ccdf(double f) const
{
    if (!(d1_ > 0 && d2_ > 0) || std::isnan(f))
        return nan;
    if (f <= 0)
        return 1;
    return f_tail(d1_, d2_, f, true);
}

}} /* namespace alps::alea */
