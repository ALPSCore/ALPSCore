/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#pragma once

namespace alps { namespace alea {

/**
 * Fisher-Snedecor F distribution with `d1` and `d2` degrees of freedom.
 *
 * Only provides what Hotelling's T^2 test needs: the distribution function
 * and its complement, both accurate in their respective small tails.
 *
 * Relative accuracy (compared to Boost.Math) is about 1e-13 for degrees of
 * freedom up to 1e3, 1e-10 up to 1e6, and 1e-7 up to 1e9.  Arbitrarily large
 * parameters do not overflow; NaN is returned if the underlying continued
 * fraction fails to converge.
 */
class fisher_f_distribution
{
public:
    fisher_f_distribution(double d1, double d2) : d1_(d1), d2_(d2) { }

    double degrees_of_freedom1() const { return d1_; }

    double degrees_of_freedom2() const { return d2_; }

    /** Probability P(F <= f); NaN for invalid parameters */
    double cdf(double f) const;

    /** Probability P(F > f); NaN for invalid parameters */
    double ccdf(double f) const;

private:
    double d1_, d2_;
};

}} /* namespace alps::alea */
