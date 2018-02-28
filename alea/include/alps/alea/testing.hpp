/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <alps/alea/core.hpp>
#include <alps/alea/util.hpp>

#include <boost/math/distributions/fisher_f.hpp>

namespace alps { namespace alea {
    class t2_result;
    template <typename T> struct diag_diffs;
}}

namespace alps { namespace alea {

/**
 * Perform Hotelling's T^2 test given a set of uncorrelated differences.
 *
 * Takes the difference `diff` between the two results under test, which must
 * be independently Gaussian distributed with variances `var`, and the number
 * of uncorrelated observations `nmeas`, and performs Hotelling's T^2 test on
 * it. `pools` is the number of stochastic results (1 or 2), and `atol` is
 * an upper tolerance for exact (zero variance) results.
 *
 * @see test_mean()
 */
template <typename T>
t2_result t2_test(const column<T> &diff,
                  const column<typename make_real<T>::type> &var,
                  double nmeas, size_t pools=1, double atol=1e-14
                  );

/** Diagonalizes the covariance matrix */
template <typename T>
diag_diffs<T> diagonalize_cov(const column<T> &diff,
                              const typename eigen<T>::matrix &cov);

/** Helper struct for diagonalize_cov() */
template <typename T>
struct diag_diffs
{
    column<T> diff;
    column<typename make_real<T>::type> var;
};

/**
 * Result of Hotelling's T^2 test
 *
 * @see test_mean(), t2_test()
 */
class t2_result
{
public:
    typedef boost::math::fisher_f_distribution<double> dist_type;

public:
    /** Initializes t2 test */
    t2_result(double score, double f_size, double f_dof)
        : score_(score), dist_(f_size, f_dof)
    { }

    double pvalue_lower() const { return cdf(dist_, score_); }

    double pvalue_upper() const { return cdf(complement(dist_, score_)); }

    double score() const { return score_; }

    const dist_type &dist() const { return dist_; }

private:
    double score_;
    dist_type dist_;
};


}} /* namespace alps::alea */
