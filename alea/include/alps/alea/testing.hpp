/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <alps/alea/core.hpp>
#include <alps/alea/util.hpp>
#include <alps/alea/variance.hpp>
#include <alps/alea/covariance.hpp>

#include <alps/alea/internal/joined.hpp>
#include <alps/alea/internal/pooling.hpp>

#include <boost/math/distributions/fisher_f.hpp>

namespace alps { namespace alea {
    class t2_result;
    template <typename T> struct diag_diffs;

    template <typename R1, typename R2, typename T>
    var_result<T> pooled_var(const R1 &, const R2 &);

    template <typename R1, typename R2, typename T>
    cov_result<T> pooled_cov(const R1 &, const R2 &);
}}

namespace alps { namespace alea {

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

    /** Returns whether the lower alternate hypothesis can be tested.
     *
     * For low values of size, the F distribution does not have a mode,
     * because there is no way to distinguish the lower alternate hypothesis
     * (too large error bars) from an accidental "hit".
     */
    bool has_plower() const { return dist_.degrees_of_freedom1() > 3; }

    /** p-value in favour of the lower alternate Hypothesis */
    double pvalue_lower() const { return has_plower() ? cdf(dist_, score_) : 1; }

    /** p-value in favour of the upper alternate Hypothesis */
    double pvalue_upper() const { return cdf(complement(dist_, score_)); }

    /** lowest p-value */
    double pvalue() const { return std::min(pvalue_lower(), pvalue_upper()); }

    /** t2 statistic and argument to `dist()` */
    double score() const { return score_; }

    /** distribution */
    const dist_type &dist() const { return dist_; }

private:
    double score_;
    dist_type dist_;
};

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

/**
 * Test mean of stochastic result `result` against known result `expected`.
 */
template <typename Result, typename Derived>
t2_result test_mean(const Result &result,
                    const Eigen::MatrixBase<Derived> &expected,
                    double atol=1e-14)
{
    static_assert(is_alea_result<Result>::value, "Result is not alea result");
    static_assert(traits<Result>::HAVE_VAR, "Result1 must have variance");

    using diff_scalar = internal::diff_scalar_type<Result, Derived>;
    var_result<diff_scalar> diff = internal::make_diff(result, expected);
    return t2_test(diff.mean(), diff.var(), diff.observations(), 1, atol);
}

/**
 * Test mean of stochastic result `result` against known result `expected`.
 */
template <typename Result, typename Derived>
t2_result test_mean(const Eigen::MatrixBase<Derived> &expected,
                    const Result &result,
                    double atol=1e-14)
{
    return test_mean(result, expected, atol);
}

/**
 * Test mean of two stochastic results against each other.
 */
template <typename Result1, typename Result2>
typename std::enable_if<
    is_alea_result<Result1>::value && is_alea_result<Result2>::value,
    t2_result>::type
test_mean(const Result1 &result1, const Result2 &result2, double atol=1e-14)
{
    static_assert(traits<Result1>::HAVE_VAR, "Result1 must have variance");
    static_assert(traits<Result2>::HAVE_VAR, "Result2 must have variance");

    using diff_scalar = internal::joined_value_type<Result1, Result2>;
    var_result<diff_scalar> diff = internal::pool_var(result1, result2);
    return t2_test(diff.mean(), diff.var(), diff.observations(), 2, atol);
}


}} /* namespace alps::alea */
