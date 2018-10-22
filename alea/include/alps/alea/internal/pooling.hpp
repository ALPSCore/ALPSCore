/**
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <alps/alea/core.hpp>
#include <alps/alea/util.hpp>

#include <alps/alea/variance.hpp>
#include <alps/alea/covariance.hpp>
#include <alps/alea/autocorr.hpp>

#include <alps/alea/internal/joined.hpp>

namespace alps { namespace alea { namespace internal {

/** Returns covariance, or construct from variances if not available */
template <typename T, typename Str>
const typename eigen<typename traits<cov_result<T,Str>>::cov_type>::matrix &
get_cov(const cov_result<T,Str> &result)
{
    return result.cov();
}

/** Returns covariance, or construct from variances if not available */
template <typename Result>
typename eigen<typename traits<Result>::cov_type>::matrix
get_cov(const Result &result)
{
    return result.cov();
}

/** Returns covariance, or construct from variances if not available */
template <typename T, typename Str>
typename eigen<typename traits<cov_result<T,Str>>::cov_type>::matrix
get_cov(const var_result<T,Str> &result)
{
    return result.var().asDiagonal();
}

/** Returns covariance, or construct from variances if not available */
template <typename T>
typename eigen<typename traits<cov_result<T>>::cov_type>::matrix
get_cov(const autocorr_result<T> &result)
{
    return result.var().asDiagonal();
}

template <typename Result, typename Derived>
using diff_scalar_type = add_scalar_type<
                            typename traits<Result>::value_type,
                            typename Eigen::internal::traits<Derived>::Scalar>;

/**
 * Get difference
 */
template <typename Result, typename Derived,
          typename T=diff_scalar_type<Result, Derived>>
var_result<T> make_diff(const Result &result,
                        const Eigen::MatrixBase<Derived> &expected)
{
    if ((int)result.size() != expected.size())
        throw size_mismatch();

    var_result<T> diff(var_data<T>(result.size()));

    // We do not copy the true values of count(), count2(), because they
    // are not consistent with stderr/variance for the autocorrelation acc.
    // TODO: come up with some grand scheme that corrects this.
    diff.store().count() = result.observations();
    diff.store().count2() = result.observations();
    diff.store().data() = result.mean() - expected;

    if (traits<Result>::HAVE_COV) {
        Eigen::SelfAdjointEigenSolver<typename eigen<T>::matrix> ecov(get_cov(result));
        diff.store().data() = diff.store().data() * ecov.eigenvectors();
        diff.store().data2() = ecov.eigenvalues();
    } else {
        diff.store().data2() = result.var();
    }
    return diff;
}

/**
 * Return result with pooled variance
 */
template <typename Result1, typename Result2,
          typename T=joined_value_type<Result1, Result2>>
var_result<T> pool_var(const Result1 &r1, const Result2 &r2)
{
    if (r1.size() != r2.size())
        throw size_mismatch();

    var_result<T> pooled(var_data<T>(r1.size()));

    // We pool the number of observations, which is the minimal version.
    // TODO: Ideally, we would like to do proper variance pooling for
    // weighted samples.
    double obs1 = r1.observations();
    double obs2 = r2.observations();
    pooled.store().count() = obs1 * obs2 / (obs1 + obs2);
    pooled.store().count2() = pooled.store().count();

    // The mean is just the difference
    pooled.store().data() = r1.mean() - r2.mean();

    if (traits<Result1>::HAVE_COV || traits<Result2>::HAVE_COV) {
        // Pooling covariance matrices - diagonalize those to yield variances
        auto pooled_cov = (obs1 - 1.) * get_cov(r1) + (obs2 - 1.) * get_cov(r2)
                          / (obs1 + obs2 - 2.0);

        Eigen::SelfAdjointEigenSolver<typename eigen<T>::matrix> ecov(pooled_cov);
        pooled.store().data() = pooled.store().data() * ecov.eigenvectors();
        pooled.store().data2() = ecov.eigenvalues();
    } else {
        // Directly pooling variances
        pooled.store().data2() = ((obs1 - 1.) * r1.var() + (obs2 - 1.) * r2.var())
                                 / (obs1 + obs2 - 2.0);
    }
    return pooled;
}

}}} /* namespace alps::alea::internal */
