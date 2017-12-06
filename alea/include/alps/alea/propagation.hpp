/**
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <Eigen/Core>

#include <alps/alea/complex_op.hpp>

// TODO maybe a better way?
#include <alps/alea/mean.hpp>
#include <alps/alea/variance.hpp>
#include <alps/alea/covariance.hpp>
#include <alps/alea/autocorr.hpp>
#include <alps/alea/batch.hpp>

namespace alps { namespace alea {

/**
 * Do not perform error propagation.
 *
 * Given a transformation `f` and random sample `X`, just trasform the sample
 * mean `f(Mean[X])` and discard any error information.
 */
struct no_prop { };

/**
 * Perform linaralized error propagation by estimating the Jacobian.
 *
 * Given a transformation `f` and a random sample `X`, estimate the propagated
 * uncertainties by performing a Taylor series and keeping the linear term:
 *
 *     Cov[f(X)] = df/dX Cov[X] (df/dX)^T + O(d^2f/dx^2)
 *
 * where `df/dX` is the Jacobian of `f` at `X`, as estimated by finite
 * differences of `dx`.  This procedure is exact for linear transformations;
 * for non-linear transformation, it will introduce bias.
 *
 * @see alps::alea::jacobian
 */
struct linear_prop
{
    linear_prop() : dx_(0) { }

    linear_prop(double dx) : dx_(dx) { assert(dx >= 0); }

    double dx() const { return dx_; }

private:
    double dx_;
};

/**
 * Estimate propagated variance by sampling the prior.
 *
 * @warning Not implemented
 */
struct sampling_prop
{
    sampling_prop(size_t nsamples=1024) : nsamples_(nsamples) { }

    size_t nsamples() const { return nsamples_; }

private:
    size_t nsamples_;
};

/**
 * Perform Jackknife rebatching.
 *
 * Jackknife is a rebatching method, which can operate on any distribution and
 * exactly removes the bias in the transformed uncertainties up to order `1/N`,
 * where `N` is the sample size.
 *
 * @see alps::alea::jackknife
 */
struct jackknife_prop { };

/**
 * Perform non-parametric bootstrap rebatching.
 *
 * @warning Not implemented
 */
struct bootstrap_prop
{
    bootstrap_prop(size_t nsamples=1024) : nsamples_(nsamples) { }

    size_t nsamples() const { return nsamples_; }

private:
    size_t nsamples_;
};

/**
 * Given a function `f`, estimate its Jacobian `J[i,j] = df[i]/dx[j]`.
 *
 * Estimate the Jacobian of a transformation `f` at the point `x` by forward
 * differences:
 *
 *           J[i,j] ~= (f(x + dx e[j]) - f(x))[i] / dx;
 *
 * where `e[j]` denotes the `j`-th unit vector.  This procedure is exact for
 * linear transformations and biased otherwise.
 */
template <typename T>
typename eigen<T>::matrix jacobian(const transformer<T> &f, column<T> x, double dx);


/**
 * Perform Jackknife transformation to pseudovalues
 */
template <typename T>
batch_data<T> jackknife(const batch_data<T> &in, const transformer<T> &tf);

}}
