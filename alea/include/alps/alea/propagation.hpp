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
 * Do not perform error propagation
 */
struct no_prop { };

/**
 * Perform linar error propagation by estimating the jacobian
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
 * Not implemented
 */
struct sampling_prop
{
    sampling_prop(size_t nsamples=1024) : nsamples_(nsamples) { }

    size_t nsamples() const { return nsamples_; }

private:
    size_t nsamples_;
};

/**
 * Not implemented
 */
struct jackknife_prop { };

/**
 * Not implemented
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
batch_data<T> jackknife(const batch_data<T> &in, transformer<T> &tf);

}}
