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
struct no_propagation { };

/**
 * Perform linar error propagation by estimating the jacobian
 */
struct linear_var_propagation { };

/**
 * Perform linar error propagation by estimating the jacobian
 */
struct linear_propagation { };  // TODO

/**
 * Not implemented
 */
struct jackknife_propagation { };  // TODO

/**
 * Not implemented
 */
struct sampling_propagation { };   // TODO


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
typename eigen<T>::matrix jacobian(const transform<T> &f, column<T> x, double dx);

/**
 * Perform Jackknife transformation to pseudovalues
 */
template <typename T>
batch_data<T> jackknife(const batch_data<T> &in, transform<T> &tf);


template <typename InResult>
struct bind<no_propagation, InResult>
{
    typedef typename traits<InResult>::value_type value_type;

    typedef InResult in_result_type;
    typedef mean_result<value_type> out_result_type;
    typedef transform<value_type> transform_type;

    out_result_type operator() (const transform_type &tf, const in_result_type &in)
    {
        if (tf.in_size() != in.size())
            throw size_mismatch();

        out_result_type res(tf.out_size());
        res.store().data() = tf(in.mean());
        res.store().count() = in.count();
        return res;
    }
};

}}
