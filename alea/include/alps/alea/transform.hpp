/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <alps/alea/core.hpp>
#include <alps/alea/computed.hpp>

#include <alps/alea/mean.hpp>
#include <alps/alea/variance.hpp>
#include <alps/alea/covariance.hpp>
#include <alps/alea/batch.hpp>

#include <alps/alea/propagation.hpp>
#include <alps/alea/convert.hpp>
#include <alps/alea/transformer.hpp> //FIXME

#include <random>
#include <type_traits>


namespace alps { namespace alea {

template <typename T, typename InResult>
mean_result<T> transform(no_prop, const transformer<T> &tf, const InResult &in)
{
    static_assert(traits<InResult>::HAVE_MEAN, "result does not have mean");
    static_assert(std::is_same<typename traits<InResult>::value_type, T>::value,
                  "Result and transform types are mismatched");

    if (tf.in_size() != in.size())
        throw size_mismatch();

    mean_result<T> res(mean_data<T>(tf.out_size()));
    res.store().data() = tf(in.mean());
    res.store().count() = in.count();
    return res;
}

// template mean_result<double> transform(no_prop, const transformer<double>&, const mean_result<double>&);

template <typename T, typename InResult>
typename std::enable_if<traits<InResult>::HAVE_COV, cov_result<T> >::type transform(linear_prop p, const transformer<T> &tf, const InResult &in)
{
    static_assert(traits<InResult>::HAVE_MEAN, "result does not have mean");
    static_assert(traits<InResult>::HAVE_COV, "result does not have covariance");
    static_assert(std::is_same<typename traits<InResult>::value_type, T>::value,
                  "Result and transform types are mismatched");

    if (tf.in_size() != in.size())
        throw size_mismatch();

    double dx = p.dx();
    if (dx == 0)
        dx = 0.125 * std::abs(in.stderror().mean());
    typename eigen<T>::matrix jac = jacobian(tf, in.mean(), dx);

    // TODO: this batch_size thing works but is conceptually hairy.
    double batch_size = in.count2() / in.count();
    cov_result<T> res(cov_data<T>(tf.out_size()));
    res.store().data() = tf(in.mean());
    res.store().data2() = jac * in.cov()/batch_size * jac.adjoint();
    res.store().count() = in.count();
    res.store().count2() = in.count2();
    return res;
}

// template cov_result<double> transform(linear_prop, const transformer<double>&, const cov_result<double>&);

template <typename T, typename InResult>
typename std::enable_if<!traits<InResult>::HAVE_COV, cov_result<T>>::type transform(linear_prop p, const transformer<T> &tf, const InResult &in)
{
    static_assert(traits<InResult>::HAVE_MEAN, "result does not have mean");
    static_assert(traits<InResult>::HAVE_VAR, "result does not have variance");
    static_assert(std::is_same<typename traits<InResult>::value_type, T>::value,
                  "Result and transform types are mismatched");

    if (tf.in_size() != in.size())
        throw size_mismatch();

    double dx = p.dx();
    if (dx == 0)
        dx = 0.125 * std::abs(in.stderror().mean());
    typename eigen<T>::matrix jac = jacobian(tf, in.mean(), dx);

    // TODO: this batch_size() thing is conceptually hairy.
    double batch_size = in.count2() / in.count();
    cov_result<T> res(cov_data<T>(tf.out_size()));
    res.store().data() = tf(in.mean());
    res.store().data2() = jac * in.var().asDiagonal()/batch_size  * jac.adjoint();
    res.store().count() = in.count();
    res.store().count2() = in.count2();
    return res;
}

// template cov_result<double> transform(linear_prop, const transformer<double>&, const var_result<double>&);

template <typename T>
batch_result<T> transform(jackknife_prop, const transformer<T> &tf, const batch_result<T> &in)
{
    if (tf.in_size() != in.size())
        throw size_mismatch();

    batch_result<T> res(jackknife(in.store(), tf));
    return res;
}

template batch_result<double> transform(jackknife_prop, const transformer<double>&, const batch_result<double>&);

}}
