/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <alps/alea/core.hpp>
#include <alps/alea/computed.hpp>
#include <alps/alea/mean.hpp>
#include <alps/alea/propagation.hpp>
#include <alps/alea/convert.hpp>

#include <random>

namespace alps { namespace alea {

template <typename T, typename InResult>
mean_result<T> transform(const no_prop &, const transformer<T> &tf, const InResult &in)
{
    static_assert(traits<InResult>::HAVE_MEAN, "result does not have mean");
    static_assert(std::is_same<typename traits<InResult>::value_type, T>::value,
                  "Result and transform types are mismatched");

    if (tf.in_size() != in.size())
        throw size_mismatch();

    mean_result<T> res(tf.out_size());
    res.store().data() = tf(in.mean());
    res.store().count() = in.count();
    return res;
}

template <typename T, typename InResult>
var_result<T> transform(const linear_prop<false> &, const transformer<T> &tf, const InResult &in)
{
    static_assert(traits<InResult>::HAVE_MEAN, "result does not have mean");
    static_assert(traits<InResult>::HAVE_VAR, "result does not have variance");
    static_assert(std::is_same<typename traits<InResult>::value_type, T>::value,
                  "Result and transform types are mismatched");

    if (tf.in_size() != in.size())
        throw size_mismatch();

    var_result<T> res(tf.out_size());
    res.store().data() = tf(in.mean());
    res.store().count() = in.count();
    return res;
}


}}
