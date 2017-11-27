/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <alps/alea/core.hpp>
#include <alps/alea/computed.hpp>

#include <alps/alea/mean.hpp>

#include <random>

namespace alps { namespace alea {

template <typename T, typename R>
mean_result<T> combine(mean_result<T> a, mean_result<T> b)
{
    mean_result<T> ret(a.size() + b.size());
    ret.store().data().topRows(a.size()) = a.store().data();
    ret.store().data().bottomRows(b.size()) = b.store().data();
    return ret;
}

/**
 * Given a function `f`, estimate its Jacobian `J[i,j] = df[i]/dx[j]`.
 *
 * Estimate the Jacobian of a transformation `f` at the point `x` by forward
 * differences:
 *
 *           J[i,j] ~= (f(x + eps[j] e[j]) - f(x))[i] / eps[j];
 *
 * where `e[j]` denotes the `j`-th unit vector.  This procedure is exact for
 * linear transformations and biased otherwise.
 */
template <typename T>
typename eigen<T>::matrix jacobian(const transform<T> &f,
                                   column<T> x, const column<T> &dx);


template <typename T>
mean_result<T> jac_transform(const transform<T> &f, const mean_result<T> &res)
{
    mean_result<T> tres(res);
    tres.store().data() = f(res.store().data());
    return tres;
}

}}
