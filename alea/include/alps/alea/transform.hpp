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
 *           J[i,j] ~= (f(x + dx e[j]) - f(x))[i] / dx;
 *
 * where `e[j]` denotes the `j`-th unit vector.  This procedure is exact for
 * linear transformations and biased otherwise.
 */
template <typename T>
typename eigen<T>::matrix jacobian(const transform<T> &f, column<T> x, double dx);


/**
 * Linear transformation mediated by a matrix.
 */
template <typename T>
struct linear_transform
    : public transform<T>
{
public:
    template <typename Derived>
    linear_transform(const Eigen::MatrixBase<Derived> &mat)
        : mat_(mat)
    { }

    size_t in_size() const { return mat_.rows(); }

    size_t out_size() const { return mat_.cols(); }

    column<T> operator() (const column<T> &in) const
    {
        // TODO figure this out
        return mat_ * typename eigen<T>::col(in);
    }

    bool is_linear() const { return true; }

private:
    typename eigen<T>::matrix mat_;
};

template <typename T>
mean_result<T> jac_transform(const transform<T> &f, const mean_result<T> &res)
{
    mean_result<T> tres(res);
    tres.store().data() = f(res.store().data());
    return tres;
}

}}
