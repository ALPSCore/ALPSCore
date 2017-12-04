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

// TODO: make this more general

template <typename T>
struct scalar_unary_transform
    : public transform<T>
{
public:
    scalar_unary_transform(const std::function<T(T)> &fn) : fn_(fn) { }

    size_t in_size() const { return 1; }

    size_t out_size() const { return 1; }

    column<T> operator() (const column<T> &in) const
    {
        if (in.size() != in_size())
            throw size_mismatch();

        column<T> ret(1);
        ret(0) = fn_(in(0));
        return ret;
    }

private:
    std::function<T(T)> fn_;
};

template <typename T>
struct scalar_binary_transform
    : public transform<T>
{
public:
    scalar_binary_transform(const std::function<T(T,T)> &fn) : fn_(fn) { }

    size_t in_size() const { return 2; }

    size_t out_size() const { return 1; }

    column<T> operator() (const column<T> &in) const
    {
        if (in.size() != in_size())
            throw size_mismatch();

        column<T> ret(1);
        ret(0) = fn_(in(0), in(1));
        return ret;
    }

private:
    std::function<T(T,T)> fn_;
};

template <typename T>
scalar_unary_transform<T> make_scalar_transform(const std::function<T(T)> &fn)
{
    return scalar_unary_transform<T>(fn);
}

template <typename T>
scalar_unary_transform<T> make_scalar_transform(const std::function<T(T,T)> &fn)
{
    return scalar_binary_transform<T>(fn);
}

// template <typename T, typename Result, typename Str=no_propagation>
// typename bind<
//
//
// template <typename Result, typename Str=no_propagation>
// typename bind<Str,Result>::out_result_type do_transform(
//                             const typename bind<Str,Result>::transform_type &t,
//                             const Result &in)
// {
//     return bind<Str,Result>(t, in);
// }

}}
