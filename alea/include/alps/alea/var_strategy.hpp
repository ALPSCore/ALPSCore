/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <Eigen/Core>

#include <alps/alea/complex_op.hpp>

// Forward declarations
namespace alps { namespace alea {
    struct circular_var;
    struct elliptic_var;
    template <typename Strategy, typename T> struct bind;
}}

// Actual declarations

namespace alps { namespace alea {

/**
 * Error understood with the circularity constraint.
 *
 * Given a complex sample mean `xbar`, the standard error of the mean `xerror`
 * is the real number defined such that roughly two thirds of the values should
 * fall in a circle of radius `xerror` around `xbar`.  The error in the real
 * case is defined in the usual sense.
 *
 * This is usually the error estimate you want for stochastic postprocessing.
 * In particular, if `z = U*x`, where `z` is a complex random vector,
 * `x` is a real random vector, and `U` is a unitary matrix (such as a Fourier
 * transform), the circular error on `z` coincides with the usual error on `x`.
 * However, plotting real or imaginary part with the circular error can be
 * misleading.  In this case, one should use `elliptic_var`.
 */
struct circular_var { };

/**
 * Error ellipse in the complex plane.
 *
 * Given a complex sample mean `xbar`, estimate the covariance matrix between
 * real and imaginary part (the covariance of the vector elements of `xbar` is
 * unaffected) and stores it as `complex_op xerror`.  This is equivalent to
 * defining an error ellipse satisfying:
 *
 *     Q(r, i) = xerror.rere() * r * r + 2 * xerror.reim() * r * i
 *               + xerror.imim() * i * i + xbar.real() * r + xbar.imag() * i,
 *
 * where roughly two thirds of the values shall fall into.  The error in the
 * real case is defined in the usual sense.
 *
 * These error ellipses can be useful when analyzing the real and imaginary
 * part seperately.  However, care has to be taken not to throw away errors
 * "hidden" in the cross-correlation between real and imaginary part.
 */
struct elliptic_var { };


template <typename T>
struct bind<circular_var, T>
{
    typedef T value_type;
    typedef T var_type;
    typedef T cov_type;

    typedef Eigen::internal::scalar_abs2_op<T> abs2_op;

    static cov_type outer(T x, T y) { return x * y; }
};

template <typename T>
struct bind<circular_var, std::complex<T> >
{
    typedef std::complex<T> value_type;
    typedef T var_type;
    typedef std::complex<T> cov_type;

    typedef Eigen::internal::scalar_abs2_op<std::complex<T> > abs2_op;

    static cov_type outer(std::complex<T> x, std::complex<T> y)
    { return x * std::conj(y); }
};

template <typename T>
struct bind<elliptic_var, T>
{
    typedef T value_type;
    typedef T var_type;
    typedef T cov_type;

    typedef Eigen::internal::scalar_abs2_op<T> abs2_op;

    static cov_type outer(T x, T y) { return x * y; }
};

template <typename T>
struct bind<elliptic_var, std::complex<T> >
{
    typedef std::complex<T> value_type;
    typedef complex_op<T> var_type;
    typedef complex_op<T> cov_type;

    struct abs2_op
    {
        typedef complex_op<T> result_type;
        const complex_op<T> operator() (const std::complex<T> &x) const
        { return complex_op<T>::outer(x, x); }
    };

    static complex_op<T> outer(std::complex<T> x, std::complex<T> y)
    { return complex_op<T>::outer(x, y); }
};


}} /* namespace alps::alea */

