/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <complex>
#include <algorithm>
#include <functional>
#include <cmath>
#include <type_traits>

#include <iosfwd>
#include <Eigen/Core>

// Forward declarations

namespace alps { namespace alea {
    template <typename T> class complex_op;
}}
namespace Eigen {
    template <typename T> struct NumTraits;
    template <typename S1, typename S2, typename Op> struct ScalarBinaryOpTraits;
}

// Actual declarations

namespace alps { namespace alea {

/**
 * General linear operation on a complex number.
 *
 * If one interprets a complex number `x` as column vector given by
 * `{ x.real(), x.imag() }`, the effect of an `complex_op A` is equivalent to
 * the left multiplication with the 2x2 matrix:
 *
 *     {{ A.rere(), A.reim() },
 *      { A.imre(), A.imim() }}
 *
 * (The elements are laid out in that way as well.) If one identifies the
 * imaginary part of the result with the prefactor of 'j', then `complex_op`
 * is equivalent to a quarternion `a + b*i + c*j + d*k`.
 *
 * Note that `dot(a, b)` must be used to multiply two complex_op instances and
 * `solve(a, b)` for division because multiplication is not commutative.
 */
template <typename T>
class complex_op
{
public:
    static_assert(std::is_floating_point<T>::value,
                  "T argument of complex_op<T> must be real scalar type");

    static complex_op outer(std::complex<T> a, std::complex<T> b)
    {
        return complex_op(a.real() * b.real(), a.real() * b.imag(),
                          a.imag() * b.real(), a.imag() * b.imag());
    }

    static complex_op diag(std::complex<T> a)
    {
        return complex_op(a.real(), 0, 0, a.imag());
    }

    static complex_op diag(T a) { return complex_op(a, 0, 0, a); }

public:
    /** Default constructed (uninitialized) */
    complex_op() { }

    /** Scaling transformation */
    complex_op(double x) : complex_op(x, 0, 0, x) { }

    /** Construct new operation */
    complex_op(T rere, T reim, T imre, T imim)
    {
        vals_[0][0] = rere;
        vals_[0][1] = reim;
        vals_[1][0] = imre;
        vals_[1][1] = imim;
    }

    T &rere() { return vals_[0][0]; }
    T &reim() { return vals_[0][1]; }
    T &imre() { return vals_[1][0]; }
    T &imim() { return vals_[1][1]; }

    const T &rere() const { return vals_[0][0]; }
    const T &reim() const { return vals_[0][1]; }
    const T &imre() const { return vals_[1][0]; }
    const T &imim() const { return vals_[1][1]; }

    complex_op &operator+=(complex_op x)
    {
        std::transform(&vals_[0][0], &vals_[2][0], &x.vals_[0][0],
                       &vals_[0][0], std::plus<T>());
        return *this;
    }

    complex_op &operator-=(complex_op x)
    {
        std::transform(&vals_[0][0], &vals_[2][0], &x.vals_[0][0],
                       &vals_[0][0], std::minus<T>());
        return *this;
    }

    complex_op &operator*=(double x)
    {
        vals_[0][0] *= x;
        vals_[0][1] *= x;
        vals_[1][0] *= x;
        vals_[1][1] *= x;
        return *this;
    }

    complex_op &operator/=(double x) { return *this *= 1/x; }

    friend complex_op operator-(complex_op x)
    {
        return complex_op(-x.rere(), -x.reim(), -x.imre(), -x.imim());
    }

    friend complex_op operator+(complex_op l, complex_op r)
    {
        return complex_op(l) += r;
    }

    friend complex_op operator-(complex_op l, complex_op r)
    {
        return complex_op(l) -= r;
    }

    friend complex_op operator*(complex_op x, double f)
    {
        return complex_op(x) *= f;
    }

    friend complex_op operator*(double f, complex_op x) { return x * f; }

    friend complex_op operator/(complex_op x, double f)
    {
        return complex_op(x) /= f;
    }

    friend complex_op dot(complex_op l, complex_op r)
    {
        // Matrix multiplication of two 2x2 matrices
        return complex_op(l.rere() * r.rere() + l.reim() * r.imre(),
                          l.rere() * r.reim() + l.reim() * r.imim(),
                          l.imre() * r.rere() + l.imim() * r.imre(),
                          l.imre() * r.reim() + l.imim() * r.imim());
    }

    friend complex_op solve(complex_op l, complex_op r)
    {
        return dot(l, inv(r));
    }

    friend bool operator==(complex_op l, complex_op r)
    {
        return std::equal(&l.vals_[0][0], &l.vals_[2][0], &r.vals_[0][0]);
    }

    friend bool operator!=(complex_op l, complex_op r)
    {
        return !std::equal(&l.vals_[0][0], &l.vals_[2][0], &r.vals_[0][0]);
    }

    friend complex_op inv(complex_op x)
    {
        // [a, b; c, d]^(-1) = 1/(ad - bc) * [d, -b; -c, a]
        T inv_det = 1/(x.rere() * x.imim() - x.reim() * x.imre());
        return complex_op(inv_det *  x.imim(), inv_det * -x.reim(),
                          inv_det * -x.imre(), inv_det *  x.rere());
    }

    friend complex_op abs2(complex_op x)
    {
        // |x|^2 = conj(x) * x = x.T * x
        T off_diag = x.rere() * x.reim() + x.imre() * x.imim();
        return complex_op(x.rere() * x.rere() + x.imre() * x.imre(),
                          off_diag, off_diag,
                          x.reim() * x.reim() + x.imim() * x.imim());
    }

    friend complex_op sqrt(complex_op x)
    {
        //  Levinger, Math. Mag. 53(4), 222-224, 1980
        T det = x.rere() * x.imim() - x.reim() * x.imre();
        T tr = x.rere() + x.imim();

        // ensure that the matrix is pos definite
        if (det < 0 || tr < 0)
            return complex_op(NAN, NAN, NAN, NAN);

        // choosing + here ensures that the result is also pos definite
        T s = std::sqrt(det);
        T inv_t = 1/std::sqrt(tr + 2 * s);
        return complex_op(inv_t * (x.rere() + s), inv_t * x.reim(),
                          inv_t * x.imre(), inv_t * (x.imim() + s));
    }

    friend bool isnan(complex_op x)
    {
        return std::any_of(&x.vals_[0][0], &x.vals_[2][0], [](T y) {return std::isnan(y); });
    }

    friend bool isfinite(complex_op x)
    {
        return std::all_of(&x.vals_[0][0], &x.vals_[2][0], [](T y) {return std::isfinite(y); });
    }

    friend bool isinf(complex_op x)
    {
        // An object cannot be inf and nan at the same time, for multi-
        // component objects nan takes precedence
        if (isnan(x))
            return false;

        return std::any_of(&x.vals_[0][0], &x.vals_[2][0], [](T y) {return std::isinf(y); });
    }

    friend complex_op abs(complex_op x) { return sqrt(abs2(x)); }

    friend std::ostream &operator<<(std::ostream &out, complex_op x)
    {
        out << '(' << x.rere() << ',' << x.reim()
            << ';' << x.imre() << ',' << x.imim() << ')';
        return out;
    }


private:
    T vals_[2][2];
};

}} /* namespace alps::alea */

namespace Eigen {

/**
 * Allows use of alps::alea::complex_op as scalar of Eigen matrices
 */
template <typename T>
struct NumTraits< alps::alea::complex_op<T> >
    : NumTraits<T>  // gets epsilon, dummy_precision, lowest, highest functions
{
    typedef alps::alea::complex_op<T> Real;
    typedef alps::alea::complex_op<T> NonInteger;
    typedef alps::alea::complex_op<T> Nested;

    enum {
        IsComplex = 0,  // Since typeof(real(x)) == typeof(x)
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = 1,
        AddCost = 3,
        MulCost = 5   // ~ (8 SMUL + 4 ADD)/4 with SMUL = 3
    };
};

template <typename T, typename BinaryOp>
struct ScalarBinaryOpTraits<alps::alea::complex_op<T>, T, BinaryOp>
{
    typedef alps::alea::complex_op<T> ReturnType;
};

template <typename T, typename BinaryOp>
struct ScalarBinaryOpTraits<T, alps::alea::complex_op<T>, BinaryOp>
{
    typedef alps::alea::complex_op<T> ReturnType;
};

} /* namespace Eigen */
