/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <Eigen/Core>

#include <alps/alea/complex_op.hpp>
#include <alps/alea/var_strategy.hpp>

// Forward declarations

namespace alps { namespace alea { namespace internal {
    template <typename Str, typename Arg1, typename Arg2> class outer_expr;
}}}

// Actual declarations

namespace alps { namespace alea { namespace internal {

/**
 * Perform outer product in the sense of a variance for eigen vectors.
 *
 * Given two Eigen vectors `x,y` of type `Str::value_type`, construct a
 * Matrix-valued eigen expression of type `Str::cov_type`, where the
 * `(i,j)`-th element is given by `Str::outer(x,y)`.
 */
template <class Str, typename Arg1, typename Arg2>
outer_expr<Str, Arg1, Arg2> outer(const Eigen::MatrixBase<Arg1> &arg1,
                                  const Eigen::MatrixBase<Arg2> &arg2)
{
    return outer_expr<Str, Arg1, Arg2>(arg1.derived(), arg2.derived());
}

/** Eigen expression class corresponding to `outer(x,y)`. */
template <typename Str, typename Arg1, typename Arg2>
class outer_expr
    : public Eigen::MatrixBase< outer_expr<Str, Arg1, Arg2> >
{
public:
    typedef typename Eigen::internal::ref_selector<outer_expr>::type Nested;
    typedef Eigen::Index Index;
    typedef typename Eigen::internal::ref_selector<Arg1>::type Arg1Nested;
    typedef typename Eigen::internal::ref_selector<Arg2>::type Arg2Nested;

    outer_expr(const Arg1 &arg1, const Arg2 &arg2)
        : arg1_(arg1), arg2_(arg2)
    {
        EIGEN_STATIC_ASSERT(Arg1::ColsAtCompileTime == 1,
                            YOU_TRIED_CALLING_A_VECTOR_METHOD_ON_A_MATRIX);
        EIGEN_STATIC_ASSERT(Arg2::ColsAtCompileTime == 1,
                            YOU_TRIED_CALLING_A_VECTOR_METHOD_ON_A_MATRIX);
    }
    Index rows() const { return arg1_.rows(); }
    Index cols() const { return arg2_.rows(); }

    Arg1Nested arg1_;
    Arg2Nested arg2_;
};

}}}

namespace Eigen { namespace internal {

using alps::alea::internal::outer_expr;

template <typename Str, typename Arg1, typename Arg2>
struct traits<outer_expr<Str, Arg1, Arg2> >
{
    typedef Eigen::Dense StorageKind;
    typedef Eigen::MatrixXpr XprKind;
    typedef typename Arg1::StorageIndex StorageIndex;
    typedef typename Str::cov_type Scalar;
    typedef typename Str::cov_type CoeffReturnType;

    enum {
        Flags = Eigen::ColMajor,
        RowsAtCompileTime = Arg1::RowsAtCompileTime,
        ColsAtCompileTime = Arg2::RowsAtCompileTime,
        MaxRowsAtCompileTime = Arg1::MaxRowsAtCompileTime,
        MaxColsAtCompileTime = Arg2::MaxRowsAtCompileTime
    };
};

template <typename Str, typename Arg1, typename Arg2>
struct evaluator<outer_expr<Str, Arg1, Arg2> >
    : evaluator_base<outer_expr<Str, Arg1, Arg2> >
{
    typedef outer_expr<Str, Arg1, Arg2> XprType;

    // determines the true type argument for the evaluated arguments:
    // nested_eval<...>::type translates to either a temporary or a on-the-fly
    // evaluated object, while remove_all<...> removes const/ref qualifiers
    typedef typename nested_eval<Arg1, XprType::ColsAtCompileTime>::type Arg1Nested;
    typedef typename nested_eval<Arg2, XprType::ColsAtCompileTime>::type Arg2Nested;
    typedef typename remove_all<Arg1Nested>::type Arg1NestedCleaned;
    typedef typename remove_all<Arg2Nested>::type Arg2NestedCleaned;

    typedef typename XprType::CoeffReturnType CoeffReturnType;
    enum {
        CoeffReadCost = evaluator<Arg1NestedCleaned>::CoeffReadCost,
        Flags = Eigen::ColMajor
    };

    evaluator(const XprType& xpr)
        : arg1_impl_(xpr.arg1_)
        , arg2_impl_(xpr.arg2_)
        , rows_(xpr.rows())
        , cols_(xpr.cols())
    { }

    CoeffReturnType coeff(Index i, Index j) const
    {
        assert(i >= 0 && i < rows_);
        assert(j >= 0 && j < cols_);
        return Str::outer(arg1_impl_.coeff(i), arg2_impl_.coeff(j));
    }

private:
    evaluator<Arg1NestedCleaned> arg1_impl_;
    evaluator<Arg2NestedCleaned> arg2_impl_;
    const Index rows_, cols_;
};

}} /* namespace Eigen::internal */
