/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include "gtest/gtest.h"

#include <Eigen/Core>
#include <type_traits>

namespace alps { namespace testing {

template <typename T>
struct is_scalar : public std::false_type { };

template <typename T>
struct is_scalar< std::complex<T> > : public is_scalar<T> { };

template <> struct is_scalar<float> : public std::true_type { };
template <> struct is_scalar<double> : public std::true_type { };
template <> struct is_scalar<long double> : public std::true_type { };

/** Helper function for implementing ALPS_EXPECT_NEAR. */
template <typename T,
          typename std::enable_if<is_scalar<T>::value,int>::type = 0>
::testing::AssertionResult NearPredFormat(const char* expr1,
                                          const char* expr2,
                                          const char* abs_error_expr,
                                          const T &val1,
                                          const T &val2,
                                          double abs_error)
{
    const double diff = std::abs(val1 - val2);
    if (diff <= abs_error)
        return ::testing::AssertionSuccess();

    return ::testing::AssertionFailure()
        << "The difference between " << expr1 << " and " << expr2
        << " is " << diff << ", which exceeds " << abs_error_expr << ", where\n"
        << expr1 << " evaluates to " << val1 << ",\n"
        << expr2 << " evaluates to " << val2 << ".";
}

/** Helper function for implementing ALPS_EXPECT_NEAR. */
template <typename Derived1, typename Derived2>
::testing::AssertionResult NearPredFormat(const char* expr1,
                                          const char* expr2,
                                          const char* abs_error_expr,
                                          const Eigen::MatrixBase<Derived1> &val1,
                                          const Eigen::MatrixBase<Derived2> &val2,
                                          double abs_error)
{
    if (val1.isApprox(val2, abs_error))
        return ::testing::AssertionSuccess();

    return ::testing::AssertionFailure()
           << "The difference between " << expr1 << " and " << expr2
           << " exceeds " << abs_error_expr << ", where\n"
           << expr1 << " evaluates to\n" << val1 << ",\n"
           << expr2 << " evaluates to\n" << val2 << "\n.";
}

}}  /* namespace alps::testing */

/**
 * Extends gtest's EXPECT_NEAR to complex variables and Eigen matrices
 */
#define ALPS_EXPECT_NEAR(val1, val2, abs_error) \
    EXPECT_PRED_FORMAT3(::alps::testing::NearPredFormat, val1, val2, abs_error)

