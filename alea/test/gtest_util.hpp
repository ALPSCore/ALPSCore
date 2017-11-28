#pragma once

#include "gtest/gtest.h"

#include <Eigen/Core>
#include <type_traits>

template <typename T>
struct is_scalar : public std::false_type { };

template <typename T>
struct is_scalar< std::complex<T> > : public is_scalar<T> { };

template <> struct is_scalar<float> : public std::true_type { };
template <> struct is_scalar<double> : public std::true_type { };
template <> struct is_scalar<long double> : public std::true_type { };

// Helper function for implementing EXPECT_CLOSE.
template <typename T,
          typename std::enable_if<is_scalar<T>::value>::type * = 0>
::testing::AssertionResult NearPredFormat(const char* expr1,
                                          const char* expr2,
                                          const char* abs_error_expr,
                                          T val1,
                                          T val2,
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

// Helper function for implementing EXPECT_CLOSE.
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

#define ALPS_EXPECT_NEAR(val1, val2, abs_error) \
  EXPECT_PRED_FORMAT3(NearPredFormat, val1, val2, abs_error)

