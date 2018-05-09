/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include "gtest/gtest.h"
#include <alps/alea/testing.hpp>

namespace alps { namespace alea {

template <typename T>
struct is_scalar : public std::false_type { };

template <typename T>
struct is_scalar< std::complex<T> > : public is_scalar<T> { };

template <> struct is_scalar<float> : public std::true_type { };
template <> struct is_scalar<double> : public std::true_type { };
template <> struct is_scalar<long double> : public std::true_type { };

/** Helper function for implementing ALPS_HYPOTHESIZE_EQUAL. */
template <typename R1, typename R2>
std::enable_if<is_result<R1>::value && is_result<R2>::value,
               ::testing::AssertionResult>::type
HypothesizeEqual(const char* expr1, const char* expr2, const char* alpha_expr,
                 const R1 &val1, const R2 &val2, double alpha)
{
    t2_result result = test_mean(val1, val2);
    if (result.pvalue() >= alpha)
        return ::testing::AssertionSuccess();

    return ::testing::AssertionFailure()
        << "A T2 test for equality of the means of " << expr1 << " and " << expr2
        << "  gives a p-value of " << result.pvalue()
        << ", which is lower than a threshold of " << alpha << ", where\n"
        << expr1 << " evaluates to " << val1 << ",\n"
        << expr2 << " evaluates to " << val2 << ".";
}

}}  /* namespace alps::alea */

/**
 *
 */
#define ALPS_HYPOTHESIZE_EQUAL(val1, val2, alpha) \
    EXPECT_PRED_FORMAT3(::alps::alea::HypothesizeEqual, val1, val2, alpha)

