/*
 * Copyright (C) 1998-2015 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_UTILITIES_TEST_VECTOR_COMPARISON_PREDICATES_HPP
#define ALPS_UTILITIES_TEST_VECTOR_COMPARISON_PREDICATES_HPP

#include <vector>
#include "gtest/gtest.h"

template <typename T>
::testing::AssertionResult is_near(T v1, T v2) {
    if (std::fabs(v1-v2)<1E-8) {
        return ::testing::AssertionSuccess() << v1 << " almost equals " << v2;
    } else {
        return ::testing::AssertionFailure() << v1 << " not equal " << v2;
    }
}

template <typename T>
::testing::AssertionResult is_near(const std::vector<T>& v1, const std::vector<T>& v2) {
    std::size_t sz1=v1.size(), sz2=v2.size();
    if (sz2!=sz1) {
        return ::testing::AssertionFailure() << "sizes differ: left=" << sz1 << " right=" << sz2;
    }
    for (std::size_t i=0; i<sz1; ++i) {
        ::testing::AssertionResult res=is_near(v1[i], v2[i]);
        if (!res) {
            res << "; content differs at #" << i;
            return res;
        }
    }
    return ::testing::AssertionSuccess() << "vectors are (almost) equal";
}

#endif /* ALPS_UTILITIES_TEST_VECTOR_COMPARISON_PREDICATES_HPP */
