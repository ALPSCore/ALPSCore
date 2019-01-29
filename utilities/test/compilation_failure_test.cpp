/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/**
   @file compilation_failure_test.cpp
   @brief A demo and a test to test for a compilation error

   Here we define a function template that accepts only floating-point arguments,
   and verify that passing an integer argument does not compile.
*/

#include <type_traits>
#include "gtest/gtest.h"

template <typename T>
void do_something(T& val)
{
    static_assert(std::is_floating_point<T>::value,
                  "do_something(T&) requires a floating point argument");
    val=0.0;
}

TEST(utilities, testCompilationFailure) {
    double x=1.0;
    do_something(x);
    SUCCEED();
#ifdef ALPS_TEST_EXPECT_COMPILE_FAILURE
    // The following code should not compile:
    int m=1;
    do_something(m);
    FAIL() << "This code should not have compiled";
#endif
}
