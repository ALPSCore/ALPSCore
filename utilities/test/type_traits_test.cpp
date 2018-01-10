/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <gtest/gtest.h>

#include <typeinfo>
#include <vector>
#include <complex>
#include <iostream>

#include "alps/numeric/scalar.hpp"
#include "alps/type_traits/is_scalar.hpp"

// FIXME: add eigen matrices here
// FIXME: make it a type-parametrized test

typedef std::vector<double> double_vec;
typedef std::complex<double> double_complex;

TEST(TypeTraits,ScalarType) {
    typedef std::vector<double> double_vec;
    typedef std::complex<double> double_complex;

    EXPECT_EQ(typeid(alps::numeric::scalar<double>::type), typeid(double));
    EXPECT_EQ(typeid(alps::numeric::scalar<double_complex>::type), typeid(double_complex));
    EXPECT_EQ(typeid(alps::numeric::scalar<double_vec>::type), typeid(double));
}

TEST(TypeTraits,IsScalar) {
    EXPECT_TRUE(alps::is_scalar<double>::value);
    EXPECT_TRUE(alps::is_scalar<double_complex>::value);
    EXPECT_FALSE(alps::is_scalar<double_vec>::value);
}

