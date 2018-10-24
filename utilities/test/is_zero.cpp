/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <gtest/gtest.h>
#include "alps/numeric/is_zero.hpp"

template <typename T>
struct FloatingPointZeroTest : public ::testing::Test {
    typedef T value_type;
    FloatingPointZeroTest() {}
    void ZeroTest() {
      EXPECT_TRUE(alps::numeric::is_zero(value_type(1e-55)));
      EXPECT_TRUE(alps::numeric::is_zero(value_type(-1e-55)));
      EXPECT_TRUE(alps::numeric::is_zero<1>(value_type(1e-15)));
      EXPECT_TRUE(alps::numeric::is_zero<2>(value_type(1e-25)));
      EXPECT_TRUE(alps::numeric::is_zero<3>(value_type(1e-35)));
      EXPECT_TRUE(alps::numeric::is_zero<4>(value_type(1e-45)));
      EXPECT_TRUE(alps::numeric::is_zero<5>(value_type(1e-55)));
    }
    void NonZeroTest() {
      EXPECT_FALSE(alps::numeric::is_zero(std::max(value_type(1e-45), std::numeric_limits<T>::min())));
      EXPECT_FALSE(alps::numeric::is_zero(-std::max(value_type(1e-45), std::numeric_limits<T>::min())));
      EXPECT_FALSE(alps::numeric::is_zero<1>(std::max(value_type(1e-5), std::numeric_limits<T>::min())));
      EXPECT_FALSE(alps::numeric::is_zero<2>(std::max(value_type(1e-15), std::numeric_limits<T>::min())));
      EXPECT_FALSE(alps::numeric::is_zero<3>(std::max(value_type(1e-25), std::numeric_limits<T>::min())));
      EXPECT_FALSE(alps::numeric::is_zero<4>(std::max(value_type(1e-35), std::numeric_limits<T>::min())));
      EXPECT_FALSE(alps::numeric::is_zero<5>(std::max(value_type(1e-45), std::numeric_limits<T>::min())));
    }
};
    
template <typename T>
struct IntegralZeroTest : public ::testing::Test {
    typedef T value_type;
    IntegralZeroTest() {}
    void ZeroTest() {
      EXPECT_TRUE(alps::numeric::is_zero(value_type(0)));
    }
    void NonZeroTest() {
      EXPECT_FALSE(alps::numeric::is_zero(value_type(1)));
      EXPECT_FALSE(alps::numeric::is_zero(value_type(-1)));
    }
};
    
typedef ::testing::Types<float, double, long double> float_types;
TYPED_TEST_CASE(FloatingPointZeroTest, float_types);

TYPED_TEST(FloatingPointZeroTest, ZeroTest) { this->TestFixture::ZeroTest(); }
TYPED_TEST(FloatingPointZeroTest, NonZeroTest) { this->TestFixture::NonZeroTest(); }

typedef ::testing::Types<bool, char, signed char, unsigned char, short, unsigned short, int, unsigned int, long, unsigned long, long long, unsigned long long> integral_types;
TYPED_TEST_CASE(IntegralZeroTest, integral_types);

TYPED_TEST(IntegralZeroTest, ZeroTest) { this->TestFixture::ZeroTest(); }
TYPED_TEST(IntegralZeroTest, NonZeroTest) { this->TestFixture::NonZeroTest(); }
