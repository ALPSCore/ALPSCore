/*
 * Copyright (C) 1998-2016 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** Testing the test utilities */

#include <iostream>
#include <gtest/gtest.h>

#include "./test_utils.hpp"

template <typename T>
class TestUtilsTest : public ::testing::Test {
  public:
    typedef T value_type;
    typedef std::vector<T> vector_type;

    void scalar_test()
    {
        using alps::testing::operator<<;
        value_type if_true(alps::testing::datapoint<value_type>::get(true));
        value_type if_false(alps::testing::datapoint<value_type>::get(false));
        
        std::cout << std::boolalpha
                  << "get(true)=" << if_true
                  << "  get(false)=" << if_false
                  << std::endl;
        EXPECT_NE(if_true, if_false);
    }
        
    void scalar_with_size_test()
    {
        using alps::testing::operator<<;
        value_type if_true(alps::testing::datapoint<value_type>::get(true,10));
        value_type if_false(alps::testing::datapoint<value_type>::get(false,10));
        
        std::cout << std::boolalpha
                  << "get(true)=" << if_true
                  << "  get(false)=" << if_false
                  << std::endl;
        EXPECT_NE(if_true, if_false);
    }
        
    void vector_test()
    {
        using alps::testing::operator<<;
        vector_type if_true(alps::testing::datapoint<vector_type>::get(true));
        vector_type if_false(alps::testing::datapoint<vector_type>::get(false));
        
        std::cout << std::boolalpha
                  << "get(true)=" << if_true
                  << "  get(false)=" << if_false
                  << std::endl;
        EXPECT_NE(if_true.size(), if_false.size());
        EXPECT_NE(if_true, if_false);
    }
        
    void vector_with_size_test()
    {
        using alps::testing::operator<<;
        const std::size_t sz=7;
        vector_type if_true(alps::testing::datapoint<vector_type>::get(true,sz));
        vector_type if_false(alps::testing::datapoint<vector_type>::get(false,sz));
        
        std::cout << std::boolalpha
                  << "get(true)=" << if_true
                  << "  get(false)=" << if_false
                  << std::endl;
        EXPECT_EQ(sz, if_true.size());
        EXPECT_EQ(sz, if_false.size());
        EXPECT_NE(if_true, if_false);
    }
};

typedef ::testing::Types<bool,
                         char,
                         int,
                         unsigned int,
                         long,
                         unsigned long,
                         float,
                         double,
                         std::string,
                         std::complex<float>,
                         std::complex<double>,
                         std::vector<double> // tests vector of vectors handling
                        > MyTestTypes;

TYPED_TEST_CASE(TestUtilsTest, MyTestTypes);

TYPED_TEST(TestUtilsTest, ScalarTest) { this->scalar_test(); }
TYPED_TEST(TestUtilsTest, ScalarWithSizeTest) { this->scalar_with_size_test(); }
TYPED_TEST(TestUtilsTest, VectorTest) { this->vector_test(); }
TYPED_TEST(TestUtilsTest, VectorWithSizeTest) { this->vector_with_size_test(); }

template <typename T>
class TestUtilsStringTest : public TestUtilsTest<T> { };

TYPED_TEST_CASE(TestUtilsStringTest, std::string);
TYPED_TEST(TestUtilsStringTest, StringWithSize) {
    typedef typename TestFixture::value_type value_type;
    const std::size_t sz=10;
    value_type if_true(alps::testing::datapoint<value_type>::get(true,sz));
    value_type if_false(alps::testing::datapoint<value_type>::get(false,sz));
    EXPECT_EQ(sz, if_true.size());
    EXPECT_EQ(sz, if_false.size());
    EXPECT_NE(if_true, if_false);
}
