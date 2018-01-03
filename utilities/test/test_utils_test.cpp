/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** Testing the test utilities */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>

#include "./test_utils.hpp"

// Different types have different useful range...
template <typename T>
struct range {
    static const std::size_t VALUE=1000;
};

template <>
struct range<bool> {
    static const std::size_t VALUE=2;
};

template <>
struct range<char> {
    static const std::size_t VALUE=256;
};

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

        EXPECT_NE(if_true, if_false);

        const unsigned int nvals=range<value_type>::VALUE;
        std::vector<value_type> vals(nvals);

        for (int i=0; i<static_cast<int>(nvals); ++i) {
            vals[i]=alps::testing::datapoint<value_type>::get(i);
        }
        for (int i=0; i<static_cast<int>(nvals); ++i) {
            for (int j=i+1; j<static_cast<int>(nvals); ++j) {
                ASSERT_NE(vals[i], vals[j]) << "Value clash at i=" << i << ", j=" << j;
            }
        }
        // Sample printout
        std::cout << "  get(0)=" << vals[0]
                  << "  get(1)=" << vals[1]
                  << "  get(" << (nvals-1) << ")=" << vals[nvals-1]
                  << std::endl;

        // Sample printouts
        std::cout << std::boolalpha
                  << "get(true)=" << if_true
                  << "  get(false)=" << if_false
                  << std::endl;
    }
        
    void scalar_with_size_test()
    {
        using alps::testing::operator<<;
        value_type if_true(alps::testing::datapoint<value_type>::get(true,10));
        value_type if_false(alps::testing::datapoint<value_type>::get(false,10));
        
        const unsigned int nvals=range<value_type>::VALUE;
        std::vector<value_type> vals(nvals);

        for (int i=0; i<static_cast<int>(nvals); ++i) {
            vals[i]=alps::testing::datapoint<value_type>::get(i,10);
        }
        for (int i=0; i<static_cast<int>(nvals); ++i) {
            for (int j=i+1; j<static_cast<int>(nvals); ++j) {
                ASSERT_NE(vals[i], vals[j]) << "Value clash at i=" << i << ", j=" << j;
            }
        }
        // Sample printout
        std::cout << "  get(0)=" << vals[0]
                  << "  get(1)=" << vals[1]
                  << "  get(" << (nvals-1) << ")=" << vals[nvals-1]
                  << std::endl;

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
        
        const unsigned int nvals=range<value_type>::VALUE;
        std::vector<vector_type> vals(nvals);

        for (int i=0; i<static_cast<int>(nvals); ++i) {
            vals[i]=alps::testing::datapoint<vector_type>::get(i);
        }
        for (int i=0; i<static_cast<int>(nvals); ++i) {
            for (int j=i+1; j<static_cast<int>(nvals); ++j) {
                ASSERT_NE(vals[i], vals[j]) << "Value clash at i=" << i << ", j=" << j;
            }
        }
        // Sample printout
        std::cout << "  get(0)=" << vals[0]
                  << "  get(1)=" << vals[1]
                  << "  get(" << (nvals-1) << ")=" << vals[nvals-1]
                  << std::endl;

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
        
        const unsigned int nvals=range<value_type>::VALUE;
        std::vector<vector_type> vals(nvals);

        for (int i=0; i<static_cast<int>(nvals); ++i) {
            vals[i]=alps::testing::datapoint<vector_type>::get(i,sz);
        }
        for (int i=0; i<static_cast<int>(nvals); ++i) {
            for (int j=i+1; j<static_cast<int>(nvals); ++j) {
                ASSERT_NE(vals[i], vals[j]) << "Value clash at i=" << i << ", j=" << j;
            }
        }
        // Sample printout
        std::cout << "  get(0)=" << vals[0]
                  << "  get(1)=" << vals[1]
                  << "  get(" << (nvals-1) << ")=" << vals[nvals-1]
                  << std::endl;

        std::cout << std::boolalpha
                  << "get(true)=" << if_true
                  << "  get(false)=" << if_false
                  << std::endl;
        EXPECT_EQ(sz, if_true.size());
        EXPECT_EQ(sz, if_false.size());
        EXPECT_NE(if_true, if_false);
    }

    void print_test()
    {
        using alps::testing::operator<<;
        vector_type empty_vec;
        vector_type true_vec=alps::testing::datapoint<vector_type>::get(true);
        vector_type false_vec=alps::testing::datapoint<vector_type>::get(false);
        std::cout << "Empty vector=" << empty_vec << std::endl;
        std::cout << "\"True\" vector=" << true_vec << std::endl;
        std::cout << "\"False\" vector=" << false_vec << std::endl;
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
TYPED_TEST(TestUtilsTest, PrintVectorTest) { this->print_test(); }

template <typename T>
class TestUtilsStringTest : public TestUtilsTest<T> { };

TYPED_TEST_CASE(TestUtilsStringTest, std::string);
TYPED_TEST(TestUtilsStringTest, StringWithSize) {
    typedef typename TestFixture::value_type value_type;
    const std::size_t sz=10;
    value_type if_true(alps::testing::datapoint<value_type>::get(true,sz));
    value_type if_false(alps::testing::datapoint<value_type>::get(false,sz));
    const unsigned int nvals=range<value_type>::VALUE;
    std::vector<value_type> vals(nvals);
    
    for (int i=0; i<static_cast<int>(nvals); ++i) {
        vals[i]=alps::testing::datapoint<value_type>::get(i,sz);
    }
    for (int i=0; i<static_cast<int>(nvals); ++i) {
        for (int j=i+1; j<static_cast<int>(nvals); ++j) {
            ASSERT_NE(vals[i], vals[j]) << "Value clash at i=" << i << ", j=" << j;
        }
    }
    // Sample printout
    std::cout << "  get(0)=" << vals[0]
              << "  get(1)=" << vals[1]
              << "  get(" << (nvals-1) << ")=" << vals[nvals-1]
              << std::endl;

    EXPECT_EQ(sz, if_true.size());
    EXPECT_EQ(sz, if_false.size());
    EXPECT_NE(if_true, if_false);
}

