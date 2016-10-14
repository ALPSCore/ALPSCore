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
class TestUtilsTest : public ::testing::Test {\
  protected:
    T if_true_;
    T if_false_;
  public:
    TestUtilsTest() : if_true_(alps::testing::datapoint<T>::get(true)),
                      if_false_(alps::testing::datapoint<T>::get(false))
    {}
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
                         std::vector<float>,
                         std::vector<double>,
                         std::vector< std::complex<double> >,
                         std::vector<std::string>,
                         std::vector< std::vector<double> >
                        > MyTestTypes;

TYPED_TEST_CASE(TestUtilsTest, MyTestTypes);

TYPED_TEST(TestUtilsTest, basics) {
    using alps::testing::operator<<;
    
    std::cout << std::boolalpha
              << "get(true)=" << this->if_true_
              << "  get(false)=" << this->if_false_
              << std::endl;
    EXPECT_EQ(this->if_true_, this->if_true_);
    EXPECT_EQ(this->if_false_, this->if_false_);
    EXPECT_NE(this->if_true_, this->if_false_);
}
