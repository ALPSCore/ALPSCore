/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file hdf5_native_type.cpp
    Tests `alps::hdf5::native_type` meta-predicate.
*/

#include <alps/hdf5/archive.hpp>
#include <alps/hdf5/vector.hpp>

#include <type_traits>
#include <gtest/gtest.h>

namespace h5=alps::hdf5;

template <typename T>
void testNative(const char* name_of_type) {
    typedef h5::is_native_type<T> test_type;
    EXPECT_TRUE(+test_type::value) << name_of_type;
    EXPECT_TRUE((std::is_same<typename test_type::type, std::true_type>::value)) << name_of_type;
}

template <typename T>
void testNotNative(const char* name_of_type) {
    typedef h5::is_native_type<T> test_type;
    EXPECT_FALSE(+test_type::value) << name_of_type;
    EXPECT_TRUE((std::is_same<typename test_type::type, std::false_type>::value)) << name_of_type;
}

TEST(Hdf5NativeTypeTest, native) {
    testNative<char>("char");
    testNative<signed char>("signed char");
    testNative<unsigned char>("unsigned char");
    testNative<short>("short");
    testNative<unsigned short>("unsigned short");
    testNative<int>("int");
    testNative<unsigned>("unsigned");
    testNative<long>("long");
    testNative<unsigned long>("unsigned long");
    testNative<long long>("long long");
    testNative<unsigned long long>("unsigned long long");
    testNative<float>("float");
    testNative<double>("double");
    testNative<long double>("long double");
    testNative<bool>("bool");
    testNative<std::string>("std::string");
}

struct some_class {};

TEST(Hdf5NativeTypeTest, notNative) {
    testNotNative<int*>("int*");
    testNotNative<void>("void");
    testNotNative< std::vector<int> >(" std::vector<int");
    testNotNative< some_class >(" some_class ");
}
