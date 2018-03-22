/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <vector>
#include <cmath>
#include "alps/numeric/vector_functions.hpp"

#include "gtest/gtest.h"

// #include "vector_comparison_predicates.hpp"

// Generate data of type T, general case
template <typename T, typename U=double>
struct gen_data {
    typedef T value_type;
    value_type value;
    gen_data(U v, std::size_t =0): value(v) {}
    T get() { return value; }
    operator value_type() { return get(); }
};

// Generate data of a vector type
template <typename T, typename U>
struct gen_data<std::vector<T>, U> {
    typedef std::vector<T> value_type;
    value_type value;
    gen_data(U v, std::size_t sz=3): value(sz, gen_data<T,U>(v,sz).get()) {}
    value_type get() { return value; }
    operator value_type() { return get(); }
};


template <typename T>
bool is_near(T v1, T v2) {
    return (std::fabs(v1-v2)<1E-8);
}

template <typename T>
bool is_near(const std::vector<T>& v1, const std::vector<T>& v2) {
    std::size_t sz1=v1.size(), sz2=v2.size();
    if (sz2!=sz1) return false;

    for (std::size_t i=0; i<sz1; ++i) {
        if (!is_near(v1[i], v2[i])) return false;
    }
    return true;
}


// GoogleTest fixture, parametrized over vector element type
template <typename T>
struct VectorFunctionsTest : public ::testing::Test
{
    typedef std::vector<T> value_type;
};

typedef std::vector<double> v_double;
typedef std::vector<v_double> vv_double;

typedef ::testing::Types< double, v_double, vv_double > my_types;
TYPED_TEST_CASE(VectorFunctionsTest, my_types);

TYPED_TEST(VectorFunctionsTest, testPlus) {
    typedef typename TestFixture::value_type value_type;
    using alps::numeric::operator+=;
    value_type vec1=gen_data<value_type>(2.25);
    value_type vec2=gen_data<value_type>(1.50);
    value_type res=gen_data<value_type>(2.25+1.50);
    vec1 += vec2;
    ASSERT_EQ(res.size(),vec1.size());
    EXPECT_EQ(res, vec1);
}

TYPED_TEST(VectorFunctionsTest, testMinus) {
    typedef typename TestFixture::value_type value_type;
    using alps::numeric::operator-=;
    value_type vec1=gen_data<value_type>(2.25);
    value_type vec2=gen_data<value_type>(1.50);
    value_type res=gen_data<value_type>(2.25-1.50);
    vec1 -= vec2;
    ASSERT_EQ(res.size(),vec1.size());
    EXPECT_EQ(res, vec1);
}

TYPED_TEST(VectorFunctionsTest, testMultiplies) {
    typedef typename TestFixture::value_type value_type;
    using alps::numeric::operator*=;
    value_type vec1=gen_data<value_type>(2.25);
    value_type vec2=gen_data<value_type>(1.50);
    value_type res=gen_data<value_type>(2.25*1.50);
    vec1 *= vec2;
    ASSERT_EQ(res.size(),vec1.size());
    EXPECT_EQ(res, vec1);
}

TYPED_TEST(VectorFunctionsTest, testDivides) {
    typedef typename TestFixture::value_type value_type;
    using alps::numeric::operator/=;
    value_type vec1=gen_data<value_type>(2.25);
    value_type vec2=gen_data<value_type>(1.50);
    value_type res=gen_data<value_type>(2.25/1.50);
    vec1 /= vec2;
    ASSERT_EQ(res.size(),vec1.size());
    EXPECT_EQ(res, vec1);
}

TYPED_TEST(VectorFunctionsTest, testMergeToSameSize) {
    typedef typename TestFixture::value_type value_type;
    using alps::numeric::merge;
    value_type vec1=gen_data<value_type>(2.25,5);
    const value_type vec2=gen_data<value_type>(1.50,5);
    value_type res=gen_data<value_type>(2.25+1.50,5);
    value_type& merged=merge(vec1,vec2);
    EXPECT_EQ(&merged, &vec1);

    ASSERT_EQ(5u,vec1.size());
    EXPECT_EQ(res, vec1);
}

TYPED_TEST(VectorFunctionsTest, testMergeToSmaller) {
    typedef typename TestFixture::value_type value_type;
    using alps::numeric::merge;

    value_type vec1=gen_data<value_type>(2.25,5);
    vec1.resize(3);

    const value_type vec2=gen_data<value_type>(1.50,5);

    value_type res= gen_data<value_type>(2.25+1.50, 5);
    value_type res2=gen_data<value_type>(0.00+1.50, 5);
    res[3]=res2[3];
    res[4]=res2[4];

    value_type& merged=merge(vec1,vec2);
    EXPECT_EQ(&merged, &vec1);

    ASSERT_EQ(5u,vec1.size());
    EXPECT_EQ(res, vec1);
}

TYPED_TEST(VectorFunctionsTest, testMergeToLarger) {
    typedef typename TestFixture::value_type value_type;
    using alps::numeric::merge;

    value_type vec1=gen_data<value_type>(2.25,5);

    value_type vec2_tmp=gen_data<value_type>(1.50,5);
    vec2_tmp.resize(3);
    const value_type& vec2=vec2_tmp;

    value_type res= gen_data<value_type>(2.25+1.50, 5);
    value_type res2=gen_data<value_type>(2.25+0.00, 5);
    res[3]=res2[3];
    res[4]=res2[4];

    value_type& merged=merge(vec1,vec2);
    EXPECT_EQ(&merged, &vec1);

    ASSERT_EQ(5u,vec1.size());
    EXPECT_EQ(res, vec1);
}
