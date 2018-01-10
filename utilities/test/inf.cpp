/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <vector>
// #include <cmath>
#include <boost/math/special_functions/fpclassify.hpp> /* for portable isinf() */

#include <gtest/gtest.h>

#include "alps/numeric/inf.hpp"
#include "alps/numeric/vector_functions.hpp"

template <typename T>
struct InfinityTest : public ::testing::Test {
    typedef T scalar_type;
    typedef std::vector<T> vector_type;
    typedef std::vector<vector_type> vector_vector_type;

    scalar_type scalar_;
    vector_type vector_;
    vector_vector_type vector_vector_;

    InfinityTest() {
        scalar_=1.5;
        vector_=vector_type(10, 2.25);
        vector_vector_=vector_vector_type(20, vector_type(10, 3.125));
    }

    void ScalarTest() {
        // using std::isinf;
        scalar_type inf=alps::numeric::inf<scalar_type>(scalar_);
        EXPECT_TRUE((boost::math::isinf)(inf));
    }
        
    void VectorTest() {
        // using std::isinf;
        vector_type inf=alps::numeric::inf<vector_type>(vector_);
        ASSERT_EQ(vector_.size(), inf.size());
        for (size_t i=0; i<inf.size(); ++i) {
            EXPECT_TRUE((boost::math::isinf)(inf[i]));
        }
    }
    
    void VectorVectorTest() {
        // using std::isinf;
        vector_vector_type inf=alps::numeric::inf<vector_vector_type>(vector_vector_);
        ASSERT_EQ(vector_vector_.size(), inf.size());
        for (size_t i=0; i<inf.size(); ++i) {
            ASSERT_EQ(vector_vector_[i].size(), inf[i].size());
            for (size_t j=0; j<inf[i].size(); ++j) {
                EXPECT_TRUE((boost::math::isinf)(inf[i][j]));
            }
        }
    }
};
    
typedef ::testing::Types<float, double, long double> basic_types;
TYPED_TEST_CASE(InfinityTest, basic_types);

TYPED_TEST(InfinityTest, ScalarTest) { this->TestFixture::ScalarTest(); }
TYPED_TEST(InfinityTest, VectorTest) { this->TestFixture::VectorTest(); }
TYPED_TEST(InfinityTest, VectorVectorTest) { this->TestFixture::VectorVectorTest(); }
