/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file testharness.cpp : Test the accumulator test harness */

#include <vector>

#include "gtest/gtest.h"
#include "gtest/gtest-spi.h" /* testing the GTest framework */

#include "accumulator_generator.hpp"

template <typename T>
class HarnessTest : public ::testing::Test {
    public:
    typedef T scalar_type;
    typedef std::vector<T> vector_type;
    vector_type vdata_;
    scalar_type sdata_;

    HarnessTest() {
        sdata_=scalar_type(1234.5);
        vdata_=vector_type(10,4321.25);
    }
};

namespace aa=alps::accumulators;
namespace aat=aa::testing;

typedef ::testing::Types<float, double, long double> basic_types;
TYPED_TEST_CASE(HarnessTest,basic_types);

TYPED_TEST(HarnessTest,CompareNearScalar) {
    typedef typename TestFixture::scalar_type scalar_type;
    aat::compare_near(scalar_type(1234.5+0.125),this->sdata_,scalar_type(0.13),"scalar_test");
    EXPECT_NONFATAL_FAILURE(
        aat::compare_near(scalar_type(1234.5+0.125), this->sdata_, scalar_type(0.), "scalar_test"),
        "Values of scalar_test differ");
}
    
TYPED_TEST(HarnessTest,CompareNearVector) {
    typedef typename TestFixture::vector_type vector_type;
    vector_type expected=vector_type(10, 4321.25);
    expected[3] += 0.125;
    vector_type wrong_sized=vector_type(9, 4321.25);
    aat::compare_near(expected,this->vdata_,0.13,"vector_test");
    EXPECT_NONFATAL_FAILURE(aat::compare_near(expected, this->vdata_, 0., "vector_test"), "Element #3 of vector_test differs");
    EXPECT_NONFATAL_FAILURE(aat::compare_near(wrong_sized, this->vdata_, 0., "vector_test2"), "Sizes of vector_test2 differ");
}

TYPED_TEST(HarnessTest,GenScalarDataImplicit) {
    typedef typename TestFixture::scalar_type scalar_type;
    scalar_type d=aat::gen_data<scalar_type>(this->sdata_); // implicit conversion
    EXPECT_EQ(this->sdata_, d);
}

TYPED_TEST(HarnessTest,GenScalarDataExplicit) {
    typedef typename TestFixture::scalar_type scalar_type;
    aat::gen_data<scalar_type> g(this->sdata_);
    EXPECT_EQ(this->sdata_, g.value());
}

TYPED_TEST(HarnessTest,GenVectorDataImplicit) {
    typedef typename TestFixture::vector_type vector_type;
    vector_type d=aat::gen_data<vector_type>(this->vdata_[0],10); // implicit conversion
    EXPECT_EQ(this->vdata_, d);
}

TYPED_TEST(HarnessTest,GenVectorDataExplicit) {
    typedef typename TestFixture::vector_type vector_type;
    aat::gen_data<vector_type> g(this->vdata_[0],10);
    EXPECT_EQ(this->vdata_, g.value());
}

TEST(HarnessTest,RandomData) {
    srand48(33);
    double expected=drand48();
    aat::RandomData rd(33);
    EXPECT_EQ(expected, rd());
}

TEST(HarnessTest,ConstantData) {
    double expected=0.25;
    aat::ConstantData cd(0.25);
    EXPECT_EQ(expected, cd());
}

TEST(HarnessTest,AlternatingData) {
    aat::AlternatingData ad(0.3);
    EXPECT_NEAR(0.5-0.3, ad(), 1E-8);
    EXPECT_NEAR(0.5+0.3, ad(), 1E-8);
}


// proxy template to pass data generator G, number of points to generate N, and expected tolerance (as integer!) E
template <typename G, std::size_t N=10000, int E=8>
struct generator {
    typedef G generator_type;
    static const std::size_t NPOINTS=N;
    static const int TOLEXP=E;
};

// GoogleTest fixture parametrized over data generator, via the proxy template above
template <typename G>
struct HarnessDatagenTest : public testing::Test {
    typedef typename G::generator_type generator_type;
    generator_type gen;
    static double tol() { return std::pow(10.,-G::TOLEXP); }
    static const std::size_t NPOINTS=G::NPOINTS;

    static void stats(const generator_type& gen, std::size_t n, double* mean, double* err) {
        double sum=0;
        double sum2=0;
        for (std::size_t i=0; i<n; ++i) {
            double x=gen();
            sum += x;
            sum2 += x*x;
        }
        *mean = sum/n;
        double dev=(sum2 - sum*sum/n)/n;
        *err = std::sqrt(dev/(n-1));
    }
};

typedef ::testing::Types<
    generator<aat::CorrelatedData<0>, 20000000, 4>,
    generator<aat::CorrelatedData<1>, 20000000, 4>,
    generator<aat::CorrelatedData<5>, 20000000, 4>,
    generator<aat::RandomData, 20000000, 4>,
    generator<aat::ConstantData>,
    generator<aat::AlternatingData>,
    generator<aat::LinearData>
    > generator_types;

TYPED_TEST_CASE(HarnessDatagenTest,generator_types);

TYPED_TEST(HarnessDatagenTest, statTest) {
    double mean,err;
    const std::size_t n=TestFixture::NPOINTS;
    this->stats(this->gen,n,&mean,&err);
    EXPECT_NEAR(mean, this->gen.mean(n), this->tol()) << "mean is incorrect";
    EXPECT_NEAR(err,  this->gen.error(n), this->tol()) << "error bar is incorrect";
}
