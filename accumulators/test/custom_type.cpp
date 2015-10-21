/*
 * Copyright (C) 1998-2015 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file custom_type.cpp: Test for using a custom type */

// Must be included before accumulators.hpp
#include "custom_type.hpp"

// Defines the notion of accumulator, instantiates templates defined above.
#include "alps/accumulators.hpp"

#include "gtest/gtest.h"
#include "accumulator_generator.hpp"


// Data generation functions

namespace alps {
    namespace accumulators {
        namespace testing {

            /// Generate a value of `my_custom_type<T>`
            template <typename T>
            struct gen_data< my_custom_type<T> > {
                typedef my_custom_type<T> value_type;
                T val_;
                gen_data(T val, unsigned int =0) : val_(val) {}
                operator value_type() { return value_type::generate(val_); }
            };
        
    
            // Data inquiry functions
            template <typename T>
            double get_value(const T& x) {
                return x;
            }
            
            template <typename T>
            double get_value(const my_custom_type<T>& x) {
                return x.get_my_value();
            }
        } // testing::
    } // accumulators::
} // alps::

using namespace alps::accumulators::testing;


// to pass accumulator types
template <template<typename> class A, typename T>
struct AccumulatorTypeGenerator {
    typedef acc_one_correlated_gen<A,T,     /*NPOINTS*/50000,/*CORRL*/10,/*VLEN*/0,RandomData>   accumulator_gen_type;     // generator for A<T>
    typedef acc_one_correlated_gen<A,double,/*NPOINTS*/50000,/*CORRL*/1 ,/*VLEN*/0,ConstantData> dbl_accumulator_gen_type; // generator for A<double>
};

template <typename G>
struct CustomTypeAccumulatorTest : public testing::Test {
    typedef typename G::accumulator_gen_type acc_gen_type;
    typedef typename acc_gen_type::acc_type acc_type;
    typedef typename acc_type::accumulator_type raw_acc_type;
    typedef typename alps::accumulators::value_type<raw_acc_type>::type value_type;
    
    typedef typename G::dbl_accumulator_gen_type dbl_acc_gen_type;
    typedef typename dbl_acc_gen_type::acc_type dbl_acc_type;

    static const int NPOINTS=acc_gen_type::NPOINTS; 
    
    acc_gen_type acc_gen;
    dbl_acc_gen_type dbl_acc_gen;

    // Ugly, but should work
    static const bool is_mean_acc=boost::is_same<alps::accumulators::MeanAccumulator<value_type>,
                                                 acc_type>::value;
    static const bool is_nobin_acc=boost::is_same<alps::accumulators::NoBinningAccumulator<value_type>,
                                                  acc_type>::value;
    
    static double tol() { return 5.E-3; }

    CustomTypeAccumulatorTest() {}

    void TestH5ScalarType() {
        typedef typename alps::hdf5::scalar_type<value_type>::type stype;
        EXPECT_EQ(typeid(typename value_type::my_constituent_type), typeid(stype)) << "type is: " << typeid(stype).name();
    }

    void TestNumScalarType() {
        typedef typename alps::numeric::scalar<value_type>::type stype;
        EXPECT_EQ(typeid(typename value_type::my_scalar_type), typeid(stype)) << "type is: " << typeid(stype).name();
    }

    void TestElementType() {
        typedef typename alps::element_type<value_type>::type stype;
        EXPECT_TRUE(alps::is_sequence<value_type>::value);
        EXPECT_EQ(typeid(typename value_type::my_element_type), typeid(stype)) << "type is: " << typeid(stype).name();
    }

    void TestCount() {
        EXPECT_EQ(NPOINTS, acc_gen.result().count());
    }

    void TestMean() {
        EXPECT_NEAR(acc_gen.expected_mean(), get_value(acc_gen.result().template mean<value_type>()), tol());
    }

    void TestError() {
        if (is_mean_acc) return;
        acc_gen.result().template error<value_type>();
    }

    void TestTau() {
        if (is_mean_acc || is_nobin_acc) return;
        value_type tau=acc_gen.result().template autocorrelation<value_type>();
        std::cout << "Autocorrelation is " << tau << std::endl;
        // EXPECT_NEAR(24, get_value(acc_gen.result().template autocorrelation<value_type>()),1.0); // << FIXME: what should be the correct value?
    }

    void TestScaleConst() {
        const alps::accumulators::result_wrapper& r=acc_gen.result()*2;
        EXPECT_NEAR(acc_gen.expected_mean()*2, get_value(r.mean<value_type>()), tol());
    }

    void TestScale() {
        const alps::accumulators::result_wrapper& r=acc_gen.result()*dbl_acc_gen.result();
        EXPECT_NEAR(acc_gen.expected_mean()*dbl_acc_gen.expected_mean(), get_value(r.mean<value_type>()), tol());
    }

    void TestAddConst() {
        const alps::accumulators::result_wrapper& r=acc_gen.result()+2;
        EXPECT_NEAR(acc_gen.expected_mean()+2, get_value(r.mean<value_type>()), tol());
    }

    void TestAddEqConst() {
        alps::accumulators::result_wrapper& r=acc_gen.results()["data"];
        r+=2;
        EXPECT_NEAR(acc_gen.expected_mean()+2, get_value(r.mean<value_type>()), tol());
    }

    void TestAdd() {
        const alps::accumulators::result_wrapper& r=acc_gen.result()+acc_gen.result();
        EXPECT_NEAR(acc_gen.expected_mean()*2, get_value(r.mean<value_type>()), tol());
    }

    void TestAddEq() {
        alps::accumulators::result_wrapper& r=acc_gen.results()["data"];
        r+=acc_gen.results()["data"];
        EXPECT_NEAR(acc_gen.expected_mean()*2, get_value(r.mean<value_type>()), tol());
    }

    void TestAddScalar() {
        const alps::accumulators::result_wrapper& r=acc_gen.result()+dbl_acc_gen.result();
        double expected_mean=acc_gen.expected_mean()+dbl_acc_gen.expected_mean();
        EXPECT_NEAR(expected_mean, get_value(r.mean<value_type>()), tol());
    }

    void TestAddEqScalar() {
        alps::accumulators::result_wrapper& r=acc_gen.results()["data"];
        r+=dbl_acc_gen.result();
        double expected_mean=acc_gen.expected_mean()+dbl_acc_gen.expected_mean();
        EXPECT_NEAR(expected_mean, get_value(r.mean<value_type>()), tol());
    }

};

typedef my_custom_type<double> dbl_custom_type;
typedef ::testing::Types<
    AccumulatorTypeGenerator<alps::accumulators::MeanAccumulator, dbl_custom_type>,
    AccumulatorTypeGenerator<alps::accumulators::NoBinningAccumulator,dbl_custom_type>,
    AccumulatorTypeGenerator<alps::accumulators::LogBinningAccumulator,dbl_custom_type>,
    AccumulatorTypeGenerator<alps::accumulators::FullBinningAccumulator,dbl_custom_type>
    > MyTypes;

TYPED_TEST_CASE(CustomTypeAccumulatorTest, MyTypes);

#define MAKE_TEST(_name_) TYPED_TEST(CustomTypeAccumulatorTest, _name_)  { this->TestFixture::_name_(); }

MAKE_TEST(TestH5ScalarType)
MAKE_TEST(TestNumScalarType)
MAKE_TEST(TestElementType)

MAKE_TEST(TestMean)
MAKE_TEST(TestError)
MAKE_TEST(TestTau)

MAKE_TEST(TestScaleConst)
MAKE_TEST(TestScale)
MAKE_TEST(TestAddConst)
MAKE_TEST(TestAdd)
MAKE_TEST(TestAddScalar)
MAKE_TEST(TestAddEqConst)
MAKE_TEST(TestAddEq)
MAKE_TEST(TestAddEqScalar)

