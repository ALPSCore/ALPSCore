/*
 * Copyright (C) 1998-2016 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file binop_mixed.cpp
    Test simple binary operations between results from different accumulators.
*/
#include <alps/accumulators.hpp>

#include <gtest/gtest.h>

#include "accumulator_generator.hpp"

template <typename A1, typename A2, typename G, std::size_t N>
struct AccPair {
    typedef A1 left_acc_type; // e.g. FullBinningAccumulator<double>
    typedef A2 right_acc_type; // e.g. NoBinningAccumulator<double>
    typedef G generator_type;  // e.g. ConstantData<double>
    static const std::size_t NPOINTS=N; // e.g. 10000
};

namespace aa=alps::accumulators;
namespace aat=alps::accumulators::testing;

template <typename PAIR>
class AccumulatorMixedBinaryTest : public ::testing::Test {
    public:
    typedef typename PAIR::left_acc_type left_acc_type;
    typedef typename PAIR::right_acc_type right_acc_type;
    typedef typename PAIR::generator_type generator_type;
    typedef typename aa::value_type<typename left_acc_type::accumulator_type>::type value_type;
    static const std::size_t NPOINTS=PAIR::NPOINTS;

    generator_type scalar_gen_;
    // left_acc_type left_acc_;
    // right_acc_type right_acc_;

    double exp_scalar_left_mean_;
    double exp_scalar_right_mean_;
    
    aa::accumulator_set aset_;
    boost::shared_ptr<aa::result_set> rset_p_;

    aa::result_set& results() {
        return *rset_p_;
    }

    aa::result_wrapper& result(const std::string& name) {
        return (*rset_p_)[name];
    }

    AccumulatorMixedBinaryTest()
    {
        aset_ << left_acc_type("left")
            << right_acc_type("right");
        
        for (int i=0; i<NPOINTS; ++i) {
            double v=scalar_gen_(); // both accs have the same sequence of values (FIXME?)
            aset_["left"] << aat::gen_data<value_type>(v).value();
            aset_["right"] << aat::gen_data<value_type>(v).value();
        }
        rset_p_=boost::shared_ptr<aa::result_set>(new aa::result_set(aset_));

        exp_scalar_left_mean_=scalar_gen_.mean(NPOINTS);
        exp_scalar_right_mean_=exp_scalar_left_mean_;
    }

#define GENERATE_TEST_MEMBER(_name_, _op_)                                     \
    void _name_() {                                                     \
        aa::result_wrapper r=result("left") _op_ result("right");       \
        value_type xmean=r.mean<value_type>();                          \
        value_type expected_mean=aat::gen_data<value_type>(exp_scalar_left_mean_ _op_ exp_scalar_right_mean_); \
        aat::compare_near(expected_mean, xmean, 1E-3, "mean");          \
    }

    GENERATE_TEST_MEMBER(add,+)
    GENERATE_TEST_MEMBER(sub,-)
    GENERATE_TEST_MEMBER(mul,*)
    GENERATE_TEST_MEMBER(div,/)
#undef GENERATE_TEST
};

TYPED_TEST_CASE_P(AccumulatorMixedBinaryTest);

#define GENERATE_TEST(_name_) \
TYPED_TEST_P(AccumulatorMixedBinaryTest,_name_) { this->TestFixture::_name_(); }

GENERATE_TEST(add)
GENERATE_TEST(sub)
GENERATE_TEST(mul)
GENERATE_TEST(div)

REGISTER_TYPED_TEST_CASE_P(AccumulatorMixedBinaryTest, add, sub, mul, div);

typedef ::testing::Types<
    AccPair<aa::MeanAccumulator<double>, aa::MeanAccumulator<double>, aat::ConstantData, 1000>,
    AccPair<aa::NoBinningAccumulator<double>, aa::MeanAccumulator<double>, aat::ConstantData, 1000>
    > test_types;

INSTANTIATE_TYPED_TEST_CASE_P(MixedBinaryTest, AccumulatorMixedBinaryTest, test_types);
