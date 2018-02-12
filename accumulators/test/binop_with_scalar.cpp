/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file binop_with_scalar.cpp
    Test simple binary operations between vector and scalr results.
*/


#include <vector>
#include <type_traits>
#include "gtest/gtest.h"

#include "alps/accumulators.hpp"

#include "accumulator_generator.hpp"
using namespace alps::accumulators::testing;


/// Google Test Fixture: argument is acc_vs_pair_gen<A,T> to test (A<vector<T>> op A<T>)
template <typename TGEN>
class AccumulatorVSBinaryOpTest : public ::testing::Test {
  public:
    typedef TGEN acc_gen_type;
    typedef typename TGEN::vector_acc_type vector_acc_type;
    typedef typename TGEN::scalar_acc_type scalar_acc_type;
    typedef typename TGEN::scalar_data_type scalar_data_type;
    typedef typename TGEN::vector_data_type vector_data_type;
    // Ugly, but should work
    static const bool is_mean_acc=std::is_same<alps::accumulators::MeanAccumulator<scalar_data_type>, scalar_acc_type>::value;

  private:
    // This will generate 2 pairs of accumulators.
    acc_gen_type lhs_acc_gen_;
    acc_gen_type rhs_acc_gen_;

  public:
    AccumulatorVSBinaryOpTest(): lhs_acc_gen_("scalar","vector"), rhs_acc_gen_("scalar","vector") {}

    /// Function FUN testing operator OP
#define ALPS_TEST_GENERATE(FUN,OP) \
    void FUN()  const                                                                               \
    {                                                                                               \
        using alps::accumulators::result_wrapper;                                                   \
        using alps::accumulators::result_set;                                                       \
                                                                                                    \
        const result_set& lhs=lhs_acc_gen_.results();                                               \
        const result_set& rhs=rhs_acc_gen_.results();                                               \
                                                                                                    \
        const result_wrapper& res_vv=lhs["vector"] OP rhs["vector"];                                \
        const result_wrapper& res_vs=lhs["vector"] OP rhs["scalar"];                                \
                                                                                                    \
        const vector_data_type vv_mean=res_vv.mean<vector_data_type>();                             \
        const vector_data_type vs_mean=res_vs.mean<vector_data_type>();                             \
                                                                                                    \
        ASSERT_EQ(vv_mean.size(),vs_mean.size()) << "Vector means sizes differ!";                   \
        for (size_t i=0; i<vv_mean.size(); ++i) {                                                      \
            EXPECT_NEAR(vv_mean[i], vs_mean[i], 1E-8) << "Vector means differ at element " << i;    \
        }                                                                                           \
                                                                                                    \
        /* do not test error if it is not implemented for the accumulator */                        \
        if (is_mean_acc) return;                                                                    \
                                                                                                    \
        const vector_data_type vv_err=res_vv.error<vector_data_type>();                             \
        const vector_data_type vs_err=res_vs.error<vector_data_type>();                             \
                                                                                                    \
        ASSERT_EQ(vv_err.size(),vs_err.size()) << "Vector errors sizes differ!";                    \
        for (size_t i=0; i<vv_err.size(); ++i) {                                                       \
            EXPECT_NEAR(vv_err[i], vs_err[i], 1E-8) << "Vector errors differ at element " << i;     \
        }                                                                                           \
    }

    ALPS_TEST_GENERATE(add,+);
    ALPS_TEST_GENERATE(sub,-);
    ALPS_TEST_GENERATE(mul,*);
    ALPS_TEST_GENERATE(div,/);
#undef ALPS_TEST_GENERATE

};

TYPED_TEST_CASE_P(AccumulatorVSBinaryOpTest);

TYPED_TEST_P(AccumulatorVSBinaryOpTest,add) { this->TestFixture::add(); }
TYPED_TEST_P(AccumulatorVSBinaryOpTest,sub) { this->TestFixture::sub(); }
TYPED_TEST_P(AccumulatorVSBinaryOpTest,mul) { this->TestFixture::mul(); }
TYPED_TEST_P(AccumulatorVSBinaryOpTest,div) { this->TestFixture::div(); }

REGISTER_TYPED_TEST_CASE_P(AccumulatorVSBinaryOpTest, add, sub, mul, div);

using namespace alps::accumulators;

/*
  for acc in Mean NoBinning LogBinning FullBinning; do
  for type in float double 'long double'; do
  echo "acc_vs_pair_gen<${acc}Accumulator,$type>,";
  done; echo; done
*/

typedef ::testing::Types<
    acc_vs_pair_gen<MeanAccumulator,float>,
    acc_vs_pair_gen<MeanAccumulator,double>,
    acc_vs_pair_gen<MeanAccumulator,long double>,

    acc_vs_pair_gen<NoBinningAccumulator,float>,
    acc_vs_pair_gen<NoBinningAccumulator,double>,
    acc_vs_pair_gen<NoBinningAccumulator,long double>,

    acc_vs_pair_gen<LogBinningAccumulator,float>,
    acc_vs_pair_gen<LogBinningAccumulator,double>,
    acc_vs_pair_gen<LogBinningAccumulator,long double>,

    acc_vs_pair_gen<FullBinningAccumulator,float>,
    acc_vs_pair_gen<FullBinningAccumulator,double>,
    acc_vs_pair_gen<FullBinningAccumulator,long double>
    > acc_types;

INSTANTIATE_TYPED_TEST_CASE_P(VSBinaryOpTest, AccumulatorVSBinaryOpTest, acc_types);
