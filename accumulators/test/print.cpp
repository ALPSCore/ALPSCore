/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include "alps/accumulators.hpp"
#include "gtest/gtest.h"

#include "accumulator_generator.hpp"

using namespace alps::accumulators::testing;

/// Google Test Fixture: argument is Accumulator<Type>
template <typename A>
class AccumulatorPrintTest : public ::testing::Test {
    public:
    typedef AccResultGenerator<A,1000> acc_gen_type;
    typedef typename acc_gen_type::value_type value_type;
    acc_gen_type acc_gen;
};

using namespace alps::accumulators;

typedef std::vector<double> v_double;

/*
  for acc in Mean NoBinning LogBinning FullBinning; do
  for type in float double 'long double' v_double; do
  echo "${acc}Accumulator<$type>,"
  done; echo; done
*/


typedef ::testing::Types<
    MeanAccumulator<float>,
    MeanAccumulator<double>,
    MeanAccumulator<long double>,
    MeanAccumulator<v_double>,

    NoBinningAccumulator<float>,
    NoBinningAccumulator<double>,
    NoBinningAccumulator<long double>,
    NoBinningAccumulator<v_double>,

    LogBinningAccumulator<float>,
    LogBinningAccumulator<double>,
    LogBinningAccumulator<long double>,
    LogBinningAccumulator<v_double>,

    FullBinningAccumulator<float>,
    FullBinningAccumulator<double>,
    FullBinningAccumulator<long double>,
    FullBinningAccumulator<v_double>
    > test_types;

TYPED_TEST_CASE(AccumulatorPrintTest,test_types);

TYPED_TEST(AccumulatorPrintTest, print)
{
    const accumulator_wrapper& a=this->acc_gen.accumulator();
    const result_wrapper& r=this->acc_gen.result();

    std::cout << "Expected: " << this->acc_gen.expected_mean() << "+/-" << this->acc_gen.expected_err()
              << "\nAccumulator: " << a
              << "\nResult: " << r
              << std::endl;
    // std::cout << "\nFull print accumulator:\n" << a.fullprint
    //           << "\nFull print result:\n" << r.fullprint
    //           << std::endl;
}

/// Google Test Fixture: argument is data Type
template <typename T>
class AccumulatorCorrelatedPrintTest : public ::testing::Test {
    public:
    typedef acc_correlated_gen<T,5000,15> acc_gen_type;
    typedef T data_type;
    acc_gen_type acc_gen;
};

typedef ::testing::Types< float, double, long double> test_types2;
TYPED_TEST_CASE(AccumulatorCorrelatedPrintTest,test_types2);

TYPED_TEST(AccumulatorCorrelatedPrintTest, print)
{
    typedef typename TestFixture::data_type data_type;
    const accumulator_set& a=this->acc_gen.accumulators();
    const result_set& r=this->acc_gen.results();

    const char* names[]={"mean","nobin","logbin","fullbin"};
    const int nnames=sizeof(names)/sizeof(*names);
    std::cout << "Uncorrelated: " << this->acc_gen.expected_mean() << "+/-" << this->acc_gen.expected_uncorr_err() << "\n";
    for (int i=0; i<nnames; ++i) {
        std::string nm=names[i];
        std::cout << nm << " Accumulator: "
                  << a[nm].mean<data_type>();
        if (i!=0) {
            std::cout << "+/-" << a[nm].error<data_type>();
        }
        std::cout <<"\nRaw print:" << a[nm]<<"\n";
        std::cout <<"\nShort print:" << short_print(a[nm]) <<"\n";
        std::cout <<"\nFull print:" << full_print(a[nm]) <<"\n";

        std::cout << nm << " Result: "
                  << r[nm].mean<data_type>();
        if (i!=0) {
            std::cout << "+/-" << r[nm].error<data_type>();
        }
        std::cout << "\nRaw print:" << r[nm]<<"\n";
        std::cout << "\nShort print:" << short_print(r[nm])<<"\n";
        std::cout << "\nFull print:" << full_print(r[nm])<<"\n";
    }
}
