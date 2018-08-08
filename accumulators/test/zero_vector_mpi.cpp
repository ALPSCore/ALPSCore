/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** Test for MPI merge for zero-sized data vectors */

#include <alps/accumulators.hpp>
#include <alps/utilities/gtest_par_xml_output.hpp>
#include <gtest/gtest.h>

namespace aa=alps::accumulators;

// Proxy template to convey accumulator type A, data size etc.
template < template<typename> class A, typename T, std::size_t N>
class Proxy {
  public:
    typedef std::vector<T> value_type;
    typedef A<value_type> named_acc_type;
    static const std::size_t NPOINTS = N;
};

template <typename P>
class AccumulatorEmptyVecTest : public ::testing::Test {
  public:
    typedef typename P::named_acc_type named_acc_type;
    typedef typename P::value_type value_type;
    static const std::size_t NPOINTS = P::NPOINTS;
    static const int MASTER=0;

    named_acc_type acc_;
    alps::mpi::communicator comm_;

    AccumulatorEmptyVecTest(): acc_("data"), comm_() {}

    void add_data_test() {
        EXPECT_ANY_THROW(acc_ << value_type());
        std::shared_ptr<aa::result_wrapper> resptr=acc_.result();
        EXPECT_EQ(0u, resptr->count());
    }

    void merge_test() {
        if (comm_.rank()==MASTER) {
            for (unsigned int i=0; i<NPOINTS; ++i) {
                EXPECT_ANY_THROW(acc_ << value_type());
            }
        }
        EXPECT_ANY_THROW(acc_.collective_merge(comm_, MASTER));
    }

};


typedef std::vector<double> double_vec;

typedef ::testing::Types<
    Proxy<aa::MeanAccumulator, double, 100>,
    Proxy<aa::NoBinningAccumulator, double, 100>,
    Proxy<aa::LogBinningAccumulator, double, 100>,
    Proxy<aa::FullBinningAccumulator, double, 100>
    > test_types;

TYPED_TEST_CASE(AccumulatorEmptyVecTest, test_types);


TYPED_TEST(AccumulatorEmptyVecTest, addData) { this->add_data_test(); }
TYPED_TEST(AccumulatorEmptyVecTest, merge) { this->merge_test(); }

int main(int argc, char** argv)
{
   alps::mpi::environment env(argc, argv, false);
   alps::gtest_par_xml_output tweak;
   tweak(alps::mpi::communicator().rank(), argc, argv);
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}
