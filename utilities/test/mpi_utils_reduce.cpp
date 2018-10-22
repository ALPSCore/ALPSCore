/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <algorithm>

#include <alps/utilities/mpi.hpp>

#include <gtest/gtest.h>

#include "./test_utils.hpp"
#include <alps/utilities/gtest_par_xml_output.hpp>

/* Test MPI all_reduce for basic types */

namespace am=alps::mpi;

template <typename T, template<typename> class OP>
class Proxy {
  public:
    typedef T value_type;
    typedef OP<T> op_type;
};

template <typename P>
class MpiReduceScalarTest : public ::testing::Test {
  private:
    typedef typename P::value_type value_type;
    typedef typename P::op_type op_type;
    static const int ROOT_=0;

    am::communicator comm_;
    int myrank_;
    bool is_root_;

    // boost::container::vector<value_type> data_, data_copy_;
    value_type in_data_;
    value_type expected_data_;

    // make the number a bit more interesting than 0...nproc
    static int mangle(int i) {
        return 7*i+23;
    }

  public:
    MpiReduceScalarTest() : comm_(),
                            myrank_(comm_.rank()),
                            is_root_(ROOT_==myrank_)
                            // ,
                            // data_(comm_.size())
    {
        // Sanity check:
        if (comm_.size()==0) {
            ADD_FAILURE() << "Zero-sized communicator??";
            throw std::logic_error("Can't happen: zero-sized communicator.");
        }
        expected_data_=alps::testing::datapoint<value_type>::get(mangle(0));
        if (0==myrank_) in_data_=expected_data_;
        op_type op;
        for (int i=1; i<comm_.size(); ++i) {
            value_type v=alps::testing::datapoint<value_type>::get(mangle(i));
            if (i==myrank_) in_data_=v;
            expected_data_=op(expected_data_, v);
        }
    }

    void allreduce_by_pointer() {
        value_type out_data;
        am::all_reduce(comm_, &in_data_, 1, &out_data, op_type());

        EXPECT_EQ(expected_data_, out_data) << "Reduce operation failed";
    }

    void allreduce_wrong_length() {
        value_type ini=alps::testing::datapoint<value_type>::get(mangle(99));
        value_type out_data=ini;
        EXPECT_THROW(am::all_reduce(comm_, &in_data_, 0, &out_data, op_type()), std::invalid_argument);
        EXPECT_THROW(am::all_reduce(comm_, &in_data_, -1, &out_data, op_type()), std::invalid_argument);
        EXPECT_EQ(ini, out_data) << "Data damaged: exception safety violated";
    }

    void allreduce_in_place() {
        value_type out_data=in_data_;;
        EXPECT_THROW(am::all_reduce(comm_, &out_data, 1, &out_data, op_type()), std::invalid_argument);
        EXPECT_EQ(in_data_, out_data) << "Data damaged: exception safety violated";
    }

    void allreduce_by_ref() {
        value_type out_data;
        am::all_reduce(comm_, in_data_, out_data, op_type());
        EXPECT_EQ(expected_data_, out_data) << "Reduce operation failed";
    }

    void allreduce_by_assign() {
        value_type out_data = am::all_reduce(comm_, in_data_, op_type());
        EXPECT_EQ(expected_data_, out_data) << "Reduce operation failed";
    }

};

// Set up type-parametrized test harness:

TYPED_TEST_CASE_P(MpiReduceScalarTest);

TYPED_TEST_P(MpiReduceScalarTest, allreduceByPointer  ) { this->allreduce_by_pointer(); }
TYPED_TEST_P(MpiReduceScalarTest, allreduceWrongLength) { this->allreduce_wrong_length(); }
TYPED_TEST_P(MpiReduceScalarTest, allreduceInPlace    ) { this->allreduce_in_place(); }
TYPED_TEST_P(MpiReduceScalarTest, allreduceByRef      ) { this->allreduce_by_ref(); }
TYPED_TEST_P(MpiReduceScalarTest, allreduceByAssign   ) { this->allreduce_by_assign(); }

REGISTER_TYPED_TEST_CASE_P(MpiReduceScalarTest,
                           allreduceByPointer,
                           allreduceWrongLength,
                           allreduceInPlace,
                           allreduceByRef,
                           allreduceByAssign);


// Now test with std::plus

typedef ::testing::Types<Proxy<char,                  std::plus>,
                         Proxy<signed short int,      std::plus>,
                         Proxy<signed int,            std::plus>,
                         Proxy<signed long int,       std::plus>,
                         Proxy<signed long long int,  std::plus>,
                         Proxy<signed char,           std::plus>,
                         Proxy<unsigned char,         std::plus>,
                         Proxy<unsigned short int,    std::plus>,
                         Proxy<unsigned int,          std::plus>,
                         Proxy<unsigned long int,     std::plus>,
                         Proxy<unsigned long long int,std::plus>,
                         Proxy<float,                 std::plus>,
                         Proxy<double,                std::plus>,
                         Proxy<long double,           std::plus>
#ifdef ALPS_MPI_HAS_MPI_CXX_BOOL
                         ,Proxy<bool,                  std::plus>
#endif
#ifdef ALPS_MPI_HAS_MPI_CXX_FLOAT_COMPLEX
                         ,Proxy<std::complex<float>,   std::plus>
#endif
#ifdef ALPS_MPI_HAS_MPI_CXX_DOUBLE_COMPLEX
                         ,Proxy<std::complex<double>,  std::plus>
#endif                         
                        > MyPlusTestTypes;

INSTANTIATE_TYPED_TEST_CASE_P(MyPlus, MpiReduceScalarTest, MyPlusTestTypes);


// Now test with alps::mpi::maximum

typedef ::testing::Types<Proxy<char,                  alps::mpi::maximum>,
                         Proxy<signed short int,      alps::mpi::maximum>,
                         Proxy<signed int,            alps::mpi::maximum>,
                         Proxy<signed long int,       alps::mpi::maximum>,
                         Proxy<signed long long int,  alps::mpi::maximum>,
                         Proxy<signed char,           alps::mpi::maximum>,
                         Proxy<unsigned char,         alps::mpi::maximum>,
                         Proxy<unsigned short int,    alps::mpi::maximum>,
                         Proxy<unsigned int,          alps::mpi::maximum>,
                         Proxy<unsigned long int,     alps::mpi::maximum>,
                         Proxy<unsigned long long int,alps::mpi::maximum>,
                         Proxy<float,                 alps::mpi::maximum>,
                         Proxy<double,                alps::mpi::maximum>,
                         Proxy<long double,           alps::mpi::maximum>
#ifdef ALPS_MPI_HAS_MPI_CXX_BOOL
                         ,Proxy<bool,                 alps::mpi::maximum>
#endif
#ifdef ALPS_MPI_HAS_MPI_CXX_FLOAT_COMPLEX
                         ,Proxy<std::complex<float>,  alps::mpi::maximum>
#endif
#ifdef ALPS_MPI_HAS_MPI_CXX_DOUBLE_COMPLEX
                         ,Proxy<std::complex<double>, alps::mpi::maximum>
#endif                         
                        > MyMaxTestTypes;

INSTANTIATE_TYPED_TEST_CASE_P(MyMax, MpiReduceScalarTest, MyMaxTestTypes);



int main(int argc, char** argv)
{
    alps::mpi::environment env(argc, argv); // initializes MPI environment
    alps::gtest_par_xml_output tweak;
    tweak(alps::mpi::communicator().rank(), argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
