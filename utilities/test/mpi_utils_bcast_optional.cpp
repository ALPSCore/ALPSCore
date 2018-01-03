/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/utilities/mpi_optional.hpp>
#include <boost/optional/optional_io.hpp> // to make gtest happy with printing optionals
#include <gtest/gtest.h>

#include "./test_utils.hpp"
#include <alps/utilities/gtest_par_xml_output.hpp>

/* Test MPI broadcasts for boost::optional<T> */

namespace am=alps::mpi;

template <typename T>
class MpiBcastOptionalTest: public ::testing::Test {
  private:
    typedef T value_type;
    typedef boost::optional<T> optional_type;
    static const int ROOT_=0;
    am::communicator comm_;
    int myrank_;
    bool is_root_;
  public:
    MpiBcastOptionalTest() : comm_(),
                             myrank_(comm_.rank()),
                             is_root_(ROOT_==myrank_)
    {}

    void bcast() {
        value_type root_data=alps::testing::datapoint<value_type>::get(true);
        value_type slave_data=alps::testing::datapoint<value_type>::get(false);
        value_type& my_data=*(this->is_root_? &root_data : &slave_data);
        optional_type opt(my_data);

        // Sanity check:
        if (this->is_root_) {
            // must be trivially true by construction
            ASSERT_EQ(root_data,my_data) << "Can't happen: Data mess-up on root";
        } else {
            // must be true if get(true|false) works 
            ASSERT_NE(root_data,my_data) << "Data mess-up on slave: root data does not differ";
        }

        am::broadcast(comm_, opt, ROOT_);

        EXPECT_TRUE(!!opt) << "Optional value is uninitialized on rank " << myrank_;
        EXPECT_EQ(root_data, *opt);
    }

    void bcast_to_none() {
        value_type root_data=alps::testing::datapoint<value_type>::get(true);
        optional_type opt(boost::none);
        if (this->is_root_) {
            opt=root_data;
        }

        // Sanity check:
        if (this->is_root_) {
            // must be trivially true by construction
            ASSERT_TRUE(!!opt);
            ASSERT_EQ(root_data,*opt) << "Can't happen: Data mess-up on root";
        } else {
            // must be trivially true by construction
            ASSERT_FALSE(!!opt) << "Can't happen: Data mess-up on slave";
        }

        am::broadcast(comm_, opt, ROOT_);

        EXPECT_TRUE(!!opt) << "Optional value is uninitialized on rank " << myrank_;
        EXPECT_EQ(root_data, *opt);
    }

    void bcast_from_none() {
        value_type slave_data=alps::testing::datapoint<value_type>::get(false);
        optional_type opt=boost::none;
        if (!this->is_root_) {
            opt=slave_data;
        }

        // Sanity check:
        if (this->is_root_) {
            // must be trivially true by construction
            ASSERT_FALSE(!!opt) << "Can't happen: Data mess-up on root";
        } else {
            // must be trivially true by construction
            ASSERT_EQ(slave_data, *opt) << "Can't happen: Data mess-up on slave";
        }

        am::broadcast(comm_, opt, ROOT_);

        EXPECT_FALSE(!!opt) << "Optional value is (erroneously) initialized on rank " << myrank_;
    }
};

typedef ::testing::Types<bool,
                         char,
                         signed short int,
                         signed int,
                         signed long int,
                         signed long long int,
                         signed char,
                         unsigned char,
                         unsigned short int,
                         unsigned int,
                         unsigned long int,
                         unsigned long long int,
                         float,
                         double,
                         long double,
                         std::string,
                         std::complex<float>,
                         std::complex<double>
                        > MyTestTypes;

TYPED_TEST_CASE(MpiBcastOptionalTest, MyTestTypes);


TYPED_TEST(MpiBcastOptionalTest, Bcast) { this->bcast(); }
TYPED_TEST(MpiBcastOptionalTest, BcastToNone) { this->bcast_to_none(); }
TYPED_TEST(MpiBcastOptionalTest, BcastFromNone) { this->bcast_from_none(); }


   
int main(int argc, char** argv)
{
    alps::mpi::environment env(argc, argv); // initializes MPI environment
    alps::gtest_par_xml_output tweak;
    tweak(alps::mpi::communicator().rank(), argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
