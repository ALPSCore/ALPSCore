/*
 * Copyright (C) 1998-2016 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <algorithm> // for std::min, std::copy
#include <boost/scoped_array.hpp>

#include <alps/utilities/mpi.hpp>
#include <gtest/gtest.h>

#include "./test_utils.hpp"

/* Test MPI broadcasts for basic types */

namespace am=alps::mpi;

template <typename T>
class MpiBcastTest_base : public ::testing::Test {
  protected:
    typedef std::vector<T> vector_type;
    typedef T value_type;
    static const int ROOT_=0;
    am::communicator comm_;
    int myrank_;
    bool is_root_;
  public:
    MpiBcastTest_base() : comm_(),
                          myrank_(comm_.rank()),
                          is_root_(ROOT_==myrank_)
    {}

    void bcast_scalar() {
        value_type root_data=alps::testing::datapoint<value_type>::get(true);
        value_type slave_data=alps::testing::datapoint<value_type>::get(false);
        value_type& my_data=*(this->is_root_? &root_data : &slave_data);

        // Sanity check:
        if (this->is_root_) {
            ASSERT_EQ(root_data,my_data) << "Data mess-up on root";
        } else {
            ASSERT_NE(root_data,my_data) << "Data mess-up on slave";
        }
    
        am::broadcast(this->comm_, my_data, this->ROOT_);

        EXPECT_EQ(root_data, my_data);
    }

    void bcast_array() {
        vector_type root_data=alps::testing::datapoint<vector_type>::get(true);
        vector_type slave_data=alps::testing::datapoint<vector_type>::get(false);
        std::size_t sz=std::min(root_data.size(), slave_data.size());
        root_data.resize(sz);
        slave_data.resize(sz);
        vector_type& my_data=*(this->is_root_? &root_data : &slave_data);

        // Sanity check:
        if (this->is_root_) {
            ASSERT_EQ(root_data,my_data) << "Data mess-up on root";
        } else {
            ASSERT_NE(root_data,my_data) << "Data mess-up on slave";
        }
    
        am::broadcast(this->comm_, &my_data[0], sz, this->ROOT_);

        EXPECT_EQ(root_data, my_data);
    }        
};

template <typename T>
class MpiBcastTest: public MpiBcastTest_base<T> {};

// specialization for T=std::string (because broadcating vector<string> is not implemented)
// FIXME: change this once it is implemented!
template <>
class MpiBcastTest<std::string>: public MpiBcastTest_base<std::string> {
  public:
    // redefinition of bcast_array
    void bcast_array() {
        std::cout << "Broadcast of std::vector<std::string> is NOT IMPLEMENTED."
                  << std::endl;
    }
};

// specialization for T=bool (because vector<bool> is special)
template <>
class MpiBcastTest<bool>: public MpiBcastTest_base<bool> {
  public:
    // redefinition of bcast_array
    void bcast_array() {
        vector_type root_data=alps::testing::datapoint<vector_type>::get(true);
        vector_type slave_data=alps::testing::datapoint<vector_type>::get(false);
        std::size_t sz=std::min(root_data.size(), slave_data.size());
        root_data.resize(sz);
        slave_data.resize(sz);
        boost::scoped_array<value_type> my_data(new value_type[sz]);
        if (is_root_) {
            std::copy(root_data.begin(), root_data.end(), my_data.get());
        }
        else {
            std::copy(slave_data.begin(), slave_data.end(), my_data.get());
        }
        // Sanity check:
        if (is_root_) {
            ASSERT_TRUE(std::equal(root_data.begin(), root_data.end(), my_data.get())) << "Data mess-up on root";
        } else {
            ASSERT_FALSE(std::equal(root_data.begin(), root_data.end(), my_data.get())) << "Data mess-up on slave";
        }
    
        am::broadcast(this->comm_, my_data.get(), sz, this->ROOT_);

        EXPECT_TRUE(std::equal(root_data.begin(), root_data.end(), my_data.get()));
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

TYPED_TEST_CASE(MpiBcastTest, MyTestTypes);


TYPED_TEST(MpiBcastTest, BcastScalar) { this->bcast_scalar(); }
TYPED_TEST(MpiBcastTest, BcastArray) { this->bcast_array(); }


int main(int argc, char** argv)
{
    alps::mpi::environment env(argc, argv); // initializes MPI environment
    // alps::gtest_par_xml_output tweak;
    // tweak(alps::mpi::communicator().rank(), argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
