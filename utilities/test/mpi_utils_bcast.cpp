/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <algorithm>
#include <boost/scoped_array.hpp>

#include <alps/utilities/mpi.hpp>

#include <gtest/gtest.h>

#include "./test_utils.hpp"
#include <alps/utilities/gtest_par_xml_output.hpp>

/* Test MPI broadcasts for basic types */

namespace am=alps::mpi;

template <typename T>
class MpiBcastTest_base : public ::testing::Test {
  private:
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
            // must be trivially true by construction
            ASSERT_EQ(root_data,my_data) << "Can't happen: Data mess-up on root";
        } else {
            // must be true if get(true|false) works 
            ASSERT_NE(root_data,my_data) << "Data mess-up on slave: root data does not differ";
        }

        am::broadcast(comm_, my_data, ROOT_);

        EXPECT_EQ(root_data, my_data);
    }

    void bcast_array() { // works for vector<bool> too
        const std::size_t bufsz=41; // some "odd" number
        // margin large enough to catch data type size mismatches up to 16x without a crash:
        const std::size_t margin=bufsz*16+3; 
        const std::size_t vsize=2*margin+bufsz; // vector is buffer surrounded by margins
        
        vector_type root_data=alps::testing::datapoint<vector_type>::get(true,vsize);
        vector_type slave_data=alps::testing::datapoint<vector_type>::get(false,vsize);

        boost::scoped_array<value_type> my_data(new value_type[vsize]);
        if (is_root_) {
            std::copy(root_data.begin(), root_data.end(), my_data.get());
        } else {
            std::copy(slave_data.begin(), slave_data.end(), my_data.get());
        }

        // broadcast the middle of the vector only
        am::broadcast(comm_, my_data.get()+margin, bufsz, ROOT_);

        typedef typename vector_type::const_iterator iter_type;
        if (is_root_) {
            // data on root should not change
            EXPECT_TRUE(std::equal(root_data.begin(), root_data.end(), my_data.get())) << "Data changed on root";
        } else {
            // data on slave should change in the buffer only
            const T* left_margin_beg=my_data.get();
            iter_type left_margin_beg0=slave_data.begin();

            const T* buffer_beg=left_margin_beg+margin;
            iter_type buffer_beg0=root_data.begin()+margin;
            
            const T* right_margin_beg=buffer_beg+bufsz;
            iter_type right_margin_beg0=slave_data.end()-margin;
            
            EXPECT_TRUE(std::equal(left_margin_beg, buffer_beg, left_margin_beg0)) << "Left margin is damaged on slave";
            EXPECT_TRUE(std::equal(buffer_beg, right_margin_beg, buffer_beg0)) << "Broadcast data incorrect";
            EXPECT_TRUE(std::equal(right_margin_beg, right_margin_beg+margin, right_margin_beg0)) << "Right margin is damaged on slave";
        }
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
    alps::gtest_par_xml_output tweak;
    tweak(alps::mpi::communicator().rank(), argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
