/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/testing/unique_file.hpp>
#include <alps/hdf5/archive.hpp>
#include <alps/hdf5/multi_array.hpp>
#include <boost/scoped_ptr.hpp>

#include <iostream>
#include <algorithm>

#include "gtest/gtest.h"

class TestHDF5Large : public ::testing::Test {
  public:
    std::string fname_;

    TestHDF5Large() {
        fname_=alps::testing::temporary_filename("hdf5_large_array.h5.");
    }

    ~TestHDF5Large() {
        if (!fname_.empty()) remove(fname_.c_str());
    }
};

namespace ah5=alps::hdf5;


namespace {
    template <typename T>
    struct not_equal_to {
        const T value;
        not_equal_to(T v) : value(v) {}
        bool operator()(T v) const { return value!=v; }
    };
}


TEST_F(TestHDF5Large, MultiarrayReadWrite)
{
    typedef float value_type;
    typedef boost::multi_array<value_type,2> data_type;
    typedef boost::scoped_ptr<data_type> ptr_type;
    const std::size_t dim=(1UL<<15);
    const std::size_t sz=(dim)*(dim+1);
    
    std::cout << "Allocating dim=0x" << std::hex << dim << std::dec 
              << " which is " << (double(sizeof(value_type)*sz/0x400UL)/0x400UL) << " MB"
              << std::endl;
    ptr_type array_ptr(new data_type(boost::extents[dim][dim+1]));
    std::fill_n(array_ptr->origin(), sz, 34.25);
    std::cout << "Allocation is done, writing..." << std::endl;

    ah5::archive ar(fname_,"am");
    ar["/array"] << *array_ptr;

    std::cout << "Writing is done, clearing..." << std::endl;
    std::fill_n(array_ptr->origin(), sz, 0.);
    

    std::cout << "Clearing is done, reading..." << std::endl;
    
    ar["/array"] >> *array_ptr;

    std::cout << "Reading is done, comparing..." << std::endl;
    const value_type* startptr=array_ptr->origin();
    const value_type* endptr=array_ptr->origin()+sz;
    const value_type* ptr=std::find_if(
        startptr,
        endptr,
        not_equal_to<value_type>(34.25));

    EXPECT_EQ(endptr,ptr) << "Garbage at offset " << (ptr - startptr);
}
