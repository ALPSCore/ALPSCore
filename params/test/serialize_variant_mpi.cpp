/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file serialize_variant_mpi.cpp
    
    @brief Tests sending of boost::variant over MPI

    @note These are implementation details as of now 
*/

#include <alps/params/serialize_variant.hpp>
#include <alps/params/dict_types.hpp>
#include <gtest/gtest.h>
#include <alps/utilities/gtest_par_xml_output.hpp>
#include <vector>
#include <stdexcept>

#include <alps/utilities/mpi.hpp>
#include <alps/utilities/mpi_vector.hpp>
#include <alps/utilities/mpi_pair.hpp>
#include "./dict_values_test.hpp"

#include <cassert>

namespace alps { namespace mpi {
inline void broadcast(alps::mpi::communicator&, alps::params_ns::detail::None&, int) {
    std::cout << "Broadcasting None is no-op" << std::endl;
}

}}

/// Consumer class to send-broadcast an object via MPI
struct broadcast_sender {
    alps::mpi::communicator& comm_;
    int root_;

    broadcast_sender(alps::mpi::communicator& comm, int root) : comm_(comm), root_(root) {}

    template <typename T>
    void operator()(const T& val) {
        using alps::mpi::broadcast;
        assert(comm_.rank()==root_ && "Should be only called by broadcast root");
        broadcast(comm_, const_cast<T&>(val), root_);
    }
};

/// Producer class to receive an object broadcast via MPI
struct broadcast_receiver {
    int target_which;
    int which_count;
    alps::mpi::communicator& comm_;
    int root_;

    broadcast_receiver(int which, alps::mpi::communicator& comm, int root)
        : target_which(which), which_count(0), comm_(comm), root_(root)
    {}

    template <typename T>
    boost::optional<T> operator()(const T*)
    {
        using alps::mpi::broadcast;
        assert(comm_.rank()!=root_ && "Should NOT be called by broadcast root");
        boost::optional<T> ret;
        if (target_which==which_count) {
            T val;
            broadcast(comm_, val, root_);
            ret=val;
        }
        ++which_count;
        return ret;
    }
};


typedef alps::detail::variant_serializer<alps::params_ns::detail::dict_all_types,
                                         broadcast_sender, broadcast_receiver> var_serializer;
typedef var_serializer::variant_type variant_type;

namespace aptest=alps::params_ns::testing;
namespace apd=alps::params_ns::detail;

inline bool operator==(const apd::None&, const apd::None&) { return true; }

// parameterized over bound type
template <typename T>
class VarSerialTest : public ::testing::Test {
    T val1_, val2_;
    alps::mpi::communicator comm_;
    int root_;
  public:
    VarSerialTest()
        : val1_(aptest::data_trait<T>::get(true)),
          val2_(aptest::data_trait<T>::get(false)),
          comm_(),
          root_(0)
    { }

    void testBcast()
    {
        variant_type var1;
        var1=val1_;
        int which=var1.which();

        broadcast(comm_, which, root_);

        variant_type var2;
        if (comm_.rank()==root_) {
            broadcast_sender consumer(comm_, root_);
            var_serializer::consume(consumer, var1);
        } else {
            broadcast_receiver producer(which, comm_, root_);
            var2=var_serializer::produce(producer);
        }

        if (comm_.rank()!=root_) {
            ASSERT_EQ(var1.which(), var2.which());
            T actual=boost::get<T>(var2);
            EXPECT_EQ(val1_, actual);
        }
    }

};

typedef ::testing::Types<
    bool
    ,
    int
    ,
    long
    ,
    unsigned long
    ,
    double
    ,
    std::string
    ,
    std::vector<int>
    ,
    std::vector<bool>
    ,
    std::vector<std::string>
    // ,
    // std::pair<std::string, int>
    > MyTypes;

TYPED_TEST_CASE(VarSerialTest, MyTypes);

TYPED_TEST(VarSerialTest, testBcast) { this->testBcast(); }

int main(int argc, char** argv)
{
   alps::mpi::environment env(argc, argv, false);
   alps::gtest_par_xml_output tweak;
   tweak(alps::mpi::communicator().rank(), argc, argv);
   ::testing::InitGoogleTest(&argc, argv);
   
   return RUN_ALL_TESTS();
}    
