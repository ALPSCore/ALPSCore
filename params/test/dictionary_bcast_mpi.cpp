/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file dictionary_bcast_mpi.cpp
    
    @brief Tests broadcastign the dictionary over MPI
*/

#include <alps/params.hpp>
#include <gtest/gtest.h>
#include <alps/utilities/gtest_par_xml_output.hpp>

#include <alps/utilities/mpi.hpp>

#include <alps/testing/fp_compare.hpp>
#include "./dict_values_test.hpp"


using boost::integer_traits;
namespace ap=alps::params_ns;
namespace apt=ap::testing;
namespace de=ap::exception;
using ap::dictionary;

typedef std::vector<bool> boolvec;
typedef std::vector<int> intvec;
typedef std::vector<double> dblvec;
typedef std::vector<std::string> strvec;

// typedef std::pair<std::string,bool> boolpair;
// typedef std::pair<std::string,int> intpair;
// typedef std::pair<std::string,double> dblpair;
// typedef std::pair<std::string,std::string> stringpair;

template <typename T>
class DictionaryTestBCast : public ::testing::Test {
  protected:
    dictionary dict1_, dict2_;
    const dictionary& cdict1_;
    const dictionary& cdict2_;

    alps::mpi::communicator comm_;
    int root_;
  public:
    DictionaryTestBCast(): dict1_(),dict2_(), cdict1_(dict1_), cdict2_(dict2_), comm_(), root_(0) {

        dict1_["my_bool"]=apt::data_trait<bool>::get(true);
        dict1_["my_int"]=apt::data_trait<  int>::get(true);
        dict1_["my_unsigned long"]=apt::data_trait<unsigned long>::get(true);
        dict1_["my_float"]=apt::data_trait<float>::get(true);
        dict1_["my_double"]=apt::data_trait<  double>::get(true);
        dict1_["my_string"]=apt::data_trait<  std::string>::get(true);
        dict1_["my_boolvec"]=apt::data_trait<  boolvec>::get(true);
        dict1_["my_intvec"]=apt::data_trait<  intvec>::get(true);
        dict1_["my_dblvec"]=apt::data_trait<  dblvec>::get(true);
        dict1_["my_strvec"]=apt::data_trait<  strvec>::get(true);
        // dict1_["my_boolpair"]=apt::data_trait<  boolpair>::get(true);
        // dict1_["my_intpair"]=apt::data_trait<  intpair>::get(true);
        // dict1_["my_dblpair"]=apt::data_trait<  dblpair>::get(true);
        // dict1_["my_stringpair"]=apt::data_trait<  stringpair>::get(true);

        dict2_["my_bool"]=apt::data_trait<bool>::get(false);
        dict2_["my_int"]=apt::data_trait<  int>::get(false);
        dict2_["my_unsigned long"]=apt::data_trait<unsigned long>::get(false);
        dict2_["my_float"]=apt::data_trait<float>::get(false);
        dict2_["my_double"]=apt::data_trait<  double>::get(false);
        dict2_["my_string"]=apt::data_trait<  std::string>::get(false);
        dict2_["my_boolvec"]=apt::data_trait<  boolvec>::get(false);
        dict2_["my_intvec"]=apt::data_trait<  intvec>::get(false);
        dict2_["my_dblvec"]=apt::data_trait<  dblvec>::get(false);
        dict2_["my_strvec"]=apt::data_trait<  strvec>::get(false);
        // dict2_["my_boolpair"]=apt::data_trait<  boolpair>::get(false);
        // dict2_["my_intpair"]=apt::data_trait<  intpair>::get(false);
        // dict2_["my_dblpair"]=apt::data_trait<  dblpair>::get(false);
        // dict2_["my_stringpair"]=apt::data_trait<  stringpair>::get(false);
    }

    void testBCast() {
        using alps::mpi::broadcast;
        ASSERT_NE(dict1_, dict2_);

        const bool is_master=comm_.rank()==root_;
        dictionary& dict = *( is_master ? &dict1_ : &dict2_);
        if (is_master) {
            ASSERT_EQ(dict1_, dict);
        } else {
            ASSERT_EQ(dict2_, dict);
        }

        broadcast(comm_, dict, root_);

        EXPECT_EQ(dict1_, dict);
    }
};

typedef ::testing::Types<
    bool
    ,
    int
    ,
    unsigned long
    ,
    float
    ,
    double
    ,
    std::string
    ,
    boolvec
    ,
    intvec
    ,
    dblvec
    ,
    strvec
    // ,
    // boolpair
    // ,
    // intpair
    // ,
    // dblpair
    // ,
    // stringpair
    > MyTypes;

TYPED_TEST_CASE(DictionaryTestBCast, MyTypes);

TYPED_TEST(DictionaryTestBCast, testBCast) { this->testBCast(); }



int main(int argc, char** argv)
{
   alps::mpi::environment env(argc, argv, false);
   alps::gtest_par_xml_output tweak;
   tweak(alps::mpi::communicator().rank(), argc, argv);
   ::testing::InitGoogleTest(&argc, argv);
   
   return RUN_ALL_TESTS();
}    
