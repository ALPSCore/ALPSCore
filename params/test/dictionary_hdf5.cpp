/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file dictionary_hdf5.cpp
    
    @brief Tests saving/loading a dictionary to HDF5
*/

#include <alps/params.hpp>
#include <gtest/gtest.h>

#include <alps/testing/fp_compare.hpp>
#include "./dict_values_test.hpp"

#include <alps/testing/unique_file.hpp>

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

class DictionaryTestHdf5 : public ::testing::Test {
  protected:
    dictionary dict1_, dict2_;
    const dictionary& cdict1_;
    const dictionary& cdict2_;

    alps::testing::unique_file file_; 

  public:
    DictionaryTestHdf5(): dict1_(),dict2_(), cdict1_(dict1_), cdict2_(dict2_),
                          file_("dict_test.h5.", alps::testing::unique_file::REMOVE_AFTER/*KEEP_AFTER*/)
    {

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

};

TEST_F(DictionaryTestHdf5, testSave) {
    alps::hdf5::archive ar(file_.name(), "w");
    ar["dictionary1"] << dict1_;
    ar["dictionary2"] << dict2_;
}

TEST_F(DictionaryTestHdf5, testSaveEmptyDict) {
    alps::hdf5::archive ar(file_.name(), "w");
    dictionary empty_dict;
    ar["dictionary"] << empty_dict;
}

TEST_F(DictionaryTestHdf5, testSaveNone) {
    alps::hdf5::archive ar(file_.name(), "w");
    dict1_["empty_value"];
    ar["dictionary1"] << dict1_;
}

TEST_F(DictionaryTestHdf5, testLoadEmptyDict) {
    {
        alps::hdf5::archive ar(file_.name(), "w");
        dictionary empty_dict;
        ar["dict"] << empty_dict;
    }
    {
        alps::hdf5::archive ar(file_.name(), "r");
        ar["dict"] >> dict2_;
    }
    EXPECT_EQ(0u, dict2_.size());
}

TEST_F(DictionaryTestHdf5, testLoad) {
    {
        alps::hdf5::archive ar(file_.name(), "w");
        ar["dict"] << dict1_;
    }
    EXPECT_NE(dict1_,dict2_);
    dict2_["some_other_int"]=9999;
    {
        alps::hdf5::archive ar(file_.name(), "r");
        ar["dict"] >> dict2_;
    }
    EXPECT_FALSE(dict2_.exists("some_other_int"));
    EXPECT_EQ(dict1_,dict2_);
}

TEST_F(DictionaryTestHdf5, testLoadNone) {
    dict1_["empty_value"];
    {
        alps::hdf5::archive ar(file_.name(), "w");
        ar["dict"] << dict1_;
    }
    EXPECT_NE(dict1_,dict2_);
    {
        alps::hdf5::archive ar(file_.name(), "r");
        ar["dict"] >> dict2_;
    }
    dict1_.erase("empty_value");
    EXPECT_EQ(dict1_,dict2_);
}
