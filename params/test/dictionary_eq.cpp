/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file dictionary_eq.cpp
    
    @brief Tests the equality/inequality of dictionaries
*/

#include <alps/params.hpp>
#include <gtest/gtest.h>

#include <alps/testing/fp_compare.hpp>
#include "./dict_values_test.hpp"
//#include <boost/integer_traits.hpp>

using boost::integer_traits;
namespace ap=alps::params_ns;
namespace apt=ap::testing;
namespace de=ap::exception;
using ap::dictionary;

class DictionaryTestEq : public ::testing::Test {
  protected:
    dictionary dict1_, dict2_;
    const dictionary& cdict1_;
    const dictionary& cdict2_;
  public:
    DictionaryTestEq(): dict1_(),dict2_(), cdict1_(dict1_), cdict2_(dict2_) {}
};

TEST_F(DictionaryTestEq, Empty) {
    EXPECT_TRUE(cdict1_==cdict1_);
    EXPECT_FALSE(cdict1_!=cdict1_);
    EXPECT_TRUE(cdict1_==cdict2_);
    EXPECT_FALSE(cdict1_!=cdict2_);
}
    
TEST_F(DictionaryTestEq, DiffSize) {
    dict1_["one"]=1;
    EXPECT_FALSE(cdict1_==cdict2_);
    EXPECT_TRUE(cdict1_!=cdict2_);

    dict2_["one"]=1;
    dict1_["two"]=2;
    EXPECT_FALSE(cdict1_==cdict2_);
    EXPECT_TRUE(cdict1_!=cdict2_);
}

TEST_F(DictionaryTestEq, DiffNames) {
    dict1_["one"]=1;
    dict2_["another_one"]=1;
    EXPECT_FALSE(cdict1_==cdict2_);
    EXPECT_TRUE(cdict1_!=cdict2_);
}

TEST_F(DictionaryTestEq, EmptyValues) {
    dict1_["one"];
    EXPECT_FALSE(cdict1_==cdict2_);
    dict2_["one"];
    EXPECT_TRUE(cdict1_==cdict2_);
}

TEST_F(DictionaryTestEq, SameTypeDiffValues) {
    dict1_["one"]=1;
    dict2_["one"]=2;
    EXPECT_FALSE(cdict1_==cdict2_);
    dict1_["string"]="string1";
    dict2_["string"]="string2";
    EXPECT_FALSE(cdict1_==cdict2_);
}

TEST_F(DictionaryTestEq, SameTypeSameValues) {
    dict1_["one"]=1;
    dict2_["one"]=1;
    EXPECT_TRUE(cdict1_==cdict2_);
    dict1_["string"]="string";
    dict2_["string"]="string";
    EXPECT_TRUE(cdict1_==cdict2_);
}

TEST_F(DictionaryTestEq, SameValuesDiffType) {
    dict1_["one"]=1;
    dict2_["one"]=1L;
    EXPECT_FALSE(cdict1_==cdict2_);
    dict2_["one"]=1;
    EXPECT_TRUE(cdict1_==cdict2_);
}
