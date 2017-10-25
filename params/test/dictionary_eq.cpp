/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file dictionary_eq.cpp
    
    @brief Tests the equality/inequality of dictionary values
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
    dictionary dict_;
    const dictionary& cdict_;

    static const long neg_long= integer_traits<long>::const_min+5;
    static const int  neg_int=  integer_traits<int>::const_min+7;
    static const int  pos_int=  integer_traits<int>::const_max-7;
    static const long pos_long= integer_traits<long>::const_max-5;
    
    static const long neg_long1= integer_traits<long>::const_min+6;
    static const int  neg_int1=  integer_traits<int>::const_min+8;
    static const int  pos_int1=  integer_traits<int>::const_max-6;
    static const long pos_long1= integer_traits<long>::const_max-4;
    
    static const unsigned int pos_uint=   integer_traits<unsigned int>::const_max-9;
    static const unsigned long pos_ulong= integer_traits<unsigned long>::const_max-10;

    static const unsigned int pos_uint1=   integer_traits<unsigned int>::const_max-8;
    static const unsigned long pos_ulong1= integer_traits<unsigned long>::const_max-9;

    // FIXME: consider this too
    // static const bool long_is_int=(sizeof(long)==sizeof(int));

    // suffix "_is" means "integer size" (that is, value small enough to fit into signed integer)
    // suffix "_uis" means "unsigned integer size"
    // suffix "_ls" means "long size"
        
    static const long neg_long_is=static_cast<long>(+neg_int);
    static const long neg_long_is1=static_cast<long>(+neg_int1);

    static const unsigned int uint_is=static_cast<unsigned int>(+pos_int);
    static const long pos_long_is=static_cast<long>(+pos_int);
    static const unsigned long ulong_is=static_cast<unsigned long>(+pos_int);
    
    static const unsigned int uint_is1=static_cast<unsigned int>(+pos_int1);
    static const long pos_long_is1=static_cast<long>(+pos_int1);
    static const unsigned long ulong_is1=static_cast<unsigned long>(+pos_int1);

    static const unsigned long ulong_ls=static_cast<unsigned long>(+pos_long);
    static const unsigned long ulong_ls1=static_cast<unsigned long>(+pos_long1);

    bool my_bool, my_bool1;
    int my_int, my_int1;
    std::string my_string, my_string1;
    std::vector<int> my_vec, my_vec1;
    std::pair<std::string, int> my_pair, my_pair1;
    
    public:
    DictionaryTestEq(): dict_(), cdict_(dict_) {
        dict_["neg_long"]= +neg_long;
        dict_["neg_long1"]= +neg_long1;

        dict_["neg_int"]= +neg_int;
        dict_["neg_long_is"]= static_cast<long>(+neg_int);
        dict_["neg_int1"]= +neg_int1;
        dict_["neg_long_is1"]= static_cast<long>(+neg_int1);
        
        dict_["pos_int"]= +pos_int;
        dict_["uint_is"]= static_cast<unsigned int>(+pos_int);
        dict_["pos_long_is"]= static_cast<long>(+pos_int);
        dict_["ulong_is"]= static_cast<unsigned long>(+pos_int);
        
        dict_["pos_int1"]= +pos_int1;
        dict_["uint_is1"]= static_cast<unsigned int>(+pos_int1);
        dict_["pos_long_is1"]= static_cast<long>(+pos_int1);
        dict_["ulong_is1"]= static_cast<unsigned long>(+pos_int1);
        
        dict_["pos_uint"]= +pos_uint;
        dict_["pos_long_uis"]= static_cast<long>(+pos_uint);
        dict_["ulong_uis"]= static_cast<unsigned long>(+pos_uint);

        dict_["pos_uint1"]= +pos_uint1;
        dict_["pos_long_uis1"]= static_cast<long>(+pos_uint1);
        dict_["ulong_uis1"]= static_cast<unsigned long>(+pos_uint1);

        dict_["pos_long"]= +pos_long;
        dict_["ulong_ls"]= static_cast<unsigned long>(+pos_long);

        dict_["pos_long1"]= +pos_long1;
        dict_["ulong_ls1"]= static_cast<unsigned long>(+pos_long1);

        dict_["pos_ulong"]= +pos_ulong;
        dict_["pos_ulong1"]= +pos_ulong1;

        my_bool=apt::data_trait< bool >::get(true);
        dict_["my_bool"]=my_bool;
        my_bool1=apt::data_trait< bool >::get(false);
        dict_["my_bool1"]=my_bool1;
        my_int=apt::data_trait< int >::get(true);
        dict_["my_int"]=my_int;
        my_int1=apt::data_trait< int >::get(false);
        dict_["my_int1"]=my_int1;
        my_string=apt::data_trait< std::string >::get(true);
        dict_["my_string"]=my_string;
        my_string1=apt::data_trait< std::string >::get(false);
        dict_["my_string1"]=my_string1;
        my_vec=apt::data_trait< std::vector<int> >::get(true);
        dict_["my_vec"]=my_vec;
        my_vec1=apt::data_trait< std::vector<int> >::get(false);
        dict_["my_vec1"]=my_vec1;
        my_pair=apt::data_trait< std::pair<std::string, int> >::get(true);
        dict_["my_pair"]=my_pair;
        my_pair1=apt::data_trait< std::pair<std::string, int> >::get(false);
        dict_["my_pair1"]=my_pair1;
    }
};

// TEST_F(DictionaryTestEq, Test) {
//     EXPECT_ANY_THROW(cdict_["dummy"]==true);
//     EXPECT_ANY_THROW(true==cdict_["dummy"]);

//     EXPECT_ANY_THROW(cdict_["dummy"]==int(1));
//     EXPECT_ANY_THROW(cdict_["dummy"]==long(1));
//     EXPECT_ANY_THROW(cdict_["dummy"]==1.0f);
//     EXPECT_ANY_THROW(cdict_["dummy"]==1.0);
//     EXPECT_ANY_THROW(cdict_["dummy"]==std::string());

    
// }

TEST_F(DictionaryTestEq, eqDouble) {
    EXPECT_TRUE(  cdict_["my_int"]==double(my_int) );
    EXPECT_FALSE( cdict_["my_int"]!=double(my_int) );
    EXPECT_TRUE(  double(my_int)==cdict_["my_int"]);
    EXPECT_FALSE( double(my_int)!=cdict_["my_int"]);

    dict_["my_double"]=double(my_int);
    
    EXPECT_TRUE(  cdict_["my_double"]==double(my_int) );
    EXPECT_FALSE( cdict_["my_double"]!=double(my_int) );
    EXPECT_TRUE(  double(my_int)==cdict_["my_double"] );
    EXPECT_FALSE( double(my_int)!=cdict_["my_double"] );
    
    EXPECT_TRUE(  cdict_["my_double"]==my_int );
    EXPECT_FALSE( cdict_["my_double"]!=my_int );
    EXPECT_TRUE(  my_int==cdict_["my_double"] );
    EXPECT_FALSE( my_int!=cdict_["my_double"] );

    dict_["my_double0"]=double(my_int);
    dict_["my_double1"]=double(my_int)+1.0;

    EXPECT_TRUE(  dict_["my_double"]==dict_["my_double0"] );
    EXPECT_FALSE( dict_["my_double"]!=dict_["my_double0"] );
    
    EXPECT_TRUE(  dict_["my_double"]!=dict_["my_double1"] );
    EXPECT_FALSE( dict_["my_double"]==dict_["my_double1"] );
}

TEST_F(DictionaryTestEq, eqFloat) {
    EXPECT_TRUE(  cdict_["my_int"]==float(my_int) );
    EXPECT_FALSE( cdict_["my_int"]!=float(my_int) );
    EXPECT_TRUE(  float(my_int)==cdict_["my_int"]);
    EXPECT_FALSE( float(my_int)!=cdict_["my_int"]);

    dict_["my_float"]=float(my_int);
    
    EXPECT_TRUE(  cdict_["my_float"]==float(my_int) );
    EXPECT_FALSE( cdict_["my_float"]!=float(my_int) );
    EXPECT_TRUE(  float(my_int)==cdict_["my_float"] );
    EXPECT_FALSE( float(my_int)!=cdict_["my_float"] );
    
    EXPECT_TRUE(  cdict_["my_float"]==my_int );
    EXPECT_FALSE( cdict_["my_float"]!=my_int );
    EXPECT_TRUE(  my_int==cdict_["my_float"] );
    EXPECT_FALSE( my_int!=cdict_["my_float"] );

    dict_["my_float0"]=float(my_int);
    dict_["my_float1"]=float(my_int)+1.0;

    EXPECT_TRUE(  dict_["my_float"]==dict_["my_float0"] );
    EXPECT_FALSE( dict_["my_float"]!=dict_["my_float0"] );
    
    EXPECT_TRUE(  dict_["my_float"]!=dict_["my_float1"] );
    EXPECT_FALSE( dict_["my_float"]==dict_["my_float1"] );
}



/* *** Script-generated code follows *** */

// Equalities within domain neg_long
TEST_F(DictionaryTestEq, eqNegLongLeft) {
    // Same values:
    EXPECT_TRUE(  cdict_["neg_long"]==+neg_long );
    EXPECT_FALSE( cdict_["neg_long"]!=+neg_long );
    // Different values:
    EXPECT_TRUE(  cdict_["neg_long"]!=+neg_long1 );
    EXPECT_FALSE( cdict_["neg_long"]==+neg_long1 );
}

TEST_F(DictionaryTestEq, eqNegLongRight) {
    // Same values:
    EXPECT_TRUE(  +neg_long==cdict_["neg_long"] );
    EXPECT_FALSE( +neg_long!=cdict_["neg_long"] );
    // Different values:
    EXPECT_TRUE(  +neg_long!=cdict_["neg_long1"] );
    EXPECT_FALSE( +neg_long==cdict_["neg_long1"] );
}

TEST_F(DictionaryTestEq, eqNegLongBoth) {
    // Same values:
    EXPECT_TRUE(  cdict_["neg_long"]==cdict_["neg_long"] );
    EXPECT_FALSE( cdict_["neg_long"]!=cdict_["neg_long"] );
    // Different values:
    EXPECT_TRUE(  cdict_["neg_long"]!=cdict_["neg_long1"] );
    EXPECT_FALSE( cdict_["neg_long"]==cdict_["neg_long1"] );
}

// Equalities between domains neg_long:neg_int
TEST_F(DictionaryTestEq, eqNegLongNegIntLeft) {
    EXPECT_TRUE(  cdict_["neg_long"]!=+neg_int );
    EXPECT_FALSE( cdict_["neg_long"]==+neg_int );
    EXPECT_TRUE(  cdict_["neg_long"]!=+neg_long_is );
    EXPECT_FALSE( cdict_["neg_long"]==+neg_long_is );
}

TEST_F(DictionaryTestEq, eqNegLongNegIntRight) {
    EXPECT_TRUE(  +neg_long!=cdict_["neg_int"] );
    EXPECT_FALSE( +neg_long==cdict_["neg_int"] );
    EXPECT_TRUE(  +neg_long!=cdict_["neg_long_is"] );
    EXPECT_FALSE( +neg_long==cdict_["neg_long_is"] );
}

TEST_F(DictionaryTestEq, eqNegLongNegIntBoth) {
    EXPECT_TRUE(  cdict_["neg_long"]!=cdict_["neg_int"] );
    EXPECT_FALSE( cdict_["neg_long"]==cdict_["neg_int"] );
    EXPECT_TRUE(  cdict_["neg_long"]!=cdict_["neg_long_is"] );
    EXPECT_FALSE( cdict_["neg_long"]==cdict_["neg_long_is"] );
}

// Equalities between domains neg_long:pos_int
TEST_F(DictionaryTestEq, eqNegLongPosIntLeft) {
    EXPECT_TRUE(  cdict_["neg_long"]!=+pos_int );
    EXPECT_FALSE( cdict_["neg_long"]==+pos_int );
    EXPECT_TRUE(  cdict_["neg_long"]!=+uint_is );
    EXPECT_FALSE( cdict_["neg_long"]==+uint_is );
    EXPECT_TRUE(  cdict_["neg_long"]!=+pos_long_is );
    EXPECT_FALSE( cdict_["neg_long"]==+pos_long_is );
    EXPECT_TRUE(  cdict_["neg_long"]!=+ulong_is );
    EXPECT_FALSE( cdict_["neg_long"]==+ulong_is );
}

TEST_F(DictionaryTestEq, eqNegLongPosIntRight) {
    EXPECT_TRUE(  +neg_long!=cdict_["pos_int"] );
    EXPECT_FALSE( +neg_long==cdict_["pos_int"] );
    EXPECT_TRUE(  +neg_long!=cdict_["uint_is"] );
    EXPECT_FALSE( +neg_long==cdict_["uint_is"] );
    EXPECT_TRUE(  +neg_long!=cdict_["pos_long_is"] );
    EXPECT_FALSE( +neg_long==cdict_["pos_long_is"] );
    EXPECT_TRUE(  +neg_long!=cdict_["ulong_is"] );
    EXPECT_FALSE( +neg_long==cdict_["ulong_is"] );
}

TEST_F(DictionaryTestEq, eqNegLongPosIntBoth) {
    EXPECT_TRUE(  cdict_["neg_long"]!=cdict_["pos_int"] );
    EXPECT_FALSE( cdict_["neg_long"]==cdict_["pos_int"] );
    EXPECT_TRUE(  cdict_["neg_long"]!=cdict_["uint_is"] );
    EXPECT_FALSE( cdict_["neg_long"]==cdict_["uint_is"] );
    EXPECT_TRUE(  cdict_["neg_long"]!=cdict_["pos_long_is"] );
    EXPECT_FALSE( cdict_["neg_long"]==cdict_["pos_long_is"] );
    EXPECT_TRUE(  cdict_["neg_long"]!=cdict_["ulong_is"] );
    EXPECT_FALSE( cdict_["neg_long"]==cdict_["ulong_is"] );
}

// Equalities between domains neg_long:pos_uint
TEST_F(DictionaryTestEq, eqNegLongPosUintLeft) {
    EXPECT_TRUE(  cdict_["neg_long"]!=+pos_uint );
    EXPECT_FALSE( cdict_["neg_long"]==+pos_uint );
}

TEST_F(DictionaryTestEq, eqNegLongPosUintRight) {
    EXPECT_TRUE(  +neg_long!=cdict_["pos_uint"] );
    EXPECT_FALSE( +neg_long==cdict_["pos_uint"] );
}

TEST_F(DictionaryTestEq, eqNegLongPosUintBoth) {
    EXPECT_TRUE(  cdict_["neg_long"]!=cdict_["pos_uint"] );
    EXPECT_FALSE( cdict_["neg_long"]==cdict_["pos_uint"] );
}

// Equalities between domains neg_long:pos_long
TEST_F(DictionaryTestEq, eqNegLongPosLongLeft) {
    EXPECT_TRUE(  cdict_["neg_long"]!=+pos_long );
    EXPECT_FALSE( cdict_["neg_long"]==+pos_long );
    EXPECT_TRUE(  cdict_["neg_long"]!=+ulong_ls );
    EXPECT_FALSE( cdict_["neg_long"]==+ulong_ls );
}

TEST_F(DictionaryTestEq, eqNegLongPosLongRight) {
    EXPECT_TRUE(  +neg_long!=cdict_["pos_long"] );
    EXPECT_FALSE( +neg_long==cdict_["pos_long"] );
    EXPECT_TRUE(  +neg_long!=cdict_["ulong_ls"] );
    EXPECT_FALSE( +neg_long==cdict_["ulong_ls"] );
}

TEST_F(DictionaryTestEq, eqNegLongPosLongBoth) {
    EXPECT_TRUE(  cdict_["neg_long"]!=cdict_["pos_long"] );
    EXPECT_FALSE( cdict_["neg_long"]==cdict_["pos_long"] );
    EXPECT_TRUE(  cdict_["neg_long"]!=cdict_["ulong_ls"] );
    EXPECT_FALSE( cdict_["neg_long"]==cdict_["ulong_ls"] );
}

// Equalities between domains neg_long:pos_ulong
TEST_F(DictionaryTestEq, eqNegLongULongLeft) {
    EXPECT_TRUE(  cdict_["neg_long"]!=+pos_ulong );
    EXPECT_FALSE( cdict_["neg_long"]==+pos_ulong );
}

TEST_F(DictionaryTestEq, eqNegLongULongRight) {
    EXPECT_TRUE(  +neg_long!=cdict_["pos_ulong"] );
    EXPECT_FALSE( +neg_long==cdict_["pos_ulong"] );
}

TEST_F(DictionaryTestEq, eqNegLongULongBoth) {
    EXPECT_TRUE(  cdict_["neg_long"]!=cdict_["pos_ulong"] );
    EXPECT_FALSE( cdict_["neg_long"]==cdict_["pos_ulong"] );
}

// Equalities within domain neg_int
TEST_F(DictionaryTestEq, eqNegIntLeft) {
    // Same values:
    EXPECT_TRUE(  cdict_["neg_int"]==+neg_int );
    EXPECT_FALSE( cdict_["neg_int"]!=+neg_int );
    // Different values:
    EXPECT_TRUE(  cdict_["neg_int"]!=+neg_int1 );
    EXPECT_FALSE( cdict_["neg_int"]==+neg_int1 );
    // Same values:
    EXPECT_TRUE(  cdict_["neg_int"]==+neg_long_is );
    EXPECT_FALSE( cdict_["neg_int"]!=+neg_long_is );
    // Different values:
    EXPECT_TRUE(  cdict_["neg_int"]!=+neg_long_is1 );
    EXPECT_FALSE( cdict_["neg_int"]==+neg_long_is1 );
    // Same values:
    EXPECT_TRUE(  cdict_["neg_long_is"]==+neg_long_is );
    EXPECT_FALSE( cdict_["neg_long_is"]!=+neg_long_is );
    // Different values:
    EXPECT_TRUE(  cdict_["neg_long_is"]!=+neg_long_is1 );
    EXPECT_FALSE( cdict_["neg_long_is"]==+neg_long_is1 );
}

TEST_F(DictionaryTestEq, eqNegIntRight) {
    // Same values:
    EXPECT_TRUE(  +neg_int==cdict_["neg_int"] );
    EXPECT_FALSE( +neg_int!=cdict_["neg_int"] );
    // Different values:
    EXPECT_TRUE(  +neg_int!=cdict_["neg_int1"] );
    EXPECT_FALSE( +neg_int==cdict_["neg_int1"] );
    // Same values:
    EXPECT_TRUE(  +neg_int==cdict_["neg_long_is"] );
    EXPECT_FALSE( +neg_int!=cdict_["neg_long_is"] );
    // Different values:
    EXPECT_TRUE(  +neg_int!=cdict_["neg_long_is1"] );
    EXPECT_FALSE( +neg_int==cdict_["neg_long_is1"] );
    // Same values:
    EXPECT_TRUE(  +neg_long_is==cdict_["neg_long_is"] );
    EXPECT_FALSE( +neg_long_is!=cdict_["neg_long_is"] );
    // Different values:
    EXPECT_TRUE(  +neg_long_is!=cdict_["neg_long_is1"] );
    EXPECT_FALSE( +neg_long_is==cdict_["neg_long_is1"] );
}

TEST_F(DictionaryTestEq, eqNegIntBoth) {
    // Same values:
    EXPECT_TRUE(  cdict_["neg_int"]==cdict_["neg_int"] );
    EXPECT_FALSE( cdict_["neg_int"]!=cdict_["neg_int"] );
    // Different values:
    EXPECT_TRUE(  cdict_["neg_int"]!=cdict_["neg_int1"] );
    EXPECT_FALSE( cdict_["neg_int"]==cdict_["neg_int1"] );
    // Same values:
    EXPECT_TRUE(  cdict_["neg_int"]==cdict_["neg_long_is"] );
    EXPECT_FALSE( cdict_["neg_int"]!=cdict_["neg_long_is"] );
    // Different values:
    EXPECT_TRUE(  cdict_["neg_int"]!=cdict_["neg_long_is1"] );
    EXPECT_FALSE( cdict_["neg_int"]==cdict_["neg_long_is1"] );
    // Same values:
    EXPECT_TRUE(  cdict_["neg_long_is"]==cdict_["neg_long_is"] );
    EXPECT_FALSE( cdict_["neg_long_is"]!=cdict_["neg_long_is"] );
    // Different values:
    EXPECT_TRUE(  cdict_["neg_long_is"]!=cdict_["neg_long_is1"] );
    EXPECT_FALSE( cdict_["neg_long_is"]==cdict_["neg_long_is1"] );
}

// Equalities between domains neg_int:pos_int
TEST_F(DictionaryTestEq, eqNegIntPosIntLeft) {
    EXPECT_TRUE(  cdict_["neg_int"]!=+pos_int );
    EXPECT_FALSE( cdict_["neg_int"]==+pos_int );
    EXPECT_TRUE(  cdict_["neg_int"]!=+uint_is );
    EXPECT_FALSE( cdict_["neg_int"]==+uint_is );
    EXPECT_TRUE(  cdict_["neg_int"]!=+pos_long_is );
    EXPECT_FALSE( cdict_["neg_int"]==+pos_long_is );
    EXPECT_TRUE(  cdict_["neg_int"]!=+ulong_is );
    EXPECT_FALSE( cdict_["neg_int"]==+ulong_is );
    EXPECT_TRUE(  cdict_["neg_long_is"]!=+pos_int );
    EXPECT_FALSE( cdict_["neg_long_is"]==+pos_int );
    EXPECT_TRUE(  cdict_["neg_long_is"]!=+uint_is );
    EXPECT_FALSE( cdict_["neg_long_is"]==+uint_is );
    EXPECT_TRUE(  cdict_["neg_long_is"]!=+pos_long_is );
    EXPECT_FALSE( cdict_["neg_long_is"]==+pos_long_is );
    EXPECT_TRUE(  cdict_["neg_long_is"]!=+ulong_is );
    EXPECT_FALSE( cdict_["neg_long_is"]==+ulong_is );
}

TEST_F(DictionaryTestEq, eqNegIntPosIntRight) {
    EXPECT_TRUE(  +neg_int!=cdict_["pos_int"] );
    EXPECT_FALSE( +neg_int==cdict_["pos_int"] );
    EXPECT_TRUE(  +neg_int!=cdict_["uint_is"] );
    EXPECT_FALSE( +neg_int==cdict_["uint_is"] );
    EXPECT_TRUE(  +neg_int!=cdict_["pos_long_is"] );
    EXPECT_FALSE( +neg_int==cdict_["pos_long_is"] );
    EXPECT_TRUE(  +neg_int!=cdict_["ulong_is"] );
    EXPECT_FALSE( +neg_int==cdict_["ulong_is"] );
    EXPECT_TRUE(  +neg_long_is!=cdict_["pos_int"] );
    EXPECT_FALSE( +neg_long_is==cdict_["pos_int"] );
    EXPECT_TRUE(  +neg_long_is!=cdict_["uint_is"] );
    EXPECT_FALSE( +neg_long_is==cdict_["uint_is"] );
    EXPECT_TRUE(  +neg_long_is!=cdict_["pos_long_is"] );
    EXPECT_FALSE( +neg_long_is==cdict_["pos_long_is"] );
    EXPECT_TRUE(  +neg_long_is!=cdict_["ulong_is"] );
    EXPECT_FALSE( +neg_long_is==cdict_["ulong_is"] );
}

TEST_F(DictionaryTestEq, eqNegIntPosIntBoth) {
    EXPECT_TRUE(  cdict_["neg_int"]!=cdict_["pos_int"] );
    EXPECT_FALSE( cdict_["neg_int"]==cdict_["pos_int"] );
    EXPECT_TRUE(  cdict_["neg_int"]!=cdict_["uint_is"] );
    EXPECT_FALSE( cdict_["neg_int"]==cdict_["uint_is"] );
    EXPECT_TRUE(  cdict_["neg_int"]!=cdict_["pos_long_is"] );
    EXPECT_FALSE( cdict_["neg_int"]==cdict_["pos_long_is"] );
    EXPECT_TRUE(  cdict_["neg_int"]!=cdict_["ulong_is"] );
    EXPECT_FALSE( cdict_["neg_int"]==cdict_["ulong_is"] );
    EXPECT_TRUE(  cdict_["neg_long_is"]!=cdict_["pos_int"] );
    EXPECT_FALSE( cdict_["neg_long_is"]==cdict_["pos_int"] );
    EXPECT_TRUE(  cdict_["neg_long_is"]!=cdict_["uint_is"] );
    EXPECT_FALSE( cdict_["neg_long_is"]==cdict_["uint_is"] );
    EXPECT_TRUE(  cdict_["neg_long_is"]!=cdict_["pos_long_is"] );
    EXPECT_FALSE( cdict_["neg_long_is"]==cdict_["pos_long_is"] );
    EXPECT_TRUE(  cdict_["neg_long_is"]!=cdict_["ulong_is"] );
    EXPECT_FALSE( cdict_["neg_long_is"]==cdict_["ulong_is"] );
}

// Equalities between domains neg_int:pos_uint
TEST_F(DictionaryTestEq, eqNegIntPosUintLeft) {
    EXPECT_TRUE(  cdict_["neg_int"]!=+pos_uint );
    EXPECT_FALSE( cdict_["neg_int"]==+pos_uint );
    EXPECT_TRUE(  cdict_["neg_long_is"]!=+pos_uint );
    EXPECT_FALSE( cdict_["neg_long_is"]==+pos_uint );
}

TEST_F(DictionaryTestEq, eqNegIntPosUintRight) {
    EXPECT_TRUE(  +neg_int!=cdict_["pos_uint"] );
    EXPECT_FALSE( +neg_int==cdict_["pos_uint"] );
    EXPECT_TRUE(  +neg_long_is!=cdict_["pos_uint"] );
    EXPECT_FALSE( +neg_long_is==cdict_["pos_uint"] );
}

TEST_F(DictionaryTestEq, eqNegIntPosUintBoth) {
    EXPECT_TRUE(  cdict_["neg_int"]!=cdict_["pos_uint"] );
    EXPECT_FALSE( cdict_["neg_int"]==cdict_["pos_uint"] );
    EXPECT_TRUE(  cdict_["neg_long_is"]!=cdict_["pos_uint"] );
    EXPECT_FALSE( cdict_["neg_long_is"]==cdict_["pos_uint"] );
}

// Equalities between domains neg_int:pos_long
TEST_F(DictionaryTestEq, eqNegIntPosLongLeft) {
    EXPECT_TRUE(  cdict_["neg_int"]!=+pos_long );
    EXPECT_FALSE( cdict_["neg_int"]==+pos_long );
    EXPECT_TRUE(  cdict_["neg_int"]!=+ulong_ls );
    EXPECT_FALSE( cdict_["neg_int"]==+ulong_ls );
    EXPECT_TRUE(  cdict_["neg_long_is"]!=+pos_long );
    EXPECT_FALSE( cdict_["neg_long_is"]==+pos_long );
    EXPECT_TRUE(  cdict_["neg_long_is"]!=+ulong_ls );
    EXPECT_FALSE( cdict_["neg_long_is"]==+ulong_ls );
}

TEST_F(DictionaryTestEq, eqNegIntPosLongRight) {
    EXPECT_TRUE(  +neg_int!=cdict_["pos_long"] );
    EXPECT_FALSE( +neg_int==cdict_["pos_long"] );
    EXPECT_TRUE(  +neg_int!=cdict_["ulong_ls"] );
    EXPECT_FALSE( +neg_int==cdict_["ulong_ls"] );
    EXPECT_TRUE(  +neg_long_is!=cdict_["pos_long"] );
    EXPECT_FALSE( +neg_long_is==cdict_["pos_long"] );
    EXPECT_TRUE(  +neg_long_is!=cdict_["ulong_ls"] );
    EXPECT_FALSE( +neg_long_is==cdict_["ulong_ls"] );
}

TEST_F(DictionaryTestEq, eqNegIntPosLongBoth) {
    EXPECT_TRUE(  cdict_["neg_int"]!=cdict_["pos_long"] );
    EXPECT_FALSE( cdict_["neg_int"]==cdict_["pos_long"] );
    EXPECT_TRUE(  cdict_["neg_int"]!=cdict_["ulong_ls"] );
    EXPECT_FALSE( cdict_["neg_int"]==cdict_["ulong_ls"] );
    EXPECT_TRUE(  cdict_["neg_long_is"]!=cdict_["pos_long"] );
    EXPECT_FALSE( cdict_["neg_long_is"]==cdict_["pos_long"] );
    EXPECT_TRUE(  cdict_["neg_long_is"]!=cdict_["ulong_ls"] );
    EXPECT_FALSE( cdict_["neg_long_is"]==cdict_["ulong_ls"] );
}

// Equalities between domains neg_int:pos_ulong
TEST_F(DictionaryTestEq, eqNegIntULongLeft) {
    EXPECT_TRUE(  cdict_["neg_int"]!=+pos_ulong );
    EXPECT_FALSE( cdict_["neg_int"]==+pos_ulong );
    EXPECT_TRUE(  cdict_["neg_long_is"]!=+pos_ulong );
    EXPECT_FALSE( cdict_["neg_long_is"]==+pos_ulong );
}

TEST_F(DictionaryTestEq, eqNegIntULongRight) {
    EXPECT_TRUE(  +neg_int!=cdict_["pos_ulong"] );
    EXPECT_FALSE( +neg_int==cdict_["pos_ulong"] );
    EXPECT_TRUE(  +neg_long_is!=cdict_["pos_ulong"] );
    EXPECT_FALSE( +neg_long_is==cdict_["pos_ulong"] );
}

TEST_F(DictionaryTestEq, eqNegIntULongBoth) {
    EXPECT_TRUE(  cdict_["neg_int"]!=cdict_["pos_ulong"] );
    EXPECT_FALSE( cdict_["neg_int"]==cdict_["pos_ulong"] );
    EXPECT_TRUE(  cdict_["neg_long_is"]!=cdict_["pos_ulong"] );
    EXPECT_FALSE( cdict_["neg_long_is"]==cdict_["pos_ulong"] );
}

// Equalities within domain pos_int
TEST_F(DictionaryTestEq, eqPosIntLeft) {
    // Same values:
    EXPECT_TRUE(  cdict_["pos_int"]==+pos_int );
    EXPECT_FALSE( cdict_["pos_int"]!=+pos_int );
    // Different values:
    EXPECT_TRUE(  cdict_["pos_int"]!=+pos_int1 );
    EXPECT_FALSE( cdict_["pos_int"]==+pos_int1 );
    // Same values:
    EXPECT_TRUE(  cdict_["pos_int"]==+uint_is );
    EXPECT_FALSE( cdict_["pos_int"]!=+uint_is );
    // Different values:
    EXPECT_TRUE(  cdict_["pos_int"]!=+uint_is1 );
    EXPECT_FALSE( cdict_["pos_int"]==+uint_is1 );
    // Same values:
    EXPECT_TRUE(  cdict_["pos_int"]==+pos_long_is );
    EXPECT_FALSE( cdict_["pos_int"]!=+pos_long_is );
    // Different values:
    EXPECT_TRUE(  cdict_["pos_int"]!=+pos_long_is1 );
    EXPECT_FALSE( cdict_["pos_int"]==+pos_long_is1 );
    // Same values:
    EXPECT_TRUE(  cdict_["pos_int"]==+ulong_is );
    EXPECT_FALSE( cdict_["pos_int"]!=+ulong_is );
    // Different values:
    EXPECT_TRUE(  cdict_["pos_int"]!=+ulong_is1 );
    EXPECT_FALSE( cdict_["pos_int"]==+ulong_is1 );
    // Same values:
    EXPECT_TRUE(  cdict_["uint_is"]==+uint_is );
    EXPECT_FALSE( cdict_["uint_is"]!=+uint_is );
    // Different values:
    EXPECT_TRUE(  cdict_["uint_is"]!=+uint_is1 );
    EXPECT_FALSE( cdict_["uint_is"]==+uint_is1 );
    // Same values:
    EXPECT_TRUE(  cdict_["uint_is"]==+pos_long_is );
    EXPECT_FALSE( cdict_["uint_is"]!=+pos_long_is );
    // Different values:
    EXPECT_TRUE(  cdict_["uint_is"]!=+pos_long_is1 );
    EXPECT_FALSE( cdict_["uint_is"]==+pos_long_is1 );
    // Same values:
    EXPECT_TRUE(  cdict_["uint_is"]==+ulong_is );
    EXPECT_FALSE( cdict_["uint_is"]!=+ulong_is );
    // Different values:
    EXPECT_TRUE(  cdict_["uint_is"]!=+ulong_is1 );
    EXPECT_FALSE( cdict_["uint_is"]==+ulong_is1 );
    // Same values:
    EXPECT_TRUE(  cdict_["pos_long_is"]==+pos_long_is );
    EXPECT_FALSE( cdict_["pos_long_is"]!=+pos_long_is );
    // Different values:
    EXPECT_TRUE(  cdict_["pos_long_is"]!=+pos_long_is1 );
    EXPECT_FALSE( cdict_["pos_long_is"]==+pos_long_is1 );
    // Same values:
    EXPECT_TRUE(  cdict_["pos_long_is"]==+ulong_is );
    EXPECT_FALSE( cdict_["pos_long_is"]!=+ulong_is );
    // Different values:
    EXPECT_TRUE(  cdict_["pos_long_is"]!=+ulong_is1 );
    EXPECT_FALSE( cdict_["pos_long_is"]==+ulong_is1 );
    // Same values:
    EXPECT_TRUE(  cdict_["ulong_is"]==+ulong_is );
    EXPECT_FALSE( cdict_["ulong_is"]!=+ulong_is );
    // Different values:
    EXPECT_TRUE(  cdict_["ulong_is"]!=+ulong_is1 );
    EXPECT_FALSE( cdict_["ulong_is"]==+ulong_is1 );
}

TEST_F(DictionaryTestEq, eqPosIntRight) {
    // Same values:
    EXPECT_TRUE(  +pos_int==cdict_["pos_int"] );
    EXPECT_FALSE( +pos_int!=cdict_["pos_int"] );
    // Different values:
    EXPECT_TRUE(  +pos_int!=cdict_["pos_int1"] );
    EXPECT_FALSE( +pos_int==cdict_["pos_int1"] );
    // Same values:
    EXPECT_TRUE(  +pos_int==cdict_["uint_is"] );
    EXPECT_FALSE( +pos_int!=cdict_["uint_is"] );
    // Different values:
    EXPECT_TRUE(  +pos_int!=cdict_["uint_is1"] );
    EXPECT_FALSE( +pos_int==cdict_["uint_is1"] );
    // Same values:
    EXPECT_TRUE(  +pos_int==cdict_["pos_long_is"] );
    EXPECT_FALSE( +pos_int!=cdict_["pos_long_is"] );
    // Different values:
    EXPECT_TRUE(  +pos_int!=cdict_["pos_long_is1"] );
    EXPECT_FALSE( +pos_int==cdict_["pos_long_is1"] );
    // Same values:
    EXPECT_TRUE(  +pos_int==cdict_["ulong_is"] );
    EXPECT_FALSE( +pos_int!=cdict_["ulong_is"] );
    // Different values:
    EXPECT_TRUE(  +pos_int!=cdict_["ulong_is1"] );
    EXPECT_FALSE( +pos_int==cdict_["ulong_is1"] );
    // Same values:
    EXPECT_TRUE(  +uint_is==cdict_["uint_is"] );
    EXPECT_FALSE( +uint_is!=cdict_["uint_is"] );
    // Different values:
    EXPECT_TRUE(  +uint_is!=cdict_["uint_is1"] );
    EXPECT_FALSE( +uint_is==cdict_["uint_is1"] );
    // Same values:
    EXPECT_TRUE(  +uint_is==cdict_["pos_long_is"] );
    EXPECT_FALSE( +uint_is!=cdict_["pos_long_is"] );
    // Different values:
    EXPECT_TRUE(  +uint_is!=cdict_["pos_long_is1"] );
    EXPECT_FALSE( +uint_is==cdict_["pos_long_is1"] );
    // Same values:
    EXPECT_TRUE(  +uint_is==cdict_["ulong_is"] );
    EXPECT_FALSE( +uint_is!=cdict_["ulong_is"] );
    // Different values:
    EXPECT_TRUE(  +uint_is!=cdict_["ulong_is1"] );
    EXPECT_FALSE( +uint_is==cdict_["ulong_is1"] );
    // Same values:
    EXPECT_TRUE(  +pos_long_is==cdict_["pos_long_is"] );
    EXPECT_FALSE( +pos_long_is!=cdict_["pos_long_is"] );
    // Different values:
    EXPECT_TRUE(  +pos_long_is!=cdict_["pos_long_is1"] );
    EXPECT_FALSE( +pos_long_is==cdict_["pos_long_is1"] );
    // Same values:
    EXPECT_TRUE(  +pos_long_is==cdict_["ulong_is"] );
    EXPECT_FALSE( +pos_long_is!=cdict_["ulong_is"] );
    // Different values:
    EXPECT_TRUE(  +pos_long_is!=cdict_["ulong_is1"] );
    EXPECT_FALSE( +pos_long_is==cdict_["ulong_is1"] );
    // Same values:
    EXPECT_TRUE(  +ulong_is==cdict_["ulong_is"] );
    EXPECT_FALSE( +ulong_is!=cdict_["ulong_is"] );
    // Different values:
    EXPECT_TRUE(  +ulong_is!=cdict_["ulong_is1"] );
    EXPECT_FALSE( +ulong_is==cdict_["ulong_is1"] );
}

TEST_F(DictionaryTestEq, eqPosIntBoth) {
    // Same values:
    EXPECT_TRUE(  cdict_["pos_int"]==cdict_["pos_int"] );
    EXPECT_FALSE( cdict_["pos_int"]!=cdict_["pos_int"] );
    // Different values:
    EXPECT_TRUE(  cdict_["pos_int"]!=cdict_["pos_int1"] );
    EXPECT_FALSE( cdict_["pos_int"]==cdict_["pos_int1"] );
    // Same values:
    EXPECT_TRUE(  cdict_["pos_int"]==cdict_["uint_is"] );
    EXPECT_FALSE( cdict_["pos_int"]!=cdict_["uint_is"] );
    // Different values:
    EXPECT_TRUE(  cdict_["pos_int"]!=cdict_["uint_is1"] );
    EXPECT_FALSE( cdict_["pos_int"]==cdict_["uint_is1"] );
    // Same values:
    EXPECT_TRUE(  cdict_["pos_int"]==cdict_["pos_long_is"] );
    EXPECT_FALSE( cdict_["pos_int"]!=cdict_["pos_long_is"] );
    // Different values:
    EXPECT_TRUE(  cdict_["pos_int"]!=cdict_["pos_long_is1"] );
    EXPECT_FALSE( cdict_["pos_int"]==cdict_["pos_long_is1"] );
    // Same values:
    EXPECT_TRUE(  cdict_["pos_int"]==cdict_["ulong_is"] );
    EXPECT_FALSE( cdict_["pos_int"]!=cdict_["ulong_is"] );
    // Different values:
    EXPECT_TRUE(  cdict_["pos_int"]!=cdict_["ulong_is1"] );
    EXPECT_FALSE( cdict_["pos_int"]==cdict_["ulong_is1"] );
    // Same values:
    EXPECT_TRUE(  cdict_["uint_is"]==cdict_["uint_is"] );
    EXPECT_FALSE( cdict_["uint_is"]!=cdict_["uint_is"] );
    // Different values:
    EXPECT_TRUE(  cdict_["uint_is"]!=cdict_["uint_is1"] );
    EXPECT_FALSE( cdict_["uint_is"]==cdict_["uint_is1"] );
    // Same values:
    EXPECT_TRUE(  cdict_["uint_is"]==cdict_["pos_long_is"] );
    EXPECT_FALSE( cdict_["uint_is"]!=cdict_["pos_long_is"] );
    // Different values:
    EXPECT_TRUE(  cdict_["uint_is"]!=cdict_["pos_long_is1"] );
    EXPECT_FALSE( cdict_["uint_is"]==cdict_["pos_long_is1"] );
    // Same values:
    EXPECT_TRUE(  cdict_["uint_is"]==cdict_["ulong_is"] );
    EXPECT_FALSE( cdict_["uint_is"]!=cdict_["ulong_is"] );
    // Different values:
    EXPECT_TRUE(  cdict_["uint_is"]!=cdict_["ulong_is1"] );
    EXPECT_FALSE( cdict_["uint_is"]==cdict_["ulong_is1"] );
    // Same values:
    EXPECT_TRUE(  cdict_["pos_long_is"]==cdict_["pos_long_is"] );
    EXPECT_FALSE( cdict_["pos_long_is"]!=cdict_["pos_long_is"] );
    // Different values:
    EXPECT_TRUE(  cdict_["pos_long_is"]!=cdict_["pos_long_is1"] );
    EXPECT_FALSE( cdict_["pos_long_is"]==cdict_["pos_long_is1"] );
    // Same values:
    EXPECT_TRUE(  cdict_["pos_long_is"]==cdict_["ulong_is"] );
    EXPECT_FALSE( cdict_["pos_long_is"]!=cdict_["ulong_is"] );
    // Different values:
    EXPECT_TRUE(  cdict_["pos_long_is"]!=cdict_["ulong_is1"] );
    EXPECT_FALSE( cdict_["pos_long_is"]==cdict_["ulong_is1"] );
    // Same values:
    EXPECT_TRUE(  cdict_["ulong_is"]==cdict_["ulong_is"] );
    EXPECT_FALSE( cdict_["ulong_is"]!=cdict_["ulong_is"] );
    // Different values:
    EXPECT_TRUE(  cdict_["ulong_is"]!=cdict_["ulong_is1"] );
    EXPECT_FALSE( cdict_["ulong_is"]==cdict_["ulong_is1"] );
}

// Equalities between domains pos_int:pos_uint
TEST_F(DictionaryTestEq, eqPosIntPosUintLeft) {
    EXPECT_TRUE(  cdict_["pos_int"]!=+pos_uint );
    EXPECT_FALSE( cdict_["pos_int"]==+pos_uint );
    EXPECT_TRUE(  cdict_["uint_is"]!=+pos_uint );
    EXPECT_FALSE( cdict_["uint_is"]==+pos_uint );
    EXPECT_TRUE(  cdict_["pos_long_is"]!=+pos_uint );
    EXPECT_FALSE( cdict_["pos_long_is"]==+pos_uint );
    EXPECT_TRUE(  cdict_["ulong_is"]!=+pos_uint );
    EXPECT_FALSE( cdict_["ulong_is"]==+pos_uint );
}

TEST_F(DictionaryTestEq, eqPosIntPosUintRight) {
    EXPECT_TRUE(  +pos_int!=cdict_["pos_uint"] );
    EXPECT_FALSE( +pos_int==cdict_["pos_uint"] );
    EXPECT_TRUE(  +uint_is!=cdict_["pos_uint"] );
    EXPECT_FALSE( +uint_is==cdict_["pos_uint"] );
    EXPECT_TRUE(  +pos_long_is!=cdict_["pos_uint"] );
    EXPECT_FALSE( +pos_long_is==cdict_["pos_uint"] );
    EXPECT_TRUE(  +ulong_is!=cdict_["pos_uint"] );
    EXPECT_FALSE( +ulong_is==cdict_["pos_uint"] );
}

TEST_F(DictionaryTestEq, eqPosIntPosUintBoth) {
    EXPECT_TRUE(  cdict_["pos_int"]!=cdict_["pos_uint"] );
    EXPECT_FALSE( cdict_["pos_int"]==cdict_["pos_uint"] );
    EXPECT_TRUE(  cdict_["uint_is"]!=cdict_["pos_uint"] );
    EXPECT_FALSE( cdict_["uint_is"]==cdict_["pos_uint"] );
    EXPECT_TRUE(  cdict_["pos_long_is"]!=cdict_["pos_uint"] );
    EXPECT_FALSE( cdict_["pos_long_is"]==cdict_["pos_uint"] );
    EXPECT_TRUE(  cdict_["ulong_is"]!=cdict_["pos_uint"] );
    EXPECT_FALSE( cdict_["ulong_is"]==cdict_["pos_uint"] );
}

// Equalities between domains pos_int:pos_long
TEST_F(DictionaryTestEq, eqPosIntPosLongLeft) {
    EXPECT_TRUE(  cdict_["pos_int"]!=+pos_long );
    EXPECT_FALSE( cdict_["pos_int"]==+pos_long );
    EXPECT_TRUE(  cdict_["pos_int"]!=+ulong_ls );
    EXPECT_FALSE( cdict_["pos_int"]==+ulong_ls );
    EXPECT_TRUE(  cdict_["uint_is"]!=+pos_long );
    EXPECT_FALSE( cdict_["uint_is"]==+pos_long );
    EXPECT_TRUE(  cdict_["uint_is"]!=+ulong_ls );
    EXPECT_FALSE( cdict_["uint_is"]==+ulong_ls );
    EXPECT_TRUE(  cdict_["pos_long_is"]!=+pos_long );
    EXPECT_FALSE( cdict_["pos_long_is"]==+pos_long );
    EXPECT_TRUE(  cdict_["pos_long_is"]!=+ulong_ls );
    EXPECT_FALSE( cdict_["pos_long_is"]==+ulong_ls );
    EXPECT_TRUE(  cdict_["ulong_is"]!=+pos_long );
    EXPECT_FALSE( cdict_["ulong_is"]==+pos_long );
    EXPECT_TRUE(  cdict_["ulong_is"]!=+ulong_ls );
    EXPECT_FALSE( cdict_["ulong_is"]==+ulong_ls );
}

TEST_F(DictionaryTestEq, eqPosIntPosLongRight) {
    EXPECT_TRUE(  +pos_int!=cdict_["pos_long"] );
    EXPECT_FALSE( +pos_int==cdict_["pos_long"] );
    EXPECT_TRUE(  +pos_int!=cdict_["ulong_ls"] );
    EXPECT_FALSE( +pos_int==cdict_["ulong_ls"] );
    EXPECT_TRUE(  +uint_is!=cdict_["pos_long"] );
    EXPECT_FALSE( +uint_is==cdict_["pos_long"] );
    EXPECT_TRUE(  +uint_is!=cdict_["ulong_ls"] );
    EXPECT_FALSE( +uint_is==cdict_["ulong_ls"] );
    EXPECT_TRUE(  +pos_long_is!=cdict_["pos_long"] );
    EXPECT_FALSE( +pos_long_is==cdict_["pos_long"] );
    EXPECT_TRUE(  +pos_long_is!=cdict_["ulong_ls"] );
    EXPECT_FALSE( +pos_long_is==cdict_["ulong_ls"] );
    EXPECT_TRUE(  +ulong_is!=cdict_["pos_long"] );
    EXPECT_FALSE( +ulong_is==cdict_["pos_long"] );
    EXPECT_TRUE(  +ulong_is!=cdict_["ulong_ls"] );
    EXPECT_FALSE( +ulong_is==cdict_["ulong_ls"] );
}

TEST_F(DictionaryTestEq, eqPosIntPosLongBoth) {
    EXPECT_TRUE(  cdict_["pos_int"]!=cdict_["pos_long"] );
    EXPECT_FALSE( cdict_["pos_int"]==cdict_["pos_long"] );
    EXPECT_TRUE(  cdict_["pos_int"]!=cdict_["ulong_ls"] );
    EXPECT_FALSE( cdict_["pos_int"]==cdict_["ulong_ls"] );
    EXPECT_TRUE(  cdict_["uint_is"]!=cdict_["pos_long"] );
    EXPECT_FALSE( cdict_["uint_is"]==cdict_["pos_long"] );
    EXPECT_TRUE(  cdict_["uint_is"]!=cdict_["ulong_ls"] );
    EXPECT_FALSE( cdict_["uint_is"]==cdict_["ulong_ls"] );
    EXPECT_TRUE(  cdict_["pos_long_is"]!=cdict_["pos_long"] );
    EXPECT_FALSE( cdict_["pos_long_is"]==cdict_["pos_long"] );
    EXPECT_TRUE(  cdict_["pos_long_is"]!=cdict_["ulong_ls"] );
    EXPECT_FALSE( cdict_["pos_long_is"]==cdict_["ulong_ls"] );
    EXPECT_TRUE(  cdict_["ulong_is"]!=cdict_["pos_long"] );
    EXPECT_FALSE( cdict_["ulong_is"]==cdict_["pos_long"] );
    EXPECT_TRUE(  cdict_["ulong_is"]!=cdict_["ulong_ls"] );
    EXPECT_FALSE( cdict_["ulong_is"]==cdict_["ulong_ls"] );
}

// Equalities between domains pos_int:pos_ulong
TEST_F(DictionaryTestEq, eqPosIntULongLeft) {
    EXPECT_TRUE(  cdict_["pos_int"]!=+pos_ulong );
    EXPECT_FALSE( cdict_["pos_int"]==+pos_ulong );
    EXPECT_TRUE(  cdict_["uint_is"]!=+pos_ulong );
    EXPECT_FALSE( cdict_["uint_is"]==+pos_ulong );
    EXPECT_TRUE(  cdict_["pos_long_is"]!=+pos_ulong );
    EXPECT_FALSE( cdict_["pos_long_is"]==+pos_ulong );
    EXPECT_TRUE(  cdict_["ulong_is"]!=+pos_ulong );
    EXPECT_FALSE( cdict_["ulong_is"]==+pos_ulong );
}

TEST_F(DictionaryTestEq, eqPosIntULongRight) {
    EXPECT_TRUE(  +pos_int!=cdict_["pos_ulong"] );
    EXPECT_FALSE( +pos_int==cdict_["pos_ulong"] );
    EXPECT_TRUE(  +uint_is!=cdict_["pos_ulong"] );
    EXPECT_FALSE( +uint_is==cdict_["pos_ulong"] );
    EXPECT_TRUE(  +pos_long_is!=cdict_["pos_ulong"] );
    EXPECT_FALSE( +pos_long_is==cdict_["pos_ulong"] );
    EXPECT_TRUE(  +ulong_is!=cdict_["pos_ulong"] );
    EXPECT_FALSE( +ulong_is==cdict_["pos_ulong"] );
}

TEST_F(DictionaryTestEq, eqPosIntULongBoth) {
    EXPECT_TRUE(  cdict_["pos_int"]!=cdict_["pos_ulong"] );
    EXPECT_FALSE( cdict_["pos_int"]==cdict_["pos_ulong"] );
    EXPECT_TRUE(  cdict_["uint_is"]!=cdict_["pos_ulong"] );
    EXPECT_FALSE( cdict_["uint_is"]==cdict_["pos_ulong"] );
    EXPECT_TRUE(  cdict_["pos_long_is"]!=cdict_["pos_ulong"] );
    EXPECT_FALSE( cdict_["pos_long_is"]==cdict_["pos_ulong"] );
    EXPECT_TRUE(  cdict_["ulong_is"]!=cdict_["pos_ulong"] );
    EXPECT_FALSE( cdict_["ulong_is"]==cdict_["pos_ulong"] );
}

// Equalities within domain pos_uint
TEST_F(DictionaryTestEq, eqPosUintLeft) {
    // Same values:
    EXPECT_TRUE(  cdict_["pos_uint"]==+pos_uint );
    EXPECT_FALSE( cdict_["pos_uint"]!=+pos_uint );
    // Different values:
    EXPECT_TRUE(  cdict_["pos_uint"]!=+pos_uint1 );
    EXPECT_FALSE( cdict_["pos_uint"]==+pos_uint1 );
}

TEST_F(DictionaryTestEq, eqPosUintRight) {
    // Same values:
    EXPECT_TRUE(  +pos_uint==cdict_["pos_uint"] );
    EXPECT_FALSE( +pos_uint!=cdict_["pos_uint"] );
    // Different values:
    EXPECT_TRUE(  +pos_uint!=cdict_["pos_uint1"] );
    EXPECT_FALSE( +pos_uint==cdict_["pos_uint1"] );
}

TEST_F(DictionaryTestEq, eqPosUintBoth) {
    // Same values:
    EXPECT_TRUE(  cdict_["pos_uint"]==cdict_["pos_uint"] );
    EXPECT_FALSE( cdict_["pos_uint"]!=cdict_["pos_uint"] );
    // Different values:
    EXPECT_TRUE(  cdict_["pos_uint"]!=cdict_["pos_uint1"] );
    EXPECT_FALSE( cdict_["pos_uint"]==cdict_["pos_uint1"] );
}

// Equalities between domains pos_uint:pos_long
TEST_F(DictionaryTestEq, eqPosUintPosLongLeft) {
    EXPECT_TRUE(  cdict_["pos_uint"]!=+pos_long );
    EXPECT_FALSE( cdict_["pos_uint"]==+pos_long );
    EXPECT_TRUE(  cdict_["pos_uint"]!=+ulong_ls );
    EXPECT_FALSE( cdict_["pos_uint"]==+ulong_ls );
}

TEST_F(DictionaryTestEq, eqPosUintPosLongRight) {
    EXPECT_TRUE(  +pos_uint!=cdict_["pos_long"] );
    EXPECT_FALSE( +pos_uint==cdict_["pos_long"] );
    EXPECT_TRUE(  +pos_uint!=cdict_["ulong_ls"] );
    EXPECT_FALSE( +pos_uint==cdict_["ulong_ls"] );
}

TEST_F(DictionaryTestEq, eqPosUintPosLongBoth) {
    EXPECT_TRUE(  cdict_["pos_uint"]!=cdict_["pos_long"] );
    EXPECT_FALSE( cdict_["pos_uint"]==cdict_["pos_long"] );
    EXPECT_TRUE(  cdict_["pos_uint"]!=cdict_["ulong_ls"] );
    EXPECT_FALSE( cdict_["pos_uint"]==cdict_["ulong_ls"] );
}

// Equalities between domains pos_uint:pos_ulong
TEST_F(DictionaryTestEq, eqPosUintULongLeft) {
    EXPECT_TRUE(  cdict_["pos_uint"]!=+pos_ulong );
    EXPECT_FALSE( cdict_["pos_uint"]==+pos_ulong );
}

TEST_F(DictionaryTestEq, eqPosUintULongRight) {
    EXPECT_TRUE(  +pos_uint!=cdict_["pos_ulong"] );
    EXPECT_FALSE( +pos_uint==cdict_["pos_ulong"] );
}

TEST_F(DictionaryTestEq, eqPosUintULongBoth) {
    EXPECT_TRUE(  cdict_["pos_uint"]!=cdict_["pos_ulong"] );
    EXPECT_FALSE( cdict_["pos_uint"]==cdict_["pos_ulong"] );
}

// Equalities within domain pos_long
TEST_F(DictionaryTestEq, eqPosLongLeft) {
    // Same values:
    EXPECT_TRUE(  cdict_["pos_long"]==+pos_long );
    EXPECT_FALSE( cdict_["pos_long"]!=+pos_long );
    // Different values:
    EXPECT_TRUE(  cdict_["pos_long"]!=+pos_long1 );
    EXPECT_FALSE( cdict_["pos_long"]==+pos_long1 );
    // Same values:
    EXPECT_TRUE(  cdict_["pos_long"]==+ulong_ls );
    EXPECT_FALSE( cdict_["pos_long"]!=+ulong_ls );
    // Different values:
    EXPECT_TRUE(  cdict_["pos_long"]!=+ulong_ls1 );
    EXPECT_FALSE( cdict_["pos_long"]==+ulong_ls1 );
    // Same values:
    EXPECT_TRUE(  cdict_["ulong_ls"]==+ulong_ls );
    EXPECT_FALSE( cdict_["ulong_ls"]!=+ulong_ls );
    // Different values:
    EXPECT_TRUE(  cdict_["ulong_ls"]!=+ulong_ls1 );
    EXPECT_FALSE( cdict_["ulong_ls"]==+ulong_ls1 );
}

TEST_F(DictionaryTestEq, eqPosLongRight) {
    // Same values:
    EXPECT_TRUE(  +pos_long==cdict_["pos_long"] );
    EXPECT_FALSE( +pos_long!=cdict_["pos_long"] );
    // Different values:
    EXPECT_TRUE(  +pos_long!=cdict_["pos_long1"] );
    EXPECT_FALSE( +pos_long==cdict_["pos_long1"] );
    // Same values:
    EXPECT_TRUE(  +pos_long==cdict_["ulong_ls"] );
    EXPECT_FALSE( +pos_long!=cdict_["ulong_ls"] );
    // Different values:
    EXPECT_TRUE(  +pos_long!=cdict_["ulong_ls1"] );
    EXPECT_FALSE( +pos_long==cdict_["ulong_ls1"] );
    // Same values:
    EXPECT_TRUE(  +ulong_ls==cdict_["ulong_ls"] );
    EXPECT_FALSE( +ulong_ls!=cdict_["ulong_ls"] );
    // Different values:
    EXPECT_TRUE(  +ulong_ls!=cdict_["ulong_ls1"] );
    EXPECT_FALSE( +ulong_ls==cdict_["ulong_ls1"] );
}

TEST_F(DictionaryTestEq, eqPosLongBoth) {
    // Same values:
    EXPECT_TRUE(  cdict_["pos_long"]==cdict_["pos_long"] );
    EXPECT_FALSE( cdict_["pos_long"]!=cdict_["pos_long"] );
    // Different values:
    EXPECT_TRUE(  cdict_["pos_long"]!=cdict_["pos_long1"] );
    EXPECT_FALSE( cdict_["pos_long"]==cdict_["pos_long1"] );
    // Same values:
    EXPECT_TRUE(  cdict_["pos_long"]==cdict_["ulong_ls"] );
    EXPECT_FALSE( cdict_["pos_long"]!=cdict_["ulong_ls"] );
    // Different values:
    EXPECT_TRUE(  cdict_["pos_long"]!=cdict_["ulong_ls1"] );
    EXPECT_FALSE( cdict_["pos_long"]==cdict_["ulong_ls1"] );
    // Same values:
    EXPECT_TRUE(  cdict_["ulong_ls"]==cdict_["ulong_ls"] );
    EXPECT_FALSE( cdict_["ulong_ls"]!=cdict_["ulong_ls"] );
    // Different values:
    EXPECT_TRUE(  cdict_["ulong_ls"]!=cdict_["ulong_ls1"] );
    EXPECT_FALSE( cdict_["ulong_ls"]==cdict_["ulong_ls1"] );
}

// Equalities between domains pos_long:pos_ulong
TEST_F(DictionaryTestEq, eqPosLongULongLeft) {
    EXPECT_TRUE(  cdict_["pos_long"]!=+pos_ulong );
    EXPECT_FALSE( cdict_["pos_long"]==+pos_ulong );
    EXPECT_TRUE(  cdict_["ulong_ls"]!=+pos_ulong );
    EXPECT_FALSE( cdict_["ulong_ls"]==+pos_ulong );
}

TEST_F(DictionaryTestEq, eqPosLongULongRight) {
    EXPECT_TRUE(  +pos_long!=cdict_["pos_ulong"] );
    EXPECT_FALSE( +pos_long==cdict_["pos_ulong"] );
    EXPECT_TRUE(  +ulong_ls!=cdict_["pos_ulong"] );
    EXPECT_FALSE( +ulong_ls==cdict_["pos_ulong"] );
}

TEST_F(DictionaryTestEq, eqPosLongULongBoth) {
    EXPECT_TRUE(  cdict_["pos_long"]!=cdict_["pos_ulong"] );
    EXPECT_FALSE( cdict_["pos_long"]==cdict_["pos_ulong"] );
    EXPECT_TRUE(  cdict_["ulong_ls"]!=cdict_["pos_ulong"] );
    EXPECT_FALSE( cdict_["ulong_ls"]==cdict_["pos_ulong"] );
}

// Equalities within domain pos_ulong
TEST_F(DictionaryTestEq, eqULongLeft) {
    // Same values:
    EXPECT_TRUE(  cdict_["pos_ulong"]==+pos_ulong );
    EXPECT_FALSE( cdict_["pos_ulong"]!=+pos_ulong );
    // Different values:
    EXPECT_TRUE(  cdict_["pos_ulong"]!=+pos_ulong1 );
    EXPECT_FALSE( cdict_["pos_ulong"]==+pos_ulong1 );
}

TEST_F(DictionaryTestEq, eqULongRight) {
    // Same values:
    EXPECT_TRUE(  +pos_ulong==cdict_["pos_ulong"] );
    EXPECT_FALSE( +pos_ulong!=cdict_["pos_ulong"] );
    // Different values:
    EXPECT_TRUE(  +pos_ulong!=cdict_["pos_ulong1"] );
    EXPECT_FALSE( +pos_ulong==cdict_["pos_ulong1"] );
}

TEST_F(DictionaryTestEq, eqULongBoth) {
    // Same values:
    EXPECT_TRUE(  cdict_["pos_ulong"]==cdict_["pos_ulong"] );
    EXPECT_FALSE( cdict_["pos_ulong"]!=cdict_["pos_ulong"] );
    // Different values:
    EXPECT_TRUE(  cdict_["pos_ulong"]!=cdict_["pos_ulong1"] );
    EXPECT_FALSE( cdict_["pos_ulong"]==cdict_["pos_ulong1"] );
}

// Equalities within same type my_bool
TEST_F(DictionaryTestEq, eqBoolLeft) {
    // Same values:
    EXPECT_TRUE(  cdict_["my_bool"]==my_bool );
    EXPECT_FALSE( cdict_["my_bool"]!=my_bool );
    // Different values:
    EXPECT_TRUE(  cdict_["my_bool"]!=my_bool1 );
    EXPECT_FALSE( cdict_["my_bool"]==my_bool1 );
}

TEST_F(DictionaryTestEq, eqBoolRight) {
    // Same values:
    EXPECT_TRUE(  my_bool==cdict_["my_bool"] );
    EXPECT_FALSE( my_bool!=cdict_["my_bool"] );
    // Different values:
    EXPECT_TRUE(  my_bool!=cdict_["my_bool1"] );
    EXPECT_FALSE( my_bool==cdict_["my_bool1"] );
}

TEST_F(DictionaryTestEq, eqBoolBoth) {
    // Same values:
    EXPECT_TRUE(  cdict_["my_bool"]==cdict_["my_bool"] );
    EXPECT_FALSE( cdict_["my_bool"]!=cdict_["my_bool"] );
    // Different values:
    EXPECT_TRUE(  cdict_["my_bool"]!=cdict_["my_bool1"] );
    EXPECT_FALSE( cdict_["my_bool"]==cdict_["my_bool1"] );
}

// Equalities between different types my_bool:my_int
TEST_F(DictionaryTestEq, eqBoolIntLeft) {
    EXPECT_THROW(  cdict_["my_bool"]!=my_int, de::type_mismatch );
    EXPECT_THROW(  cdict_["my_bool"]==my_int, de::type_mismatch );
}

TEST_F(DictionaryTestEq, eqBoolIntRight) {
    EXPECT_THROW(  my_bool!=cdict_["my_int"], de::type_mismatch );
    EXPECT_THROW(  my_bool==cdict_["my_int"], de::type_mismatch );
}

TEST_F(DictionaryTestEq, eqBoolIntBoth) {
    EXPECT_THROW(  cdict_["my_bool"]!=cdict_["my_int"], de::type_mismatch );
    EXPECT_THROW(  cdict_["my_bool"]==cdict_["my_int"], de::type_mismatch );
}

// Equalities between different types my_bool:my_string
TEST_F(DictionaryTestEq, eqBoolStringLeft) {
    EXPECT_THROW(  cdict_["my_bool"]!=my_string, de::type_mismatch );
    EXPECT_THROW(  cdict_["my_bool"]==my_string, de::type_mismatch );
}

TEST_F(DictionaryTestEq, eqBoolStringRight) {
    EXPECT_THROW(  my_bool!=cdict_["my_string"], de::type_mismatch );
    EXPECT_THROW(  my_bool==cdict_["my_string"], de::type_mismatch );
}

TEST_F(DictionaryTestEq, eqBoolStringBoth) {
    EXPECT_THROW(  cdict_["my_bool"]!=cdict_["my_string"], de::type_mismatch );
    EXPECT_THROW(  cdict_["my_bool"]==cdict_["my_string"], de::type_mismatch );
}

// Equalities between different types my_bool:my_vec
TEST_F(DictionaryTestEq, eqBoolVecLeft) {
    EXPECT_THROW(  cdict_["my_bool"]!=my_vec, de::type_mismatch );
    EXPECT_THROW(  cdict_["my_bool"]==my_vec, de::type_mismatch );
}

TEST_F(DictionaryTestEq, eqBoolVecRight) {
    EXPECT_THROW(  my_bool!=cdict_["my_vec"], de::type_mismatch );
    EXPECT_THROW(  my_bool==cdict_["my_vec"], de::type_mismatch );
}

TEST_F(DictionaryTestEq, eqBoolVecBoth) {
    EXPECT_THROW(  cdict_["my_bool"]!=cdict_["my_vec"], de::type_mismatch );
    EXPECT_THROW(  cdict_["my_bool"]==cdict_["my_vec"], de::type_mismatch );
}

// Equalities between different types my_bool:my_pair
TEST_F(DictionaryTestEq, eqBoolPairLeft) {
    EXPECT_THROW(  cdict_["my_bool"]!=my_pair, de::type_mismatch );
    EXPECT_THROW(  cdict_["my_bool"]==my_pair, de::type_mismatch );
}

TEST_F(DictionaryTestEq, eqBoolPairRight) {
    EXPECT_THROW(  my_bool!=cdict_["my_pair"], de::type_mismatch );
    EXPECT_THROW(  my_bool==cdict_["my_pair"], de::type_mismatch );
}

TEST_F(DictionaryTestEq, eqBoolPairBoth) {
    EXPECT_THROW(  cdict_["my_bool"]!=cdict_["my_pair"], de::type_mismatch );
    EXPECT_THROW(  cdict_["my_bool"]==cdict_["my_pair"], de::type_mismatch );
}

// Equalities within same type my_int
TEST_F(DictionaryTestEq, eqIntLeft) {
    // Same values:
    EXPECT_TRUE(  cdict_["my_int"]==my_int );
    EXPECT_FALSE( cdict_["my_int"]!=my_int );
    // Different values:
    EXPECT_TRUE(  cdict_["my_int"]!=my_int1 );
    EXPECT_FALSE( cdict_["my_int"]==my_int1 );
}

TEST_F(DictionaryTestEq, eqIntRight) {
    // Same values:
    EXPECT_TRUE(  my_int==cdict_["my_int"] );
    EXPECT_FALSE( my_int!=cdict_["my_int"] );
    // Different values:
    EXPECT_TRUE(  my_int!=cdict_["my_int1"] );
    EXPECT_FALSE( my_int==cdict_["my_int1"] );
}

TEST_F(DictionaryTestEq, eqIntBoth) {
    // Same values:
    EXPECT_TRUE(  cdict_["my_int"]==cdict_["my_int"] );
    EXPECT_FALSE( cdict_["my_int"]!=cdict_["my_int"] );
    // Different values:
    EXPECT_TRUE(  cdict_["my_int"]!=cdict_["my_int1"] );
    EXPECT_FALSE( cdict_["my_int"]==cdict_["my_int1"] );
}

// Equalities between different types my_int:my_string
TEST_F(DictionaryTestEq, eqIntStringLeft) {
    EXPECT_THROW(  cdict_["my_int"]!=my_string, de::type_mismatch );
    EXPECT_THROW(  cdict_["my_int"]==my_string, de::type_mismatch );
}

TEST_F(DictionaryTestEq, eqIntStringRight) {
    EXPECT_THROW(  my_int!=cdict_["my_string"], de::type_mismatch );
    EXPECT_THROW(  my_int==cdict_["my_string"], de::type_mismatch );
}

TEST_F(DictionaryTestEq, eqIntStringBoth) {
    EXPECT_THROW(  cdict_["my_int"]!=cdict_["my_string"], de::type_mismatch );
    EXPECT_THROW(  cdict_["my_int"]==cdict_["my_string"], de::type_mismatch );
}

// Equalities between different types my_int:my_vec
TEST_F(DictionaryTestEq, eqIntVecLeft) {
    EXPECT_THROW(  cdict_["my_int"]!=my_vec, de::type_mismatch );
    EXPECT_THROW(  cdict_["my_int"]==my_vec, de::type_mismatch );
}

TEST_F(DictionaryTestEq, eqIntVecRight) {
    EXPECT_THROW(  my_int!=cdict_["my_vec"], de::type_mismatch );
    EXPECT_THROW(  my_int==cdict_["my_vec"], de::type_mismatch );
}

TEST_F(DictionaryTestEq, eqIntVecBoth) {
    EXPECT_THROW(  cdict_["my_int"]!=cdict_["my_vec"], de::type_mismatch );
    EXPECT_THROW(  cdict_["my_int"]==cdict_["my_vec"], de::type_mismatch );
}

// Equalities between different types my_int:my_pair
TEST_F(DictionaryTestEq, eqIntPairLeft) {
    EXPECT_THROW(  cdict_["my_int"]!=my_pair, de::type_mismatch );
    EXPECT_THROW(  cdict_["my_int"]==my_pair, de::type_mismatch );
}

TEST_F(DictionaryTestEq, eqIntPairRight) {
    EXPECT_THROW(  my_int!=cdict_["my_pair"], de::type_mismatch );
    EXPECT_THROW(  my_int==cdict_["my_pair"], de::type_mismatch );
}

TEST_F(DictionaryTestEq, eqIntPairBoth) {
    EXPECT_THROW(  cdict_["my_int"]!=cdict_["my_pair"], de::type_mismatch );
    EXPECT_THROW(  cdict_["my_int"]==cdict_["my_pair"], de::type_mismatch );
}

// Equalities within same type my_string
TEST_F(DictionaryTestEq, eqStringLeft) {
    // Same values:
    EXPECT_TRUE(  cdict_["my_string"]==my_string );
    EXPECT_FALSE( cdict_["my_string"]!=my_string );
    // Different values:
    EXPECT_TRUE(  cdict_["my_string"]!=my_string1 );
    EXPECT_FALSE( cdict_["my_string"]==my_string1 );
}

TEST_F(DictionaryTestEq, eqStringRight) {
    // Same values:
    EXPECT_TRUE(  my_string==cdict_["my_string"] );
    EXPECT_FALSE( my_string!=cdict_["my_string"] );
    // Different values:
    EXPECT_TRUE(  my_string!=cdict_["my_string1"] );
    EXPECT_FALSE( my_string==cdict_["my_string1"] );
}

TEST_F(DictionaryTestEq, eqStringBoth) {
    // Same values:
    EXPECT_TRUE(  cdict_["my_string"]==cdict_["my_string"] );
    EXPECT_FALSE( cdict_["my_string"]!=cdict_["my_string"] );
    // Different values:
    EXPECT_TRUE(  cdict_["my_string"]!=cdict_["my_string1"] );
    EXPECT_FALSE( cdict_["my_string"]==cdict_["my_string1"] );
}

// Equalities between different types my_string:my_vec
TEST_F(DictionaryTestEq, eqStringVecLeft) {
    EXPECT_THROW(  cdict_["my_string"]!=my_vec, de::type_mismatch );
    EXPECT_THROW(  cdict_["my_string"]==my_vec, de::type_mismatch );
}

TEST_F(DictionaryTestEq, eqStringVecRight) {
    EXPECT_THROW(  my_string!=cdict_["my_vec"], de::type_mismatch );
    EXPECT_THROW(  my_string==cdict_["my_vec"], de::type_mismatch );
}

TEST_F(DictionaryTestEq, eqStringVecBoth) {
    EXPECT_THROW(  cdict_["my_string"]!=cdict_["my_vec"], de::type_mismatch );
    EXPECT_THROW(  cdict_["my_string"]==cdict_["my_vec"], de::type_mismatch );
}

// Equalities between different types my_string:my_pair
TEST_F(DictionaryTestEq, eqStringPairLeft) {
    EXPECT_THROW(  cdict_["my_string"]!=my_pair, de::type_mismatch );
    EXPECT_THROW(  cdict_["my_string"]==my_pair, de::type_mismatch );
}

TEST_F(DictionaryTestEq, eqStringPairRight) {
    EXPECT_THROW(  my_string!=cdict_["my_pair"], de::type_mismatch );
    EXPECT_THROW(  my_string==cdict_["my_pair"], de::type_mismatch );
}

TEST_F(DictionaryTestEq, eqStringPairBoth) {
    EXPECT_THROW(  cdict_["my_string"]!=cdict_["my_pair"], de::type_mismatch );
    EXPECT_THROW(  cdict_["my_string"]==cdict_["my_pair"], de::type_mismatch );
}

// Equalities within same type my_vec
TEST_F(DictionaryTestEq, eqVecLeft) {
    // Same values:
    EXPECT_TRUE(  cdict_["my_vec"]==my_vec );
    EXPECT_FALSE( cdict_["my_vec"]!=my_vec );
    // Different values:
    EXPECT_TRUE(  cdict_["my_vec"]!=my_vec1 );
    EXPECT_FALSE( cdict_["my_vec"]==my_vec1 );
}

TEST_F(DictionaryTestEq, eqVecRight) {
    // Same values:
    EXPECT_TRUE(  my_vec==cdict_["my_vec"] );
    EXPECT_FALSE( my_vec!=cdict_["my_vec"] );
    // Different values:
    EXPECT_TRUE(  my_vec!=cdict_["my_vec1"] );
    EXPECT_FALSE( my_vec==cdict_["my_vec1"] );
}

TEST_F(DictionaryTestEq, eqVecBoth) {
    // Same values:
    EXPECT_TRUE(  cdict_["my_vec"]==cdict_["my_vec"] );
    EXPECT_FALSE( cdict_["my_vec"]!=cdict_["my_vec"] );
    // Different values:
    EXPECT_TRUE(  cdict_["my_vec"]!=cdict_["my_vec1"] );
    EXPECT_FALSE( cdict_["my_vec"]==cdict_["my_vec1"] );
}

// Equalities between different types my_vec:my_pair
TEST_F(DictionaryTestEq, eqVecPairLeft) {
    EXPECT_THROW(  cdict_["my_vec"]!=my_pair, de::type_mismatch );
    EXPECT_THROW(  cdict_["my_vec"]==my_pair, de::type_mismatch );
}

TEST_F(DictionaryTestEq, eqVecPairRight) {
    EXPECT_THROW(  my_vec!=cdict_["my_pair"], de::type_mismatch );
    EXPECT_THROW(  my_vec==cdict_["my_pair"], de::type_mismatch );
}

TEST_F(DictionaryTestEq, eqVecPairBoth) {
    EXPECT_THROW(  cdict_["my_vec"]!=cdict_["my_pair"], de::type_mismatch );
    EXPECT_THROW(  cdict_["my_vec"]==cdict_["my_pair"], de::type_mismatch );
}

// Equalities within same type my_pair
TEST_F(DictionaryTestEq, eqPairLeft) {
    // Same values:
    EXPECT_TRUE(  cdict_["my_pair"]==my_pair );
    EXPECT_FALSE( cdict_["my_pair"]!=my_pair );
    // Different values:
    EXPECT_TRUE(  cdict_["my_pair"]!=my_pair1 );
    EXPECT_FALSE( cdict_["my_pair"]==my_pair1 );
}

TEST_F(DictionaryTestEq, eqPairRight) {
    // Same values:
    EXPECT_TRUE(  my_pair==cdict_["my_pair"] );
    EXPECT_FALSE( my_pair!=cdict_["my_pair"] );
    // Different values:
    EXPECT_TRUE(  my_pair!=cdict_["my_pair1"] );
    EXPECT_FALSE( my_pair==cdict_["my_pair1"] );
}

TEST_F(DictionaryTestEq, eqPairBoth) {
    // Same values:
    EXPECT_TRUE(  cdict_["my_pair"]==cdict_["my_pair"] );
    EXPECT_FALSE( cdict_["my_pair"]!=cdict_["my_pair"] );
    // Different values:
    EXPECT_TRUE(  cdict_["my_pair"]!=cdict_["my_pair1"] );
    EXPECT_FALSE( cdict_["my_pair"]==cdict_["my_pair1"] );
}

