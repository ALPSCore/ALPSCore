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
TEST_F(DictionaryTestEq, eqNegLong) {
    // Same values:
    EXPECT_TRUE(  +neg_long==cdict_["neg_long"] );
    EXPECT_FALSE( +neg_long!=cdict_["neg_long"] );
    EXPECT_TRUE(  cdict_["neg_long"]==+neg_long );
    EXPECT_FALSE( cdict_["neg_long"]!=+neg_long );
    EXPECT_TRUE(  cdict_["neg_long"]==cdict_["neg_long"] );
    EXPECT_FALSE( cdict_["neg_long"]!=cdict_["neg_long"] );
    // Different values:
    EXPECT_TRUE(  +neg_long!=cdict_["neg_long1"] );
    EXPECT_FALSE( +neg_long==cdict_["neg_long1"] );
    EXPECT_TRUE(  cdict_["neg_long"]!=+neg_long1 );
    EXPECT_FALSE( cdict_["neg_long"]==+neg_long1 );
    EXPECT_TRUE(  cdict_["neg_long"]!=cdict_["neg_long1"] );
    EXPECT_FALSE( cdict_["neg_long"]==cdict_["neg_long1"] );
}

// Equalities between domains neg_long:neg_int
TEST_F(DictionaryTestEq, eqNegLongNegInt) {
    EXPECT_TRUE(  +neg_long!=cdict_["neg_int"] );
    EXPECT_FALSE( +neg_long==cdict_["neg_int"] );
    EXPECT_TRUE(  cdict_["neg_long"]!=+neg_int );
    EXPECT_FALSE( cdict_["neg_long"]==+neg_int );
    EXPECT_TRUE(  cdict_["neg_long"]!=cdict_["neg_int"] );
    EXPECT_FALSE( cdict_["neg_long"]==cdict_["neg_int"] );
    EXPECT_TRUE(  +neg_long!=cdict_["neg_long_is"] );
    EXPECT_FALSE( +neg_long==cdict_["neg_long_is"] );
    EXPECT_TRUE(  cdict_["neg_long"]!=+neg_long_is );
    EXPECT_FALSE( cdict_["neg_long"]==+neg_long_is );
    EXPECT_TRUE(  cdict_["neg_long"]!=cdict_["neg_long_is"] );
    EXPECT_FALSE( cdict_["neg_long"]==cdict_["neg_long_is"] );
}

// Equalities between domains neg_long:pos_int
TEST_F(DictionaryTestEq, eqNegLongPosInt) {
    EXPECT_TRUE(  +neg_long!=cdict_["pos_int"] );
    EXPECT_FALSE( +neg_long==cdict_["pos_int"] );
    EXPECT_TRUE(  cdict_["neg_long"]!=+pos_int );
    EXPECT_FALSE( cdict_["neg_long"]==+pos_int );
    EXPECT_TRUE(  cdict_["neg_long"]!=cdict_["pos_int"] );
    EXPECT_FALSE( cdict_["neg_long"]==cdict_["pos_int"] );
    EXPECT_TRUE(  +neg_long!=cdict_["uint_is"] );
    EXPECT_FALSE( +neg_long==cdict_["uint_is"] );
    EXPECT_TRUE(  cdict_["neg_long"]!=+uint_is );
    EXPECT_FALSE( cdict_["neg_long"]==+uint_is );
    EXPECT_TRUE(  cdict_["neg_long"]!=cdict_["uint_is"] );
    EXPECT_FALSE( cdict_["neg_long"]==cdict_["uint_is"] );
    EXPECT_TRUE(  +neg_long!=cdict_["pos_long_is"] );
    EXPECT_FALSE( +neg_long==cdict_["pos_long_is"] );
    EXPECT_TRUE(  cdict_["neg_long"]!=+pos_long_is );
    EXPECT_FALSE( cdict_["neg_long"]==+pos_long_is );
    EXPECT_TRUE(  cdict_["neg_long"]!=cdict_["pos_long_is"] );
    EXPECT_FALSE( cdict_["neg_long"]==cdict_["pos_long_is"] );
    EXPECT_TRUE(  +neg_long!=cdict_["ulong_is"] );
    EXPECT_FALSE( +neg_long==cdict_["ulong_is"] );
    EXPECT_TRUE(  cdict_["neg_long"]!=+ulong_is );
    EXPECT_FALSE( cdict_["neg_long"]==+ulong_is );
    EXPECT_TRUE(  cdict_["neg_long"]!=cdict_["ulong_is"] );
    EXPECT_FALSE( cdict_["neg_long"]==cdict_["ulong_is"] );
}

// Equalities between domains neg_long:pos_uint
TEST_F(DictionaryTestEq, eqNegLongPosUint) {
    EXPECT_TRUE(  +neg_long!=cdict_["pos_uint"] );
    EXPECT_FALSE( +neg_long==cdict_["pos_uint"] );
    EXPECT_TRUE(  cdict_["neg_long"]!=+pos_uint );
    EXPECT_FALSE( cdict_["neg_long"]==+pos_uint );
    EXPECT_TRUE(  cdict_["neg_long"]!=cdict_["pos_uint"] );
    EXPECT_FALSE( cdict_["neg_long"]==cdict_["pos_uint"] );
}

// Equalities between domains neg_long:pos_long
TEST_F(DictionaryTestEq, eqNegLongPosLong) {
    EXPECT_TRUE(  +neg_long!=cdict_["pos_long"] );
    EXPECT_FALSE( +neg_long==cdict_["pos_long"] );
    EXPECT_TRUE(  cdict_["neg_long"]!=+pos_long );
    EXPECT_FALSE( cdict_["neg_long"]==+pos_long );
    EXPECT_TRUE(  cdict_["neg_long"]!=cdict_["pos_long"] );
    EXPECT_FALSE( cdict_["neg_long"]==cdict_["pos_long"] );
    EXPECT_TRUE(  +neg_long!=cdict_["ulong_ls"] );
    EXPECT_FALSE( +neg_long==cdict_["ulong_ls"] );
    EXPECT_TRUE(  cdict_["neg_long"]!=+ulong_ls );
    EXPECT_FALSE( cdict_["neg_long"]==+ulong_ls );
    EXPECT_TRUE(  cdict_["neg_long"]!=cdict_["ulong_ls"] );
    EXPECT_FALSE( cdict_["neg_long"]==cdict_["ulong_ls"] );
}

// Equalities between domains neg_long:pos_ulong
TEST_F(DictionaryTestEq, eqNegLongULong) {
    EXPECT_TRUE(  +neg_long!=cdict_["pos_ulong"] );
    EXPECT_FALSE( +neg_long==cdict_["pos_ulong"] );
    EXPECT_TRUE(  cdict_["neg_long"]!=+pos_ulong );
    EXPECT_FALSE( cdict_["neg_long"]==+pos_ulong );
    EXPECT_TRUE(  cdict_["neg_long"]!=cdict_["pos_ulong"] );
    EXPECT_FALSE( cdict_["neg_long"]==cdict_["pos_ulong"] );
}

// Equalities within domain neg_int
TEST_F(DictionaryTestEq, eqNegInt) {
    // Same values:
    EXPECT_TRUE(  +neg_int==cdict_["neg_int"] );
    EXPECT_FALSE( +neg_int!=cdict_["neg_int"] );
    EXPECT_TRUE(  cdict_["neg_int"]==+neg_int );
    EXPECT_FALSE( cdict_["neg_int"]!=+neg_int );
    EXPECT_TRUE(  cdict_["neg_int"]==cdict_["neg_int"] );
    EXPECT_FALSE( cdict_["neg_int"]!=cdict_["neg_int"] );
    // Different values:
    EXPECT_TRUE(  +neg_int!=cdict_["neg_int1"] );
    EXPECT_FALSE( +neg_int==cdict_["neg_int1"] );
    EXPECT_TRUE(  cdict_["neg_int"]!=+neg_int1 );
    EXPECT_FALSE( cdict_["neg_int"]==+neg_int1 );
    EXPECT_TRUE(  cdict_["neg_int"]!=cdict_["neg_int1"] );
    EXPECT_FALSE( cdict_["neg_int"]==cdict_["neg_int1"] );
    // Same values:
    EXPECT_TRUE(  +neg_int==cdict_["neg_long_is"] );
    EXPECT_FALSE( +neg_int!=cdict_["neg_long_is"] );
    EXPECT_TRUE(  cdict_["neg_int"]==+neg_long_is );
    EXPECT_FALSE( cdict_["neg_int"]!=+neg_long_is );
    EXPECT_TRUE(  cdict_["neg_int"]==cdict_["neg_long_is"] );
    EXPECT_FALSE( cdict_["neg_int"]!=cdict_["neg_long_is"] );
    // Different values:
    EXPECT_TRUE(  +neg_int!=cdict_["neg_long_is1"] );
    EXPECT_FALSE( +neg_int==cdict_["neg_long_is1"] );
    EXPECT_TRUE(  cdict_["neg_int"]!=+neg_long_is1 );
    EXPECT_FALSE( cdict_["neg_int"]==+neg_long_is1 );
    EXPECT_TRUE(  cdict_["neg_int"]!=cdict_["neg_long_is1"] );
    EXPECT_FALSE( cdict_["neg_int"]==cdict_["neg_long_is1"] );
    // Same values:
    EXPECT_TRUE(  +neg_long_is==cdict_["neg_long_is"] );
    EXPECT_FALSE( +neg_long_is!=cdict_["neg_long_is"] );
    EXPECT_TRUE(  cdict_["neg_long_is"]==+neg_long_is );
    EXPECT_FALSE( cdict_["neg_long_is"]!=+neg_long_is );
    EXPECT_TRUE(  cdict_["neg_long_is"]==cdict_["neg_long_is"] );
    EXPECT_FALSE( cdict_["neg_long_is"]!=cdict_["neg_long_is"] );
    // Different values:
    EXPECT_TRUE(  +neg_long_is!=cdict_["neg_long_is1"] );
    EXPECT_FALSE( +neg_long_is==cdict_["neg_long_is1"] );
    EXPECT_TRUE(  cdict_["neg_long_is"]!=+neg_long_is1 );
    EXPECT_FALSE( cdict_["neg_long_is"]==+neg_long_is1 );
    EXPECT_TRUE(  cdict_["neg_long_is"]!=cdict_["neg_long_is1"] );
    EXPECT_FALSE( cdict_["neg_long_is"]==cdict_["neg_long_is1"] );
}

// Equalities between domains neg_int:pos_int
TEST_F(DictionaryTestEq, eqNegIntPosInt) {
    EXPECT_TRUE(  +neg_int!=cdict_["pos_int"] );
    EXPECT_FALSE( +neg_int==cdict_["pos_int"] );
    EXPECT_TRUE(  cdict_["neg_int"]!=+pos_int );
    EXPECT_FALSE( cdict_["neg_int"]==+pos_int );
    EXPECT_TRUE(  cdict_["neg_int"]!=cdict_["pos_int"] );
    EXPECT_FALSE( cdict_["neg_int"]==cdict_["pos_int"] );
    EXPECT_TRUE(  +neg_int!=cdict_["uint_is"] );
    EXPECT_FALSE( +neg_int==cdict_["uint_is"] );
    EXPECT_TRUE(  cdict_["neg_int"]!=+uint_is );
    EXPECT_FALSE( cdict_["neg_int"]==+uint_is );
    EXPECT_TRUE(  cdict_["neg_int"]!=cdict_["uint_is"] );
    EXPECT_FALSE( cdict_["neg_int"]==cdict_["uint_is"] );
    EXPECT_TRUE(  +neg_int!=cdict_["pos_long_is"] );
    EXPECT_FALSE( +neg_int==cdict_["pos_long_is"] );
    EXPECT_TRUE(  cdict_["neg_int"]!=+pos_long_is );
    EXPECT_FALSE( cdict_["neg_int"]==+pos_long_is );
    EXPECT_TRUE(  cdict_["neg_int"]!=cdict_["pos_long_is"] );
    EXPECT_FALSE( cdict_["neg_int"]==cdict_["pos_long_is"] );
    EXPECT_TRUE(  +neg_int!=cdict_["ulong_is"] );
    EXPECT_FALSE( +neg_int==cdict_["ulong_is"] );
    EXPECT_TRUE(  cdict_["neg_int"]!=+ulong_is );
    EXPECT_FALSE( cdict_["neg_int"]==+ulong_is );
    EXPECT_TRUE(  cdict_["neg_int"]!=cdict_["ulong_is"] );
    EXPECT_FALSE( cdict_["neg_int"]==cdict_["ulong_is"] );
    EXPECT_TRUE(  +neg_long_is!=cdict_["pos_int"] );
    EXPECT_FALSE( +neg_long_is==cdict_["pos_int"] );
    EXPECT_TRUE(  cdict_["neg_long_is"]!=+pos_int );
    EXPECT_FALSE( cdict_["neg_long_is"]==+pos_int );
    EXPECT_TRUE(  cdict_["neg_long_is"]!=cdict_["pos_int"] );
    EXPECT_FALSE( cdict_["neg_long_is"]==cdict_["pos_int"] );
    EXPECT_TRUE(  +neg_long_is!=cdict_["uint_is"] );
    EXPECT_FALSE( +neg_long_is==cdict_["uint_is"] );
    EXPECT_TRUE(  cdict_["neg_long_is"]!=+uint_is );
    EXPECT_FALSE( cdict_["neg_long_is"]==+uint_is );
    EXPECT_TRUE(  cdict_["neg_long_is"]!=cdict_["uint_is"] );
    EXPECT_FALSE( cdict_["neg_long_is"]==cdict_["uint_is"] );
    EXPECT_TRUE(  +neg_long_is!=cdict_["pos_long_is"] );
    EXPECT_FALSE( +neg_long_is==cdict_["pos_long_is"] );
    EXPECT_TRUE(  cdict_["neg_long_is"]!=+pos_long_is );
    EXPECT_FALSE( cdict_["neg_long_is"]==+pos_long_is );
    EXPECT_TRUE(  cdict_["neg_long_is"]!=cdict_["pos_long_is"] );
    EXPECT_FALSE( cdict_["neg_long_is"]==cdict_["pos_long_is"] );
    EXPECT_TRUE(  +neg_long_is!=cdict_["ulong_is"] );
    EXPECT_FALSE( +neg_long_is==cdict_["ulong_is"] );
    EXPECT_TRUE(  cdict_["neg_long_is"]!=+ulong_is );
    EXPECT_FALSE( cdict_["neg_long_is"]==+ulong_is );
    EXPECT_TRUE(  cdict_["neg_long_is"]!=cdict_["ulong_is"] );
    EXPECT_FALSE( cdict_["neg_long_is"]==cdict_["ulong_is"] );
}

// Equalities between domains neg_int:pos_uint
TEST_F(DictionaryTestEq, eqNegIntPosUint) {
    EXPECT_TRUE(  +neg_int!=cdict_["pos_uint"] );
    EXPECT_FALSE( +neg_int==cdict_["pos_uint"] );
    EXPECT_TRUE(  cdict_["neg_int"]!=+pos_uint );
    EXPECT_FALSE( cdict_["neg_int"]==+pos_uint );
    EXPECT_TRUE(  cdict_["neg_int"]!=cdict_["pos_uint"] );
    EXPECT_FALSE( cdict_["neg_int"]==cdict_["pos_uint"] );
    EXPECT_TRUE(  +neg_long_is!=cdict_["pos_uint"] );
    EXPECT_FALSE( +neg_long_is==cdict_["pos_uint"] );
    EXPECT_TRUE(  cdict_["neg_long_is"]!=+pos_uint );
    EXPECT_FALSE( cdict_["neg_long_is"]==+pos_uint );
    EXPECT_TRUE(  cdict_["neg_long_is"]!=cdict_["pos_uint"] );
    EXPECT_FALSE( cdict_["neg_long_is"]==cdict_["pos_uint"] );
}

// Equalities between domains neg_int:pos_long
TEST_F(DictionaryTestEq, eqNegIntPosLong) {
    EXPECT_TRUE(  +neg_int!=cdict_["pos_long"] );
    EXPECT_FALSE( +neg_int==cdict_["pos_long"] );
    EXPECT_TRUE(  cdict_["neg_int"]!=+pos_long );
    EXPECT_FALSE( cdict_["neg_int"]==+pos_long );
    EXPECT_TRUE(  cdict_["neg_int"]!=cdict_["pos_long"] );
    EXPECT_FALSE( cdict_["neg_int"]==cdict_["pos_long"] );
    EXPECT_TRUE(  +neg_int!=cdict_["ulong_ls"] );
    EXPECT_FALSE( +neg_int==cdict_["ulong_ls"] );
    EXPECT_TRUE(  cdict_["neg_int"]!=+ulong_ls );
    EXPECT_FALSE( cdict_["neg_int"]==+ulong_ls );
    EXPECT_TRUE(  cdict_["neg_int"]!=cdict_["ulong_ls"] );
    EXPECT_FALSE( cdict_["neg_int"]==cdict_["ulong_ls"] );
    EXPECT_TRUE(  +neg_long_is!=cdict_["pos_long"] );
    EXPECT_FALSE( +neg_long_is==cdict_["pos_long"] );
    EXPECT_TRUE(  cdict_["neg_long_is"]!=+pos_long );
    EXPECT_FALSE( cdict_["neg_long_is"]==+pos_long );
    EXPECT_TRUE(  cdict_["neg_long_is"]!=cdict_["pos_long"] );
    EXPECT_FALSE( cdict_["neg_long_is"]==cdict_["pos_long"] );
    EXPECT_TRUE(  +neg_long_is!=cdict_["ulong_ls"] );
    EXPECT_FALSE( +neg_long_is==cdict_["ulong_ls"] );
    EXPECT_TRUE(  cdict_["neg_long_is"]!=+ulong_ls );
    EXPECT_FALSE( cdict_["neg_long_is"]==+ulong_ls );
    EXPECT_TRUE(  cdict_["neg_long_is"]!=cdict_["ulong_ls"] );
    EXPECT_FALSE( cdict_["neg_long_is"]==cdict_["ulong_ls"] );
}

// Equalities between domains neg_int:pos_ulong
TEST_F(DictionaryTestEq, eqNegIntULong) {
    EXPECT_TRUE(  +neg_int!=cdict_["pos_ulong"] );
    EXPECT_FALSE( +neg_int==cdict_["pos_ulong"] );
    EXPECT_TRUE(  cdict_["neg_int"]!=+pos_ulong );
    EXPECT_FALSE( cdict_["neg_int"]==+pos_ulong );
    EXPECT_TRUE(  cdict_["neg_int"]!=cdict_["pos_ulong"] );
    EXPECT_FALSE( cdict_["neg_int"]==cdict_["pos_ulong"] );
    EXPECT_TRUE(  +neg_long_is!=cdict_["pos_ulong"] );
    EXPECT_FALSE( +neg_long_is==cdict_["pos_ulong"] );
    EXPECT_TRUE(  cdict_["neg_long_is"]!=+pos_ulong );
    EXPECT_FALSE( cdict_["neg_long_is"]==+pos_ulong );
    EXPECT_TRUE(  cdict_["neg_long_is"]!=cdict_["pos_ulong"] );
    EXPECT_FALSE( cdict_["neg_long_is"]==cdict_["pos_ulong"] );
}

// Equalities within domain pos_int
TEST_F(DictionaryTestEq, eqPosInt) {
    // Same values:
    EXPECT_TRUE(  +pos_int==cdict_["pos_int"] );
    EXPECT_FALSE( +pos_int!=cdict_["pos_int"] );
    EXPECT_TRUE(  cdict_["pos_int"]==+pos_int );
    EXPECT_FALSE( cdict_["pos_int"]!=+pos_int );
    EXPECT_TRUE(  cdict_["pos_int"]==cdict_["pos_int"] );
    EXPECT_FALSE( cdict_["pos_int"]!=cdict_["pos_int"] );
    // Different values:
    EXPECT_TRUE(  +pos_int!=cdict_["pos_int1"] );
    EXPECT_FALSE( +pos_int==cdict_["pos_int1"] );
    EXPECT_TRUE(  cdict_["pos_int"]!=+pos_int1 );
    EXPECT_FALSE( cdict_["pos_int"]==+pos_int1 );
    EXPECT_TRUE(  cdict_["pos_int"]!=cdict_["pos_int1"] );
    EXPECT_FALSE( cdict_["pos_int"]==cdict_["pos_int1"] );
    // Same values:
    EXPECT_TRUE(  +pos_int==cdict_["uint_is"] );
    EXPECT_FALSE( +pos_int!=cdict_["uint_is"] );
    EXPECT_TRUE(  cdict_["pos_int"]==+uint_is );
    EXPECT_FALSE( cdict_["pos_int"]!=+uint_is );
    EXPECT_TRUE(  cdict_["pos_int"]==cdict_["uint_is"] );
    EXPECT_FALSE( cdict_["pos_int"]!=cdict_["uint_is"] );
    // Different values:
    EXPECT_TRUE(  +pos_int!=cdict_["uint_is1"] );
    EXPECT_FALSE( +pos_int==cdict_["uint_is1"] );
    EXPECT_TRUE(  cdict_["pos_int"]!=+uint_is1 );
    EXPECT_FALSE( cdict_["pos_int"]==+uint_is1 );
    EXPECT_TRUE(  cdict_["pos_int"]!=cdict_["uint_is1"] );
    EXPECT_FALSE( cdict_["pos_int"]==cdict_["uint_is1"] );
    // Same values:
    EXPECT_TRUE(  +pos_int==cdict_["pos_long_is"] );
    EXPECT_FALSE( +pos_int!=cdict_["pos_long_is"] );
    EXPECT_TRUE(  cdict_["pos_int"]==+pos_long_is );
    EXPECT_FALSE( cdict_["pos_int"]!=+pos_long_is );
    EXPECT_TRUE(  cdict_["pos_int"]==cdict_["pos_long_is"] );
    EXPECT_FALSE( cdict_["pos_int"]!=cdict_["pos_long_is"] );
    // Different values:
    EXPECT_TRUE(  +pos_int!=cdict_["pos_long_is1"] );
    EXPECT_FALSE( +pos_int==cdict_["pos_long_is1"] );
    EXPECT_TRUE(  cdict_["pos_int"]!=+pos_long_is1 );
    EXPECT_FALSE( cdict_["pos_int"]==+pos_long_is1 );
    EXPECT_TRUE(  cdict_["pos_int"]!=cdict_["pos_long_is1"] );
    EXPECT_FALSE( cdict_["pos_int"]==cdict_["pos_long_is1"] );
    // Same values:
    EXPECT_TRUE(  +pos_int==cdict_["ulong_is"] );
    EXPECT_FALSE( +pos_int!=cdict_["ulong_is"] );
    EXPECT_TRUE(  cdict_["pos_int"]==+ulong_is );
    EXPECT_FALSE( cdict_["pos_int"]!=+ulong_is );
    EXPECT_TRUE(  cdict_["pos_int"]==cdict_["ulong_is"] );
    EXPECT_FALSE( cdict_["pos_int"]!=cdict_["ulong_is"] );
    // Different values:
    EXPECT_TRUE(  +pos_int!=cdict_["ulong_is1"] );
    EXPECT_FALSE( +pos_int==cdict_["ulong_is1"] );
    EXPECT_TRUE(  cdict_["pos_int"]!=+ulong_is1 );
    EXPECT_FALSE( cdict_["pos_int"]==+ulong_is1 );
    EXPECT_TRUE(  cdict_["pos_int"]!=cdict_["ulong_is1"] );
    EXPECT_FALSE( cdict_["pos_int"]==cdict_["ulong_is1"] );
    // Same values:
    EXPECT_TRUE(  +uint_is==cdict_["uint_is"] );
    EXPECT_FALSE( +uint_is!=cdict_["uint_is"] );
    EXPECT_TRUE(  cdict_["uint_is"]==+uint_is );
    EXPECT_FALSE( cdict_["uint_is"]!=+uint_is );
    EXPECT_TRUE(  cdict_["uint_is"]==cdict_["uint_is"] );
    EXPECT_FALSE( cdict_["uint_is"]!=cdict_["uint_is"] );
    // Different values:
    EXPECT_TRUE(  +uint_is!=cdict_["uint_is1"] );
    EXPECT_FALSE( +uint_is==cdict_["uint_is1"] );
    EXPECT_TRUE(  cdict_["uint_is"]!=+uint_is1 );
    EXPECT_FALSE( cdict_["uint_is"]==+uint_is1 );
    EXPECT_TRUE(  cdict_["uint_is"]!=cdict_["uint_is1"] );
    EXPECT_FALSE( cdict_["uint_is"]==cdict_["uint_is1"] );
    // Same values:
    EXPECT_TRUE(  +uint_is==cdict_["pos_long_is"] );
    EXPECT_FALSE( +uint_is!=cdict_["pos_long_is"] );
    EXPECT_TRUE(  cdict_["uint_is"]==+pos_long_is );
    EXPECT_FALSE( cdict_["uint_is"]!=+pos_long_is );
    EXPECT_TRUE(  cdict_["uint_is"]==cdict_["pos_long_is"] );
    EXPECT_FALSE( cdict_["uint_is"]!=cdict_["pos_long_is"] );
    // Different values:
    EXPECT_TRUE(  +uint_is!=cdict_["pos_long_is1"] );
    EXPECT_FALSE( +uint_is==cdict_["pos_long_is1"] );
    EXPECT_TRUE(  cdict_["uint_is"]!=+pos_long_is1 );
    EXPECT_FALSE( cdict_["uint_is"]==+pos_long_is1 );
    EXPECT_TRUE(  cdict_["uint_is"]!=cdict_["pos_long_is1"] );
    EXPECT_FALSE( cdict_["uint_is"]==cdict_["pos_long_is1"] );
    // Same values:
    EXPECT_TRUE(  +uint_is==cdict_["ulong_is"] );
    EXPECT_FALSE( +uint_is!=cdict_["ulong_is"] );
    EXPECT_TRUE(  cdict_["uint_is"]==+ulong_is );
    EXPECT_FALSE( cdict_["uint_is"]!=+ulong_is );
    EXPECT_TRUE(  cdict_["uint_is"]==cdict_["ulong_is"] );
    EXPECT_FALSE( cdict_["uint_is"]!=cdict_["ulong_is"] );
    // Different values:
    EXPECT_TRUE(  +uint_is!=cdict_["ulong_is1"] );
    EXPECT_FALSE( +uint_is==cdict_["ulong_is1"] );
    EXPECT_TRUE(  cdict_["uint_is"]!=+ulong_is1 );
    EXPECT_FALSE( cdict_["uint_is"]==+ulong_is1 );
    EXPECT_TRUE(  cdict_["uint_is"]!=cdict_["ulong_is1"] );
    EXPECT_FALSE( cdict_["uint_is"]==cdict_["ulong_is1"] );
    // Same values:
    EXPECT_TRUE(  +pos_long_is==cdict_["pos_long_is"] );
    EXPECT_FALSE( +pos_long_is!=cdict_["pos_long_is"] );
    EXPECT_TRUE(  cdict_["pos_long_is"]==+pos_long_is );
    EXPECT_FALSE( cdict_["pos_long_is"]!=+pos_long_is );
    EXPECT_TRUE(  cdict_["pos_long_is"]==cdict_["pos_long_is"] );
    EXPECT_FALSE( cdict_["pos_long_is"]!=cdict_["pos_long_is"] );
    // Different values:
    EXPECT_TRUE(  +pos_long_is!=cdict_["pos_long_is1"] );
    EXPECT_FALSE( +pos_long_is==cdict_["pos_long_is1"] );
    EXPECT_TRUE(  cdict_["pos_long_is"]!=+pos_long_is1 );
    EXPECT_FALSE( cdict_["pos_long_is"]==+pos_long_is1 );
    EXPECT_TRUE(  cdict_["pos_long_is"]!=cdict_["pos_long_is1"] );
    EXPECT_FALSE( cdict_["pos_long_is"]==cdict_["pos_long_is1"] );
    // Same values:
    EXPECT_TRUE(  +pos_long_is==cdict_["ulong_is"] );
    EXPECT_FALSE( +pos_long_is!=cdict_["ulong_is"] );
    EXPECT_TRUE(  cdict_["pos_long_is"]==+ulong_is );
    EXPECT_FALSE( cdict_["pos_long_is"]!=+ulong_is );
    EXPECT_TRUE(  cdict_["pos_long_is"]==cdict_["ulong_is"] );
    EXPECT_FALSE( cdict_["pos_long_is"]!=cdict_["ulong_is"] );
    // Different values:
    EXPECT_TRUE(  +pos_long_is!=cdict_["ulong_is1"] );
    EXPECT_FALSE( +pos_long_is==cdict_["ulong_is1"] );
    EXPECT_TRUE(  cdict_["pos_long_is"]!=+ulong_is1 );
    EXPECT_FALSE( cdict_["pos_long_is"]==+ulong_is1 );
    EXPECT_TRUE(  cdict_["pos_long_is"]!=cdict_["ulong_is1"] );
    EXPECT_FALSE( cdict_["pos_long_is"]==cdict_["ulong_is1"] );
    // Same values:
    EXPECT_TRUE(  +ulong_is==cdict_["ulong_is"] );
    EXPECT_FALSE( +ulong_is!=cdict_["ulong_is"] );
    EXPECT_TRUE(  cdict_["ulong_is"]==+ulong_is );
    EXPECT_FALSE( cdict_["ulong_is"]!=+ulong_is );
    EXPECT_TRUE(  cdict_["ulong_is"]==cdict_["ulong_is"] );
    EXPECT_FALSE( cdict_["ulong_is"]!=cdict_["ulong_is"] );
    // Different values:
    EXPECT_TRUE(  +ulong_is!=cdict_["ulong_is1"] );
    EXPECT_FALSE( +ulong_is==cdict_["ulong_is1"] );
    EXPECT_TRUE(  cdict_["ulong_is"]!=+ulong_is1 );
    EXPECT_FALSE( cdict_["ulong_is"]==+ulong_is1 );
    EXPECT_TRUE(  cdict_["ulong_is"]!=cdict_["ulong_is1"] );
    EXPECT_FALSE( cdict_["ulong_is"]==cdict_["ulong_is1"] );
}

// Equalities between domains pos_int:pos_uint
TEST_F(DictionaryTestEq, eqPosIntPosUint) {
    EXPECT_TRUE(  +pos_int!=cdict_["pos_uint"] );
    EXPECT_FALSE( +pos_int==cdict_["pos_uint"] );
    EXPECT_TRUE(  cdict_["pos_int"]!=+pos_uint );
    EXPECT_FALSE( cdict_["pos_int"]==+pos_uint );
    EXPECT_TRUE(  cdict_["pos_int"]!=cdict_["pos_uint"] );
    EXPECT_FALSE( cdict_["pos_int"]==cdict_["pos_uint"] );
    EXPECT_TRUE(  +uint_is!=cdict_["pos_uint"] );
    EXPECT_FALSE( +uint_is==cdict_["pos_uint"] );
    EXPECT_TRUE(  cdict_["uint_is"]!=+pos_uint );
    EXPECT_FALSE( cdict_["uint_is"]==+pos_uint );
    EXPECT_TRUE(  cdict_["uint_is"]!=cdict_["pos_uint"] );
    EXPECT_FALSE( cdict_["uint_is"]==cdict_["pos_uint"] );
    EXPECT_TRUE(  +pos_long_is!=cdict_["pos_uint"] );
    EXPECT_FALSE( +pos_long_is==cdict_["pos_uint"] );
    EXPECT_TRUE(  cdict_["pos_long_is"]!=+pos_uint );
    EXPECT_FALSE( cdict_["pos_long_is"]==+pos_uint );
    EXPECT_TRUE(  cdict_["pos_long_is"]!=cdict_["pos_uint"] );
    EXPECT_FALSE( cdict_["pos_long_is"]==cdict_["pos_uint"] );
    EXPECT_TRUE(  +ulong_is!=cdict_["pos_uint"] );
    EXPECT_FALSE( +ulong_is==cdict_["pos_uint"] );
    EXPECT_TRUE(  cdict_["ulong_is"]!=+pos_uint );
    EXPECT_FALSE( cdict_["ulong_is"]==+pos_uint );
    EXPECT_TRUE(  cdict_["ulong_is"]!=cdict_["pos_uint"] );
    EXPECT_FALSE( cdict_["ulong_is"]==cdict_["pos_uint"] );
}

// Equalities between domains pos_int:pos_long
TEST_F(DictionaryTestEq, eqPosIntPosLong) {
    EXPECT_TRUE(  +pos_int!=cdict_["pos_long"] );
    EXPECT_FALSE( +pos_int==cdict_["pos_long"] );
    EXPECT_TRUE(  cdict_["pos_int"]!=+pos_long );
    EXPECT_FALSE( cdict_["pos_int"]==+pos_long );
    EXPECT_TRUE(  cdict_["pos_int"]!=cdict_["pos_long"] );
    EXPECT_FALSE( cdict_["pos_int"]==cdict_["pos_long"] );
    EXPECT_TRUE(  +pos_int!=cdict_["ulong_ls"] );
    EXPECT_FALSE( +pos_int==cdict_["ulong_ls"] );
    EXPECT_TRUE(  cdict_["pos_int"]!=+ulong_ls );
    EXPECT_FALSE( cdict_["pos_int"]==+ulong_ls );
    EXPECT_TRUE(  cdict_["pos_int"]!=cdict_["ulong_ls"] );
    EXPECT_FALSE( cdict_["pos_int"]==cdict_["ulong_ls"] );
    EXPECT_TRUE(  +uint_is!=cdict_["pos_long"] );
    EXPECT_FALSE( +uint_is==cdict_["pos_long"] );
    EXPECT_TRUE(  cdict_["uint_is"]!=+pos_long );
    EXPECT_FALSE( cdict_["uint_is"]==+pos_long );
    EXPECT_TRUE(  cdict_["uint_is"]!=cdict_["pos_long"] );
    EXPECT_FALSE( cdict_["uint_is"]==cdict_["pos_long"] );
    EXPECT_TRUE(  +uint_is!=cdict_["ulong_ls"] );
    EXPECT_FALSE( +uint_is==cdict_["ulong_ls"] );
    EXPECT_TRUE(  cdict_["uint_is"]!=+ulong_ls );
    EXPECT_FALSE( cdict_["uint_is"]==+ulong_ls );
    EXPECT_TRUE(  cdict_["uint_is"]!=cdict_["ulong_ls"] );
    EXPECT_FALSE( cdict_["uint_is"]==cdict_["ulong_ls"] );
    EXPECT_TRUE(  +pos_long_is!=cdict_["pos_long"] );
    EXPECT_FALSE( +pos_long_is==cdict_["pos_long"] );
    EXPECT_TRUE(  cdict_["pos_long_is"]!=+pos_long );
    EXPECT_FALSE( cdict_["pos_long_is"]==+pos_long );
    EXPECT_TRUE(  cdict_["pos_long_is"]!=cdict_["pos_long"] );
    EXPECT_FALSE( cdict_["pos_long_is"]==cdict_["pos_long"] );
    EXPECT_TRUE(  +pos_long_is!=cdict_["ulong_ls"] );
    EXPECT_FALSE( +pos_long_is==cdict_["ulong_ls"] );
    EXPECT_TRUE(  cdict_["pos_long_is"]!=+ulong_ls );
    EXPECT_FALSE( cdict_["pos_long_is"]==+ulong_ls );
    EXPECT_TRUE(  cdict_["pos_long_is"]!=cdict_["ulong_ls"] );
    EXPECT_FALSE( cdict_["pos_long_is"]==cdict_["ulong_ls"] );
    EXPECT_TRUE(  +ulong_is!=cdict_["pos_long"] );
    EXPECT_FALSE( +ulong_is==cdict_["pos_long"] );
    EXPECT_TRUE(  cdict_["ulong_is"]!=+pos_long );
    EXPECT_FALSE( cdict_["ulong_is"]==+pos_long );
    EXPECT_TRUE(  cdict_["ulong_is"]!=cdict_["pos_long"] );
    EXPECT_FALSE( cdict_["ulong_is"]==cdict_["pos_long"] );
    EXPECT_TRUE(  +ulong_is!=cdict_["ulong_ls"] );
    EXPECT_FALSE( +ulong_is==cdict_["ulong_ls"] );
    EXPECT_TRUE(  cdict_["ulong_is"]!=+ulong_ls );
    EXPECT_FALSE( cdict_["ulong_is"]==+ulong_ls );
    EXPECT_TRUE(  cdict_["ulong_is"]!=cdict_["ulong_ls"] );
    EXPECT_FALSE( cdict_["ulong_is"]==cdict_["ulong_ls"] );
}

// Equalities between domains pos_int:pos_ulong
TEST_F(DictionaryTestEq, eqPosIntULong) {
    EXPECT_TRUE(  +pos_int!=cdict_["pos_ulong"] );
    EXPECT_FALSE( +pos_int==cdict_["pos_ulong"] );
    EXPECT_TRUE(  cdict_["pos_int"]!=+pos_ulong );
    EXPECT_FALSE( cdict_["pos_int"]==+pos_ulong );
    EXPECT_TRUE(  cdict_["pos_int"]!=cdict_["pos_ulong"] );
    EXPECT_FALSE( cdict_["pos_int"]==cdict_["pos_ulong"] );
    EXPECT_TRUE(  +uint_is!=cdict_["pos_ulong"] );
    EXPECT_FALSE( +uint_is==cdict_["pos_ulong"] );
    EXPECT_TRUE(  cdict_["uint_is"]!=+pos_ulong );
    EXPECT_FALSE( cdict_["uint_is"]==+pos_ulong );
    EXPECT_TRUE(  cdict_["uint_is"]!=cdict_["pos_ulong"] );
    EXPECT_FALSE( cdict_["uint_is"]==cdict_["pos_ulong"] );
    EXPECT_TRUE(  +pos_long_is!=cdict_["pos_ulong"] );
    EXPECT_FALSE( +pos_long_is==cdict_["pos_ulong"] );
    EXPECT_TRUE(  cdict_["pos_long_is"]!=+pos_ulong );
    EXPECT_FALSE( cdict_["pos_long_is"]==+pos_ulong );
    EXPECT_TRUE(  cdict_["pos_long_is"]!=cdict_["pos_ulong"] );
    EXPECT_FALSE( cdict_["pos_long_is"]==cdict_["pos_ulong"] );
    EXPECT_TRUE(  +ulong_is!=cdict_["pos_ulong"] );
    EXPECT_FALSE( +ulong_is==cdict_["pos_ulong"] );
    EXPECT_TRUE(  cdict_["ulong_is"]!=+pos_ulong );
    EXPECT_FALSE( cdict_["ulong_is"]==+pos_ulong );
    EXPECT_TRUE(  cdict_["ulong_is"]!=cdict_["pos_ulong"] );
    EXPECT_FALSE( cdict_["ulong_is"]==cdict_["pos_ulong"] );
}

// Equalities within domain pos_uint
TEST_F(DictionaryTestEq, eqPosUint) {
    // Same values:
    EXPECT_TRUE(  +pos_uint==cdict_["pos_uint"] );
    EXPECT_FALSE( +pos_uint!=cdict_["pos_uint"] );
    EXPECT_TRUE(  cdict_["pos_uint"]==+pos_uint );
    EXPECT_FALSE( cdict_["pos_uint"]!=+pos_uint );
    EXPECT_TRUE(  cdict_["pos_uint"]==cdict_["pos_uint"] );
    EXPECT_FALSE( cdict_["pos_uint"]!=cdict_["pos_uint"] );
    // Different values:
    EXPECT_TRUE(  +pos_uint!=cdict_["pos_uint1"] );
    EXPECT_FALSE( +pos_uint==cdict_["pos_uint1"] );
    EXPECT_TRUE(  cdict_["pos_uint"]!=+pos_uint1 );
    EXPECT_FALSE( cdict_["pos_uint"]==+pos_uint1 );
    EXPECT_TRUE(  cdict_["pos_uint"]!=cdict_["pos_uint1"] );
    EXPECT_FALSE( cdict_["pos_uint"]==cdict_["pos_uint1"] );
}

// Equalities between domains pos_uint:pos_long
TEST_F(DictionaryTestEq, eqPosUintPosLong) {
    EXPECT_TRUE(  +pos_uint!=cdict_["pos_long"] );
    EXPECT_FALSE( +pos_uint==cdict_["pos_long"] );
    EXPECT_TRUE(  cdict_["pos_uint"]!=+pos_long );
    EXPECT_FALSE( cdict_["pos_uint"]==+pos_long );
    EXPECT_TRUE(  cdict_["pos_uint"]!=cdict_["pos_long"] );
    EXPECT_FALSE( cdict_["pos_uint"]==cdict_["pos_long"] );
    EXPECT_TRUE(  +pos_uint!=cdict_["ulong_ls"] );
    EXPECT_FALSE( +pos_uint==cdict_["ulong_ls"] );
    EXPECT_TRUE(  cdict_["pos_uint"]!=+ulong_ls );
    EXPECT_FALSE( cdict_["pos_uint"]==+ulong_ls );
    EXPECT_TRUE(  cdict_["pos_uint"]!=cdict_["ulong_ls"] );
    EXPECT_FALSE( cdict_["pos_uint"]==cdict_["ulong_ls"] );
}

// Equalities between domains pos_uint:pos_ulong
TEST_F(DictionaryTestEq, eqPosUintULong) {
    EXPECT_TRUE(  +pos_uint!=cdict_["pos_ulong"] );
    EXPECT_FALSE( +pos_uint==cdict_["pos_ulong"] );
    EXPECT_TRUE(  cdict_["pos_uint"]!=+pos_ulong );
    EXPECT_FALSE( cdict_["pos_uint"]==+pos_ulong );
    EXPECT_TRUE(  cdict_["pos_uint"]!=cdict_["pos_ulong"] );
    EXPECT_FALSE( cdict_["pos_uint"]==cdict_["pos_ulong"] );
}

// Equalities within domain pos_long
TEST_F(DictionaryTestEq, eqPosLong) {
    // Same values:
    EXPECT_TRUE(  +pos_long==cdict_["pos_long"] );
    EXPECT_FALSE( +pos_long!=cdict_["pos_long"] );
    EXPECT_TRUE(  cdict_["pos_long"]==+pos_long );
    EXPECT_FALSE( cdict_["pos_long"]!=+pos_long );
    EXPECT_TRUE(  cdict_["pos_long"]==cdict_["pos_long"] );
    EXPECT_FALSE( cdict_["pos_long"]!=cdict_["pos_long"] );
    // Different values:
    EXPECT_TRUE(  +pos_long!=cdict_["pos_long1"] );
    EXPECT_FALSE( +pos_long==cdict_["pos_long1"] );
    EXPECT_TRUE(  cdict_["pos_long"]!=+pos_long1 );
    EXPECT_FALSE( cdict_["pos_long"]==+pos_long1 );
    EXPECT_TRUE(  cdict_["pos_long"]!=cdict_["pos_long1"] );
    EXPECT_FALSE( cdict_["pos_long"]==cdict_["pos_long1"] );
    // Same values:
    EXPECT_TRUE(  +pos_long==cdict_["ulong_ls"] );
    EXPECT_FALSE( +pos_long!=cdict_["ulong_ls"] );
    EXPECT_TRUE(  cdict_["pos_long"]==+ulong_ls );
    EXPECT_FALSE( cdict_["pos_long"]!=+ulong_ls );
    EXPECT_TRUE(  cdict_["pos_long"]==cdict_["ulong_ls"] );
    EXPECT_FALSE( cdict_["pos_long"]!=cdict_["ulong_ls"] );
    // Different values:
    EXPECT_TRUE(  +pos_long!=cdict_["ulong_ls1"] );
    EXPECT_FALSE( +pos_long==cdict_["ulong_ls1"] );
    EXPECT_TRUE(  cdict_["pos_long"]!=+ulong_ls1 );
    EXPECT_FALSE( cdict_["pos_long"]==+ulong_ls1 );
    EXPECT_TRUE(  cdict_["pos_long"]!=cdict_["ulong_ls1"] );
    EXPECT_FALSE( cdict_["pos_long"]==cdict_["ulong_ls1"] );
    // Same values:
    EXPECT_TRUE(  +ulong_ls==cdict_["ulong_ls"] );
    EXPECT_FALSE( +ulong_ls!=cdict_["ulong_ls"] );
    EXPECT_TRUE(  cdict_["ulong_ls"]==+ulong_ls );
    EXPECT_FALSE( cdict_["ulong_ls"]!=+ulong_ls );
    EXPECT_TRUE(  cdict_["ulong_ls"]==cdict_["ulong_ls"] );
    EXPECT_FALSE( cdict_["ulong_ls"]!=cdict_["ulong_ls"] );
    // Different values:
    EXPECT_TRUE(  +ulong_ls!=cdict_["ulong_ls1"] );
    EXPECT_FALSE( +ulong_ls==cdict_["ulong_ls1"] );
    EXPECT_TRUE(  cdict_["ulong_ls"]!=+ulong_ls1 );
    EXPECT_FALSE( cdict_["ulong_ls"]==+ulong_ls1 );
    EXPECT_TRUE(  cdict_["ulong_ls"]!=cdict_["ulong_ls1"] );
    EXPECT_FALSE( cdict_["ulong_ls"]==cdict_["ulong_ls1"] );
}

// Equalities between domains pos_long:pos_ulong
TEST_F(DictionaryTestEq, eqPosLongULong) {
    EXPECT_TRUE(  +pos_long!=cdict_["pos_ulong"] );
    EXPECT_FALSE( +pos_long==cdict_["pos_ulong"] );
    EXPECT_TRUE(  cdict_["pos_long"]!=+pos_ulong );
    EXPECT_FALSE( cdict_["pos_long"]==+pos_ulong );
    EXPECT_TRUE(  cdict_["pos_long"]!=cdict_["pos_ulong"] );
    EXPECT_FALSE( cdict_["pos_long"]==cdict_["pos_ulong"] );
    EXPECT_TRUE(  +ulong_ls!=cdict_["pos_ulong"] );
    EXPECT_FALSE( +ulong_ls==cdict_["pos_ulong"] );
    EXPECT_TRUE(  cdict_["ulong_ls"]!=+pos_ulong );
    EXPECT_FALSE( cdict_["ulong_ls"]==+pos_ulong );
    EXPECT_TRUE(  cdict_["ulong_ls"]!=cdict_["pos_ulong"] );
    EXPECT_FALSE( cdict_["ulong_ls"]==cdict_["pos_ulong"] );
}

// Equalities within domain pos_ulong
TEST_F(DictionaryTestEq, eqULong) {
    // Same values:
    EXPECT_TRUE(  +pos_ulong==cdict_["pos_ulong"] );
    EXPECT_FALSE( +pos_ulong!=cdict_["pos_ulong"] );
    EXPECT_TRUE(  cdict_["pos_ulong"]==+pos_ulong );
    EXPECT_FALSE( cdict_["pos_ulong"]!=+pos_ulong );
    EXPECT_TRUE(  cdict_["pos_ulong"]==cdict_["pos_ulong"] );
    EXPECT_FALSE( cdict_["pos_ulong"]!=cdict_["pos_ulong"] );
    // Different values:
    EXPECT_TRUE(  +pos_ulong!=cdict_["pos_ulong1"] );
    EXPECT_FALSE( +pos_ulong==cdict_["pos_ulong1"] );
    EXPECT_TRUE(  cdict_["pos_ulong"]!=+pos_ulong1 );
    EXPECT_FALSE( cdict_["pos_ulong"]==+pos_ulong1 );
    EXPECT_TRUE(  cdict_["pos_ulong"]!=cdict_["pos_ulong1"] );
    EXPECT_FALSE( cdict_["pos_ulong"]==cdict_["pos_ulong1"] );
}

// Equalities within same type my_bool
TEST_F(DictionaryTestEq, eqBool) {
    // Same values:
    EXPECT_TRUE(  my_bool==cdict_["my_bool"] );
    EXPECT_FALSE( my_bool!=cdict_["my_bool"] );
    EXPECT_TRUE(  cdict_["my_bool"]==my_bool );
    EXPECT_FALSE( cdict_["my_bool"]!=my_bool );
    EXPECT_TRUE(  cdict_["my_bool"]==cdict_["my_bool"] );
    EXPECT_FALSE( cdict_["my_bool"]!=cdict_["my_bool"] );
    // Different values:
    EXPECT_TRUE(  my_bool!=cdict_["my_bool1"] );
    EXPECT_FALSE( my_bool==cdict_["my_bool1"] );
    EXPECT_TRUE(  cdict_["my_bool"]!=my_bool1 );
    EXPECT_FALSE( cdict_["my_bool"]==my_bool1 );
    EXPECT_TRUE(  cdict_["my_bool"]!=cdict_["my_bool1"] );
    EXPECT_FALSE( cdict_["my_bool"]==cdict_["my_bool1"] );
}

// Equalities between different types my_bool:my_int
TEST_F(DictionaryTestEq, eqBoolInt) {
    EXPECT_TRUE(  my_bool!=cdict_["my_int"] );
    EXPECT_FALSE( my_bool==cdict_["my_int"] );
    EXPECT_TRUE(  cdict_["my_bool"]!=my_int );
    EXPECT_FALSE( cdict_["my_bool"]==my_int );
    EXPECT_TRUE(  cdict_["my_bool"]!=cdict_["my_int"] );
    EXPECT_FALSE( cdict_["my_bool"]==cdict_["my_int"] );
}

// Equalities between different types my_bool:my_string
TEST_F(DictionaryTestEq, eqBoolString) {
    EXPECT_TRUE(  my_bool!=cdict_["my_string"] );
    EXPECT_FALSE( my_bool==cdict_["my_string"] );
    EXPECT_TRUE(  cdict_["my_bool"]!=my_string );
    EXPECT_FALSE( cdict_["my_bool"]==my_string );
    EXPECT_TRUE(  cdict_["my_bool"]!=cdict_["my_string"] );
    EXPECT_FALSE( cdict_["my_bool"]==cdict_["my_string"] );
}

// Equalities between different types my_bool:my_vec
TEST_F(DictionaryTestEq, eqBoolVec) {
    EXPECT_TRUE(  my_bool!=cdict_["my_vec"] );
    EXPECT_FALSE( my_bool==cdict_["my_vec"] );
    EXPECT_TRUE(  cdict_["my_bool"]!=my_vec );
    EXPECT_FALSE( cdict_["my_bool"]==my_vec );
    EXPECT_TRUE(  cdict_["my_bool"]!=cdict_["my_vec"] );
    EXPECT_FALSE( cdict_["my_bool"]==cdict_["my_vec"] );
}

// Equalities between different types my_bool:my_pair
TEST_F(DictionaryTestEq, eqBoolPair) {
    EXPECT_TRUE(  my_bool!=cdict_["my_pair"] );
    EXPECT_FALSE( my_bool==cdict_["my_pair"] );
    EXPECT_TRUE(  cdict_["my_bool"]!=my_pair );
    EXPECT_FALSE( cdict_["my_bool"]==my_pair );
    EXPECT_TRUE(  cdict_["my_bool"]!=cdict_["my_pair"] );
    EXPECT_FALSE( cdict_["my_bool"]==cdict_["my_pair"] );
}

// Equalities within same type my_int
TEST_F(DictionaryTestEq, eqInt) {
    // Same values:
    EXPECT_TRUE(  my_int==cdict_["my_int"] );
    EXPECT_FALSE( my_int!=cdict_["my_int"] );
    EXPECT_TRUE(  cdict_["my_int"]==my_int );
    EXPECT_FALSE( cdict_["my_int"]!=my_int );
    EXPECT_TRUE(  cdict_["my_int"]==cdict_["my_int"] );
    EXPECT_FALSE( cdict_["my_int"]!=cdict_["my_int"] );
    // Different values:
    EXPECT_TRUE(  my_int!=cdict_["my_int1"] );
    EXPECT_FALSE( my_int==cdict_["my_int1"] );
    EXPECT_TRUE(  cdict_["my_int"]!=my_int1 );
    EXPECT_FALSE( cdict_["my_int"]==my_int1 );
    EXPECT_TRUE(  cdict_["my_int"]!=cdict_["my_int1"] );
    EXPECT_FALSE( cdict_["my_int"]==cdict_["my_int1"] );
}

// Equalities between different types my_int:my_string
TEST_F(DictionaryTestEq, eqIntString) {
    EXPECT_TRUE(  my_int!=cdict_["my_string"] );
    EXPECT_FALSE( my_int==cdict_["my_string"] );
    EXPECT_TRUE(  cdict_["my_int"]!=my_string );
    EXPECT_FALSE( cdict_["my_int"]==my_string );
    EXPECT_TRUE(  cdict_["my_int"]!=cdict_["my_string"] );
    EXPECT_FALSE( cdict_["my_int"]==cdict_["my_string"] );
}

// Equalities between different types my_int:my_vec
TEST_F(DictionaryTestEq, eqIntVec) {
    EXPECT_TRUE(  my_int!=cdict_["my_vec"] );
    EXPECT_FALSE( my_int==cdict_["my_vec"] );
    EXPECT_TRUE(  cdict_["my_int"]!=my_vec );
    EXPECT_FALSE( cdict_["my_int"]==my_vec );
    EXPECT_TRUE(  cdict_["my_int"]!=cdict_["my_vec"] );
    EXPECT_FALSE( cdict_["my_int"]==cdict_["my_vec"] );
}

// Equalities between different types my_int:my_pair
TEST_F(DictionaryTestEq, eqIntPair) {
    EXPECT_TRUE(  my_int!=cdict_["my_pair"] );
    EXPECT_FALSE( my_int==cdict_["my_pair"] );
    EXPECT_TRUE(  cdict_["my_int"]!=my_pair );
    EXPECT_FALSE( cdict_["my_int"]==my_pair );
    EXPECT_TRUE(  cdict_["my_int"]!=cdict_["my_pair"] );
    EXPECT_FALSE( cdict_["my_int"]==cdict_["my_pair"] );
}

// Equalities within same type my_string
TEST_F(DictionaryTestEq, eqString) {
    // Same values:
    EXPECT_TRUE(  my_string==cdict_["my_string"] );
    EXPECT_FALSE( my_string!=cdict_["my_string"] );
    EXPECT_TRUE(  cdict_["my_string"]==my_string );
    EXPECT_FALSE( cdict_["my_string"]!=my_string );
    EXPECT_TRUE(  cdict_["my_string"]==cdict_["my_string"] );
    EXPECT_FALSE( cdict_["my_string"]!=cdict_["my_string"] );
    // Different values:
    EXPECT_TRUE(  my_string!=cdict_["my_string1"] );
    EXPECT_FALSE( my_string==cdict_["my_string1"] );
    EXPECT_TRUE(  cdict_["my_string"]!=my_string1 );
    EXPECT_FALSE( cdict_["my_string"]==my_string1 );
    EXPECT_TRUE(  cdict_["my_string"]!=cdict_["my_string1"] );
    EXPECT_FALSE( cdict_["my_string"]==cdict_["my_string1"] );
}

// Equalities between different types my_string:my_vec
TEST_F(DictionaryTestEq, eqStringVec) {
    EXPECT_TRUE(  my_string!=cdict_["my_vec"] );
    EXPECT_FALSE( my_string==cdict_["my_vec"] );
    EXPECT_TRUE(  cdict_["my_string"]!=my_vec );
    EXPECT_FALSE( cdict_["my_string"]==my_vec );
    EXPECT_TRUE(  cdict_["my_string"]!=cdict_["my_vec"] );
    EXPECT_FALSE( cdict_["my_string"]==cdict_["my_vec"] );
}

// Equalities between different types my_string:my_pair
TEST_F(DictionaryTestEq, eqStringPair) {
    EXPECT_TRUE(  my_string!=cdict_["my_pair"] );
    EXPECT_FALSE( my_string==cdict_["my_pair"] );
    EXPECT_TRUE(  cdict_["my_string"]!=my_pair );
    EXPECT_FALSE( cdict_["my_string"]==my_pair );
    EXPECT_TRUE(  cdict_["my_string"]!=cdict_["my_pair"] );
    EXPECT_FALSE( cdict_["my_string"]==cdict_["my_pair"] );
}

// Equalities within same type my_vec
TEST_F(DictionaryTestEq, eqVec) {
    // Same values:
    EXPECT_TRUE(  my_vec==cdict_["my_vec"] );
    EXPECT_FALSE( my_vec!=cdict_["my_vec"] );
    EXPECT_TRUE(  cdict_["my_vec"]==my_vec );
    EXPECT_FALSE( cdict_["my_vec"]!=my_vec );
    EXPECT_TRUE(  cdict_["my_vec"]==cdict_["my_vec"] );
    EXPECT_FALSE( cdict_["my_vec"]!=cdict_["my_vec"] );
    // Different values:
    EXPECT_TRUE(  my_vec!=cdict_["my_vec1"] );
    EXPECT_FALSE( my_vec==cdict_["my_vec1"] );
    EXPECT_TRUE(  cdict_["my_vec"]!=my_vec1 );
    EXPECT_FALSE( cdict_["my_vec"]==my_vec1 );
    EXPECT_TRUE(  cdict_["my_vec"]!=cdict_["my_vec1"] );
    EXPECT_FALSE( cdict_["my_vec"]==cdict_["my_vec1"] );
}

// Equalities between different types my_vec:my_pair
TEST_F(DictionaryTestEq, eqVecPair) {
    EXPECT_TRUE(  my_vec!=cdict_["my_pair"] );
    EXPECT_FALSE( my_vec==cdict_["my_pair"] );
    EXPECT_TRUE(  cdict_["my_vec"]!=my_pair );
    EXPECT_FALSE( cdict_["my_vec"]==my_pair );
    EXPECT_TRUE(  cdict_["my_vec"]!=cdict_["my_pair"] );
    EXPECT_FALSE( cdict_["my_vec"]==cdict_["my_pair"] );
}

// Equalities within same type my_pair
TEST_F(DictionaryTestEq, eqPair) {
    // Same values:
    EXPECT_TRUE(  my_pair==cdict_["my_pair"] );
    EXPECT_FALSE( my_pair!=cdict_["my_pair"] );
    EXPECT_TRUE(  cdict_["my_pair"]==my_pair );
    EXPECT_FALSE( cdict_["my_pair"]!=my_pair );
    EXPECT_TRUE(  cdict_["my_pair"]==cdict_["my_pair"] );
    EXPECT_FALSE( cdict_["my_pair"]!=cdict_["my_pair"] );
    // Different values:
    EXPECT_TRUE(  my_pair!=cdict_["my_pair1"] );
    EXPECT_FALSE( my_pair==cdict_["my_pair1"] );
    EXPECT_TRUE(  cdict_["my_pair"]!=my_pair1 );
    EXPECT_FALSE( cdict_["my_pair"]==my_pair1 );
    EXPECT_TRUE(  cdict_["my_pair"]!=cdict_["my_pair1"] );
    EXPECT_FALSE( cdict_["my_pair"]==cdict_["my_pair1"] );
}

