/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file dictionary.cpp
    
    @brief Tests the behaviour of dictionary
*/

#include <alps/params_new.hpp>
#include <gtest/gtest.h>

// #include <boost/utility.hpp> // for enable_if<>
#include <boost/type_traits/is_same.hpp>
using boost::is_same;

#include "./dict_values_test.hpp"

namespace ap=alps::params_new_ns;
using ap::dictionary;
namespace de=ap::exception;
namespace aptest=ap::testing;

// // Parametrized on the value type stored in the dictionary
// template <typename T>
class DictionaryTest0 : public ::testing::Test {
    protected:
    dictionary dict_;
    const dictionary& cdict_;

    public:
    DictionaryTest0(): dict_(), cdict_(dict_) {}
};

TEST_F(DictionaryTest0, ctor) {
    EXPECT_TRUE(cdict_.empty());
    EXPECT_EQ(0ul, cdict_.size());
}

TEST_F(DictionaryTest0, access) {
    EXPECT_TRUE(dict_["name"].empty());
    EXPECT_FALSE(cdict_.empty());
    EXPECT_EQ(1ul, cdict_.size());
    EXPECT_TRUE(dict_["name"].empty());
    
    EXPECT_TRUE(dict_["name2"].empty());
    EXPECT_FALSE(cdict_.empty());
    EXPECT_EQ(2ul, cdict_.size());
    EXPECT_TRUE(dict_["name2"].empty());

    EXPECT_TRUE(dict_["name"].empty());
    EXPECT_FALSE(cdict_.empty());
    EXPECT_EQ(2ul, cdict_.size());

    EXPECT_TRUE(dict_["name1"].empty());
    EXPECT_FALSE(cdict_.empty());
    EXPECT_EQ(3ul, cdict_.size());
}

TEST_F(DictionaryTest0, constAccess) {
    EXPECT_THROW(cdict_["name"], de::uninitialized_value);
    EXPECT_TRUE(cdict_.empty());
    EXPECT_EQ(0ul, cdict_.size());

    dict_["name"];
    EXPECT_TRUE(cdict_["name"].empty());
}

TEST_F(DictionaryTest0, charAssign) {
    dict_["name"]='x';
    EXPECT_EQ(1ul, cdict_.size());
}

TEST_F(DictionaryTest0, charGetter) {
    char expected='x';
    dict_["name"]=expected;
    char actual=cdict_["name"];
    EXPECT_EQ(expected, actual);
}

TEST_F(DictionaryTest0, charAsIntGetter) {
    char expected='x';
    dict_["name"]=expected;
    int actual=cdict_["name"];
    EXPECT_EQ(expected, actual);
}

TEST_F(DictionaryTest0, boolToString) {
    bool expected=true;
    dict_["name"]=expected;
    std::string actual=cdict_["name"];
    // EXPECT_EQ(expected, actual);
}


/// Helper metafunctions / metapredicates
template <bool V> struct yes_no {};
template <> struct yes_no<true> { typedef bool yes; static const bool value=true; };
template <> struct yes_no<false> { typedef bool no; static const bool value=false; };

template <typename T> struct is_vector : public yes_no<false> {};

template <typename T>
struct is_vector< std::vector<T> > : public yes_no<true> {};

template <typename T>
struct is_string : public yes_no<is_same<T,std::string>::value> {};

template <typename T>
struct is_string_or_vec : public yes_no<is_string<T>::value || is_vector<T>::value> {};

template <typename T>
struct is_bool : public yes_no<is_same<T,bool>::value> {};


// Parametrized on the value type stored in the dictionary
template <typename T>
class DictionaryTest : public ::testing::Test {
    protected:
    dictionary dict_;
    const dictionary& cdict_;
    
    typedef aptest::data_trait<T> trait;
    // typedef typename trait::larger_type larger_type;
    // typedef typename trait::smaller_type smaller_type;
    // typedef typename trait::incompatible_type incompatible_type;
    // static const bool has_larger_type=trait::has_larger_type;
    // static const bool has_smaller_type=trait::has_smaller_type;

    public:
    DictionaryTest(): dict_(), cdict_(dict_) {
        dict_["none"];
        const T expected=trait::get(false);
        dict_["name"]=expected;
    }

    void afterCtor() {
        EXPECT_TRUE(cdict_["none"].empty());
        EXPECT_FALSE(cdict_["name"].empty());
    }


    void assignRetval() {
        const T expected=trait::get(true);
        const T actual=(dict_["name"]=expected);
        EXPECT_EQ(expected, actual);
    }

    // assignment is done in ctor; check if it worked
    template <typename X>
    void explicitAssignSameType(typename is_string_or_vec<X>::no =true) {
        const T expected=trait::get(false);
        const T actual=static_cast<T>(cdict_["name"]);
        EXPECT_EQ(expected, actual) << "Explicit conversion";
    }
    
    // assignment is done in ctor; check if it worked
    template <typename X>
    void explicitAssignSameType(typename is_string_or_vec<X>::yes =true) {
        // The following does not compile, due to ambiguous string/vector ctor call.
        /*
          const T expected=trait::get(false);
          const T actual=static_cast<T>(cdict_["name"]);
          EXPECT_EQ(expected, actual) << "Explicit conversion";
        */
    }
    

    void implicitAssignSameType() {
        const T expected=trait::get(false);
        const T actual=cdict_["name"];
        EXPECT_EQ(expected, actual) << "Implicit conversion";
    }
    

    void asSameType() {
        const T expected=trait::get(false);
        const T actual=cdict_["name"].template as<T>();
        EXPECT_EQ(expected, actual) << "Shortcut conversion";
    }
    

    void reassignSameType() {
        const T expected=trait::get(true);
        dict_["name"]=expected;
        const T actual=cdict_["name"];
        EXPECT_EQ(expected, actual);
    }

    template <typename X>
    void assignFromNone(typename is_string<X>::no =true) {
        const T expected=trait::get(false);
        T actual=expected;
        ASSERT_THROW(actual=cdict_["none"], de::uninitialized_value);
        EXPECT_EQ(expected,actual);
    }

    template <typename X>
    void assignFromNone(typename is_string<X>::yes =true) {
        // The following does not compile due to ambiguous string assignment
        /*
          const T expected=trait::get(false);
          T actual=expected;
          ASSERT_THROW(actual=cdict_["none"], de::uninitialized_value);
          EXPECT_EQ(expected,actual);
        */
    }

    void convertFromNoneExplicit() {
        const T expected=trait::get(false);
        T actual=expected;
        ASSERT_THROW(actual=cdict_["none"].template as<T>(), de::uninitialized_value);
        EXPECT_EQ(expected,actual);
    }
    
    void setToNone() {
        dict_["name"].clear();
        EXPECT_TRUE(cdict_["name"].empty());
    }

    // Nothing (except bool) can be converted to bool
    template <typename X>
    void toBool(typename is_bool<X>::no =true) {
        bool actual=true;
        EXPECT_THROW(actual=cdict_["name"], de::type_mismatch);
        EXPECT_TRUE(actual);
    }

    template <typename X>
    void toBool(typename is_bool<X>::yes =true) {}
    
};

typedef ::testing::Types<
    bool
    ,
    int
    ,
    long
    ,
    unsigned long int
    ,
    float
    ,
    double
    ,
    std::string
    ,
    std::vector<bool>
    ,
    std::vector<int>
    ,
    std::vector<long>
    ,
    std::vector<unsigned long int>
    ,
    std::vector<double>
    ,
    std::vector<std::string>
    
    > my_all_types;

TYPED_TEST_CASE(DictionaryTest, my_all_types);

#define MAKE_TEST(_name_) TYPED_TEST(DictionaryTest,_name_) { this->_name_(); }
#define MAKE_TEST_TMPL(_name_) TYPED_TEST(DictionaryTest,_name_) { this->template _name_<TypeParam>(); }

MAKE_TEST(afterCtor);
MAKE_TEST(assignRetval);
MAKE_TEST_TMPL(explicitAssignSameType);
MAKE_TEST(implicitAssignSameType);
MAKE_TEST(asSameType);
MAKE_TEST(reassignSameType);
MAKE_TEST_TMPL(assignFromNone);
MAKE_TEST(convertFromNoneExplicit);
MAKE_TEST(setToNone);
MAKE_TEST_TMPL(toBool);

#undef MAKE_TEST

// Parametrized on the value type stored in the dictionary
template <typename T>
class DictionaryTestBool : public ::testing::Test {
    protected:
    dictionary dict_;
    const dictionary& cdict_;
    
    typedef aptest::data_trait<T> trait;

    public:
    DictionaryTestBool(): dict_(), cdict_(dict_) {
        dict_["true"]=true;
        dict_["false"]=false;
    }

    // Bool can be converted to any integral type
    void toIntegral() {
        {
            T expected=true;
            T actual=cdict_["true"];
            EXPECT_EQ(expected, actual) << "true value test";
        }
        {
            T expected=false;
            T actual=cdict_["false"];
            EXPECT_EQ(expected, actual) << "false value test";
        }
    }
};

typedef ::testing::Types<
    char
    ,
    int
    ,
    long
    ,
    unsigned long int
    > my_integral_types;

TYPED_TEST_CASE(DictionaryTestBool, my_integral_types);

#define MAKE_TEST(_name_) TYPED_TEST(DictionaryTestBool,_name_) { this->_name_(); }

MAKE_TEST(toIntegral);

#undef MAKE_TEST
