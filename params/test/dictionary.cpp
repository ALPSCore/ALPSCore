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

#include <boost/utility.hpp> // for enable_if<>

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


// Helper meta-predicate to distinguish strings (general)
template <typename T>
struct is_string {
    typedef bool no;
};

// Helper meta-predicate to distinguish strings (specialization)
template <>
struct is_string<std::string> {
    typedef bool yes;
};


// Parametrized on the value type stored in the dictionary
template <typename T>
class DictionaryTest : public ::testing::Test {
    protected:
    dictionary dict_;
    const dictionary& cdict_;
    
    typedef aptest::data_trait<T> trait;
    typedef typename trait::larger_type larger_type;
    typedef typename trait::smaller_type smaller_type;
    typedef typename trait::incompatible_type incompatible_type;
    static const bool has_larger_type=trait::has_larger_type;
    static const bool has_smaller_type=trait::has_smaller_type;

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
    void explicitAssignSameType_helper(typename is_string<X>::no =true) {
        const T expected=trait::get(false);
        const T actual=static_cast<T>(cdict_["name"]);
        EXPECT_EQ(expected, actual) << "Explicit conversion";
    }
    
    // assignment is done in ctor; check if it worked
    template <typename X>
    void explicitAssignSameType_helper(typename is_string<X>::yes =true) {
        // The following does not compile, due to ambiguous string ctor call.
        /*
          const T expected=trait::get(false);
          const T actual=static_cast<T>(cdict_["name"]);
          EXPECT_EQ(expected, actual) << "Explicit conversion";
        */
    }
    
    // assignment is done in ctor; check if it worked
    void explicitAssignSameType() { explicitAssignSameType_helper<T>(); }

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
    void assignFromNone_helper(typename is_string<X>::no =true) {
        const T expected=trait::get(false);
        T actual=expected;
        ASSERT_THROW(actual=cdict_["none"], de::uninitialized_value);
        EXPECT_EQ(expected,actual);
    }

    template <typename X>
    void assignFromNone_helper(typename is_string<X>::yes =true) {
        // The following does not compile due to ambiguous string assignment
        /*
          const T expected=trait::get(false);
          T actual=expected;
          ASSERT_THROW(actual=cdict_["none"], de::uninitialized_value);
          EXPECT_EQ(expected,actual);
        */
    }

    void assignFromNone() {
        assignFromNone_helper<T>();
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
    double
    ,
    std::string
    > my_types;

TYPED_TEST_CASE(DictionaryTest, my_types);

#define MAKE_TEST(_name_) TYPED_TEST(DictionaryTest,_name_) { this->_name_(); }

MAKE_TEST(afterCtor);
MAKE_TEST(assignRetval);
MAKE_TEST(explicitAssignSameType);
MAKE_TEST(implicitAssignSameType);
MAKE_TEST(asSameType);
MAKE_TEST(reassignSameType);
MAKE_TEST(assignFromNone);
MAKE_TEST(convertFromNoneExplicit);
MAKE_TEST(setToNone);

