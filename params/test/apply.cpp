/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file apply.cpp

    @brief Tests the behaviour of apply() and foreach() free functions
*/

#include <alps/params.hpp>

#include <gtest/gtest.h>

struct ParamTest : public ::testing::Test {
    alps::params params_;

    ParamTest() {
        params_.define<int>("integer", 42, "Very important integer");
        params_.define<double>("pi", "Something from geometry");
        params_.define<bool>("switch", true, "Has no meaning too");
        params_.define<std::string>("name", "qwerty", "Name of the game");

        params_["pi"] = 22.0 / 7;
        params_["name"] = "asdfgh";
    }
};

template <typename ExpectedType>
struct apply_test_functor {

    /// Expected option properties
    std::string name_;
    bool has_val_;
    ExpectedType val_;
    bool has_deflt_;
    ExpectedType deflt_;
    std::string descr_;

    apply_test_functor(const std::string& name,
                       bool has_val, const ExpectedType& val,
                       bool has_deflt, const ExpectedType& deflt,
                       const std::string& descr) :
        name_(name), has_val_(has_val), val_(val), has_deflt_(has_deflt), deflt_(deflt), descr_(descr) {}

    /// This overload is called if option has the ExpectedType
    void operator()(const std::string& name,
                    boost::optional<ExpectedType> const& val,
                    boost::optional<ExpectedType> const& defval,
                    const std::string& descr) const {
        EXPECT_EQ(name_, name);
        EXPECT_EQ(has_val_, static_cast<bool>(val));
        if(static_cast<bool>(val)) EXPECT_EQ(val_, boost::get(val));
        EXPECT_EQ(has_deflt_, static_cast<bool>(defval));
        if(static_cast<bool>(defval)) EXPECT_EQ(deflt_, boost::get(defval));
        EXPECT_EQ(descr_, descr);
    }

    /// Unexpected option type
    template <typename T>
    void operator()(const std::string& name,
                    boost::optional<T> const& val,
                    boost::optional<T> const& defval,
                    const std::string& descr) const {
        FAIL();
    }
};

TEST_F(ParamTest,apply) {
    apply_test_functor<int> f1("integer", true, 42, true, 42, "Very important integer");
    apply(params_, "integer", f1);

    apply_test_functor<double> f2("pi", true, 22.0 / 7, false, 0, "Something from geometry");
    apply(params_, "pi", f2);

    apply_test_functor<bool> f3("switch", true, true, true, true, "Has no meaning too");
    apply(params_, "switch", f3);

    apply_test_functor<std::string> f4("name", true, "asdfgh", true, "qwerty", "Name of the game");
    apply(params_, "name", f4);
}

struct foreach_test_functor {

#define CALL_OPERATOR(TYPE,NAME,HAS_VAL,VAL,HAS_DEFLT,DEFLT,DESCR)                        \
    void operator()(const std::string& name, boost::optional<TYPE> const& val,            \
                    boost::optional<TYPE> const& defval, const std::string& descr) const  \
    {                                                                                     \
        if(name == "help") return;                                                        \
        EXPECT_EQ(NAME, name);                                                            \
        EXPECT_TRUE(HAS_VAL == static_cast<bool>(val));                                   \
        if(static_cast<bool>(val)) EXPECT_EQ(VAL, boost::get(val));                       \
        EXPECT_TRUE(HAS_DEFLT == static_cast<bool>(defval));                              \
        if(static_cast<bool>(defval)) EXPECT_EQ(DEFLT, boost::get(defval));               \
        EXPECT_EQ(DESCR, descr);                                                          \
    }

    CALL_OPERATOR(int, "integer", true, 42, true, 42, "Very important integer")
    CALL_OPERATOR(double, "pi", true, (22.0/7), false, 0, "Something from geometry")
    CALL_OPERATOR(bool, "switch", true, true, true, true, "Has no meaning too")
    CALL_OPERATOR(std::string, "name", true, "asdfgh", true, "qwerty", "Name of the game")
#undef CALL_OPERATOR

    /// Unexpected option type
    template <typename T>
    void operator()(const std::string& name, boost::optional<T> const& val, \
                    boost::optional<T> const& defval, const std::string& descr) const {
        FAIL();
    }
};

TEST_F(ParamTest,foreach) {
    foreach(params_, foreach_test_functor());
}
