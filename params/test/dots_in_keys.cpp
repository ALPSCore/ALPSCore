/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file dots_in_keys.cpp

    @brief Tests the behaviour of parameters with dots in keys and sections
*/

#include "./params_test_support.hpp"
#include <alps/params/iniparser_interface.hpp>

class ParamsWithDots : public ::testing::Test {
  public:
    ini_maker ini;
    ParamsWithDots() : ini("dots_in_keys.ini.", alps::testing::unique_file::REMOVE_AFTER) { }
};

TEST_F(ParamsWithDots, dotInSection) {
    ini
        .add("[head]")
        .add("anykey=anyvalue")
        .add("[section.with_dot]")
        .add("key=key_from_section_with_dot");
    alps::params par(ini.name());

    par.define<std::string>("section.with_dot.key","param");
    ASSERT_TRUE(par.ok()) << "Parameter dump:\n" << par << "\n===\n";
    const std::string expected="key_from_section_with_dot";
    std::string actual;
    ASSERT_NO_THROW(actual=par["section.with_dot.key"].as<std::string>());
    EXPECT_EQ(expected, actual);
}

TEST_F(ParamsWithDots, dotInSectionWithDotlessSection) {
    ini
        .add("headless=headless_value")
        .add("[head]")
        .add("anykey=anyvalue")
        .add("[section.with_dot]")
        .add("key=key_from_section_with_dot")
        .add("[section]")
        .add("key2=key2_from_section");

    alps::params par(ini.name());

    par.define<std::string>("section.with_dot.key","param");
    ASSERT_TRUE(par.ok()) << "Parameter dump:\n" << par << "\n===\n";
    par.define<std::string>("section.key2","param");
    ASSERT_TRUE(par.ok()) << "Parameter dump:\n" << par << "\n===\n";
    {
        const std::string expected="key_from_section_with_dot";
        std::string actual;
        ASSERT_NO_THROW(actual=par["section.with_dot.key"].as<std::string>());
        EXPECT_EQ(expected, actual);
    }
    {
        const std::string expected="key2_from_section";
        std::string actual;
        ASSERT_NO_THROW(actual=par["section.key2"].as<std::string>());
        EXPECT_EQ(expected, actual);
    }
}

TEST_F(ParamsWithDots, dotInKey) {
    ini
        .add("[head]")
        .add("anykey=anyvalue")
        .add("[section]")
        .add("key.with_dot=key_with_dot_from_section");
    alps::params par(ini.name());

    par.define<std::string>("section.key.with_dot","param");
    ASSERT_TRUE(par.ok()) << "Parameter dump:\n" << par << "\n===\n";
    const std::string expected="key_with_dot_from_section";
    std::string actual;
    ASSERT_NO_THROW(actual=par["section.key.with_dot"].as<std::string>());
    EXPECT_EQ(expected, actual);
}

TEST_F(ParamsWithDots, dotInSectionAndKey) {
    ini
        .add("[head]")
        .add("anykey=anyvalue")
        .add("[section2.with_dot]")
        .add("key.with_dot=key_with_dot_from_section2_with_dot");
    alps::params par(ini.name());

    par.define<std::string>("section2.with_dot.key.with_dot","param");
    ASSERT_TRUE(par.ok()) << "Parameter dump:\n" << par << "\n===\n";
    const std::string expected="key_with_dot_from_section2_with_dot";
    std::string actual;
    ASSERT_NO_THROW(actual=par["section2.with_dot.key.with_dot"].as<std::string>());
    EXPECT_EQ(expected, actual);
}
