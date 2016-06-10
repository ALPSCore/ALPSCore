/*
 * Copyright (C) 1998-2016 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/params.hpp>
#include <gtest/gtest.h>

class ParamsTest : public ::testing::Test {
  public:
    alps::params par;

    ParamsTest()
    {
        const char* argv[]={ "program",
                             "--present_no_default=100",
                             "--present_has_default=200" };
        const int argc=sizeof(argv)/sizeof(*argv);

        par=alps::params(argc,argv);
        par
            .define<long>("present_no_default","Option w/o default")
            .define<long>("present_has_default",999,"Option with default")
            .define<long>("missing_no_default1", "Another option w/o default")
            .define<long>("missing_no_default2", "Another option w/o default")
            .define<long>("missing_has_default", 300, "Another option with default")
            .define<long>("missing_no_default3", "Another option w/o default")
            ;
        par["assigned"]=400L;
    }

    template <typename T>
    T get(const std::string& name) { return par[name]; }

    template <typename T>
    void set(const std::string& name, T val) { par[name]=val; }
};

TEST_F(ParamsTest, MissingIterator)
{
    alps::params::missing_params_iterator it=par.begin_missing();
    alps::params::missing_params_iterator end=par.end_missing();
    ASSERT_TRUE(it!=end);
    EXPECT_EQ(std::string("missing_no_default1"), *it);
    ++it;
    EXPECT_EQ(std::string("missing_no_default2"), *it);
    ++it;
    EXPECT_EQ(std::string("missing_no_default3"), *it);
    ++it;
    EXPECT_TRUE(it==end);
}
