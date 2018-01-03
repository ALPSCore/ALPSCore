/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file params_bcast_mpi.cpp
    
    @brief Tests MPI broadcast of parameters
*/

#include "./params_test_support.hpp"
#include <alps/utilities/gtest_par_xml_output.hpp>

using alps::params;

namespace test_data {
    static const char inifile_content[]=
        "my_bool=true\n"
        "my_int=1234\n"
        "my_string=simple\n"
        "my_double=12.75\n"
        ;

}

class ParamsTest : public ::testing::Test {
  protected:
    const alps::mpi::communicator comm_;
    int root_;
    bool is_master_;
  public:
    ParamsTest(): comm_(), root_(0),
                  is_master_(comm_.rank()==root_)
                   
    { }
};


TEST_F(ParamsTest, bcast) {
    using alps::mpi::broadcast;

    ini_maker ini("params_bcast_mpi.ini.");
    ini.add(test_data::inifile_content);
    
    params p_as_on_root(ini.name());

    std::string root_ini_name=ini.name();
    broadcast(comm_, root_ini_name, root_);

    params p_slave;

    params& p=*(is_master_ ? &p_as_on_root : &p_slave);
    
    p_as_on_root.define<int>("my_int", "Integer param");
    p_as_on_root.define<std::string>("my_string", "String param");

    // Sanity check
    if (is_master_) {
        ASSERT_TRUE(p_as_on_root==p) << "Observed on rank " << comm_.rank();
    } else {
        ASSERT_TRUE(p_slave==p) << "Observed on rank " << comm_.rank();
    }
    
    broadcast(comm_, p, root_);

    EXPECT_TRUE(p==p_as_on_root) << "Observed on rank " << comm_.rank();

    EXPECT_EQ(p_as_on_root.get_argv0(), p.get_argv0()) << "Observed on rank " << comm_.rank();
    EXPECT_EQ(1, p.get_ini_name_count()) << "Observed on rank " << comm_.rank();
    EXPECT_EQ(root_ini_name, p.get_ini_name(0)) << "Observed on rank " << comm_.rank();
}

TEST_F(ParamsTest, bcastCtor) {
    arg_holder args;
    if (is_master_) {
        args.add("my_int=1").add("my_string=abc");
    }
    params p(args.argc(), args.argv(), comm_, root_);

    ASSERT_TRUE(p.define<int>("my_int", "Integer").ok()) << "Observed on rank " << comm_.rank();
    ASSERT_TRUE(p.define<std::string>("my_string", "String").ok()) << "Observed on rank " << comm_.rank();
    EXPECT_EQ(1, p["my_int"].as<int>()) << "Observed on rank " << comm_.rank();
    EXPECT_EQ("abc", p["my_string"].as<std::string>()) << "Observed on rank " << comm_.rank();
}


int main(int argc, char** argv)
{
   alps::mpi::environment env(argc, argv, false);
   alps::gtest_par_xml_output tweak;
   tweak(alps::mpi::communicator().rank(), argc, argv);
   ::testing::InitGoogleTest(&argc, argv);
   
   return RUN_ALL_TESTS();
}    
