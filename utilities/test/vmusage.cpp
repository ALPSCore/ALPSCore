/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/utilities/vmusage.hpp>
#include <boost/foreach.hpp>
#include <iostream>
#include <cstdlib>
#include "gtest/gtest.h"

pid_t global_child_pid;
TEST(vmusage, main)
{
    BOOST_FOREACH(alps::vmusage_type::value_type v, alps::vmusage(global_child_pid)) {
        std::cerr << v.first << " = " << v.second << "\n";
        };
}

int main(int argc, char **argv) 
{
    ::testing::InitGoogleTest(&argc, argv);

    pid_t pid = fork(); // create child process
    int status;
    switch(pid){
    case 0:
      system("sleep 1");
      exit(0);
    default:
      global_child_pid=pid;
      return RUN_ALL_TESTS();
    }
}
 
