/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <iostream>
#include <alps/utility/temporary_filename.hpp>

#include <gtest/gtest.h>

TEST(temporary_fname, main)
{
  std::string prefix="alps_temp_filename_test";
  std::string filename=alps::temporary_filename(prefix);
}

int main(int argc, char **argv) 
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

