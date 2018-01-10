/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/utilities/mpi.hpp>

#include <gtest/gtest.h>

#include "mpi_test_support.hpp"

/* Test MPI environment wrapper functionality. */

// we can initialize MPI exactly once, so everything is clubbed into this test
TEST(MpiEnvTest, Environment) {
    int argc=1;
    char arg0[]="";
    char* args[]={arg0};
    char** argv=args;
    ASSERT_FALSE(mpi_is_up());
    ASSERT_FALSE(mpi_is_down());
    ASSERT_FALSE(alps::mpi::environment::initialized()) << "MPI Env should not be initialized";
    ASSERT_FALSE(alps::mpi::environment::finalized()) << "MPI Env should not be finalized";
    {
        alps::mpi::environment env(argc, argv);
        ASSERT_TRUE(mpi_is_up());
        ASSERT_FALSE(mpi_is_down());

        ASSERT_TRUE(alps::mpi::environment::initialized()) << "MPI Env should be initialized by now";
        ASSERT_FALSE(alps::mpi::environment::finalized()) << "MPI Env should not be finalized yet";

        {
            alps::mpi::environment sub_env(argc, argv);
            ASSERT_TRUE(mpi_is_up());
            ASSERT_FALSE(mpi_is_down());
        } // sub_env destroyed
        ASSERT_TRUE(mpi_is_up()) << "MPI should remain initialized after sub-object destruction";
        ASSERT_FALSE(mpi_is_down()) << "MPI should not be finalized after sub-object destruction";
    } // env destroyed

    ASSERT_TRUE(mpi_is_up());
    ASSERT_TRUE(mpi_is_down()) << "MPI should be finalized after environment destruction";

    ASSERT_TRUE(alps::mpi::environment::initialized()) << "MPI Env should be initialized after environment destruction";
    ASSERT_TRUE(alps::mpi::environment::finalized()) << "MPI Env should be finalized after environment destruction";
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
