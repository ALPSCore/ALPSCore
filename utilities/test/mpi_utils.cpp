/*
 * Copyright (C) 1998-2016 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/utilities/mpi.hpp>

#include <gtest/gtest.h>

/* Test MPI wrapper functionality. How?

   1) Create communicator by attaching. Verify that the returned C-communicator is the same.
   2) Create communicator by duplication. Verify that the returned C-communicator is different.
      Verify that C-communicator is working (e.g., can be used to determine the number of procs).
   3) Create communciator by taking ownership. Verify that the C-communicator is destroyed after
      destructor call.
*/

class MpiTest : public ::testing::Test {
  public:
    int nproc_;
    int myrank_;
    MPI_Comm newcomm_;
    MpiTest() {
        MPI_Comm_size(MPI_COMM_WORLD, &nproc_);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank_);
        MPI_Comm_dup(MPI_COMM_WORLD, &newcomm_);
        // sanity check:
        EXPECT_TRUE(is_valid(newcomm_));
    }

    bool is_valid(const MPI_Comm& comm) const {
        // FIXME? This is not guaranteed to work. :(
        int np=-1;
        int rank=-1;
        MPI_Comm_size(comm, &np);
        if (nproc_!=np) return false;
        MPI_Comm_rank(comm, &rank);
        return (myrank_==rank);
    }
};

namespace am=alps::mpi;

TEST_F(MpiTest, CommConstructDefault) {
    am::communicator comm;
    EXPECT_EQ(MPI_COMM_WORLD, comm);
    EXPECT_EQ(myrank_, comm.rank());
    EXPECT_EQ(nproc_, comm.size());
}

TEST_F(MpiTest, CommConstructAttach) {
    {
        am::communicator comm(newcomm_, am::comm_attach);
        EXPECT_EQ(newcomm_, comm);
        EXPECT_EQ(myrank_, comm.rank());
        EXPECT_EQ(nproc_, comm.size());
    }
    EXPECT_TRUE(is_valid(newcomm_));
}

int main(int argc, char** argv)
{
    alps::mpi::environment env(argc, argv, false);
    // alps::gtest_par_xml_output tweak;
    // tweak(alps::mpi::communicator().rank(), argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
