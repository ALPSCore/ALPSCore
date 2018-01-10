/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/utilities/mpi.hpp>

#include <gtest/gtest.h>

/* Test MPI communicator wrapper functionality. How?

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

TEST_F(MpiTest, CommConstructDuplicate) {
    MPI_Comm tmpcomm;
    {
        am::communicator comm(newcomm_, am::comm_duplicate);
        EXPECT_NE(newcomm_, comm);
        EXPECT_EQ(myrank_, comm.rank());
        EXPECT_EQ(nproc_, comm.size());
        tmpcomm=comm;
    }
    EXPECT_TRUE(is_valid(newcomm_));
    EXPECT_TRUE(tmpcomm!=MPI_COMM_NULL); // to pacify "unused variable" warning
    // This is undefined and will crash:
    // EXPECT_FALSE(is_valid(tmpcomm));
}

TEST_F(MpiTest, CommConstructOwn) {
    {
        am::communicator comm(newcomm_, am::take_ownership);
        EXPECT_EQ(newcomm_, comm);
        EXPECT_EQ(myrank_, comm.rank());
        EXPECT_EQ(nproc_, comm.size());
    }
    // This is undefined and will crash:
    // EXPECT_FALSE(is_valid(newcomm_));
}

TEST_F(MpiTest, CommConctructCopy) {
    am::communicator comm_d1(newcomm_, am::comm_duplicate);
    am::communicator comm_a1(newcomm_, am::comm_attach);
    {
        am::communicator comm_d2(comm_d1);
        EXPECT_EQ(static_cast<MPI_Comm>(comm_d1), static_cast<MPI_Comm>(comm_d2));
        am::communicator comm_a2(comm_a1);
        EXPECT_EQ(static_cast<MPI_Comm>(comm_a1), static_cast<MPI_Comm>(comm_a2));
    }
    // after comm_{a,d}2 are destroyed, the corresponding comm1's are valid
    EXPECT_TRUE(is_valid(comm_d1));
    EXPECT_TRUE(is_valid(comm_a1));
}

TEST_F(MpiTest, CommAssignAttached) {
    am::communicator comm_d1(newcomm_, am::comm_duplicate);
    am::communicator comm_a1(newcomm_, am::comm_attach);
    MPI_Comm mpicomm_d1=comm_d1;
    {
        am::communicator comm_a2(newcomm_, am::comm_attach);
        comm_d1=comm_a2;
        EXPECT_EQ(static_cast<MPI_Comm>(comm_a2), static_cast<MPI_Comm>(comm_d1));
        comm_a1=comm_a2;
        EXPECT_EQ(static_cast<MPI_Comm>(comm_a2), static_cast<MPI_Comm>(comm_a1));
    }
    // after comm_a2 is destroyed, the comm1's are valid
    EXPECT_EQ(newcomm_,comm_d1);
    EXPECT_EQ(newcomm_,comm_a1);
    EXPECT_TRUE(is_valid(comm_d1));
    EXPECT_TRUE(is_valid(comm_a1));

    EXPECT_TRUE(mpicomm_d1!=MPI_COMM_NULL); // to pacify "unused variable" warning
    // and mpicomm_d1 (underlying MPI communicator for comm_d1) is not valid and may crash
    // EXPECT_FALSE(is_valid(mpicomm_d1));
}

TEST_F(MpiTest, CommAssignDuplicated) {
    am::communicator comm_d1(newcomm_, am::comm_duplicate);
    am::communicator comm_a1(newcomm_, am::comm_attach);
    MPI_Comm mpicomm_d1=comm_d1;
    MPI_Comm mpicomm_a2;
    {
        am::communicator comm_a2(newcomm_, am::comm_duplicate);
        mpicomm_a2=comm_a2;
        comm_d1=comm_a2;
        EXPECT_EQ(static_cast<MPI_Comm>(comm_a2), static_cast<MPI_Comm>(comm_d1));
        comm_a1=comm_a2;
        EXPECT_EQ(static_cast<MPI_Comm>(comm_a2), static_cast<MPI_Comm>(comm_a1));
    }
    // after comm_a2 is destroyed, the comm1's (and the underlying MPI comm) are valid
    EXPECT_EQ(mpicomm_a2,comm_d1);
    EXPECT_EQ(mpicomm_a2,comm_a1);
    EXPECT_TRUE(is_valid(comm_d1));
    EXPECT_TRUE(is_valid(comm_a1));

    EXPECT_TRUE(mpicomm_d1!=MPI_COMM_NULL); // to pacify "unused variable" warning
    // and mpicomm_d1 (underlying MPI communicator for comm_d1) is not valid and may crash
    // EXPECT_FALSE(is_valid(mpicomm_d1));
}

int main(int argc, char** argv)
{
    alps::mpi::environment env(argc, argv); // initializes MPI environment
    // alps::gtest_par_xml_output tweak;
    // tweak(alps::mpi::communicator().rank(), argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
