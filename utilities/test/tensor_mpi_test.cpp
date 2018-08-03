/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */


#include <gtest/gtest.h>
#include <mpi.h>
#include <alps/config.hpp>
#include <alps/numeric/tensors.hpp>

class TensorMpiTest : public ::testing::Test {
public:
  int nproc_;
  int myrank_;
  TensorMpiTest() {
    MPI_Comm_size(MPI_COMM_WORLD, &nproc_);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank_);
  }
};

TEST_F(TensorMpiTest, TestInit) {
  MPI_Comm shmcomm;
  MPI_Comm_split_type (MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,0, MPI_INFO_NULL,&shmcomm);
  int shnprocs;
  int shmyrank;
  MPI_Comm_size(MPI_COMM_WORLD, &shnprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &shmyrank);
  int N1 = 10, N2 = 10, N3 = 10;
  MPI_Win win;
  alps::numerics::shared_storage<double> container(N1*N2*N3, alps::numerics::detail::mpi_shared_allocator<double>(shmcomm, win));
  alps::numerics::shared_tensor<double, 3> T(container, N1, N2, N3);
  size_t alloc_length = T.size();
  T.storage().lock();
  for (int i = 0; i < alloc_length; ++i) {
    if((i%shnprocs) == shmyrank)
      T.data()[i] = (1<<shmyrank)*i;
  }
  T.storage().release();

  for (int i = 0; i < alloc_length; ++i) {
    const alps::numerics::shared_tensor<double, 3>& X = T;
    ASSERT_DOUBLE_EQ(X.data()[i],(1<<(i%shnprocs))*i);
  }
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv); // initializes MPI environment
  // alps::gtest_par_xml_output tweak;
  // tweak(alps::mpi::communicator().rank(), argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  int res = RUN_ALL_TESTS();
  MPI_Finalize();
  return res;
}
