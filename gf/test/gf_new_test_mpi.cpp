/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */


#include <gtest/gtest.h>

#include <alps/gf/gf_base.hpp>
#include <alps/gf/mesh.hpp>
#include <alps/utilities/gtest_par_xml_output.hpp>
#include "mpi_guard.hpp"

class GreensFunctionMPI : public ::testing::Test
{
public:
  const double beta;
  const int nfreq ;
  const int nsites ;
  int rank;
  bool is_root;
  static const int MASTER=0;
  typedef alps::gf::matsubara_mesh<alps::gf::mesh::POSITIVE_ONLY> matsubara_mesh;
  typedef alps::gf::index_mesh index_mesh;
  typedef alps::gf::greenf<double, matsubara_mesh, index_mesh> gf_type;
  gf_type gf;
  gf_type gf2;

  GreensFunctionMPI():beta(10), nfreq(10), nsites(5),
                      rank(alps::mpi::communicator().rank()),
                      is_root(alps::mpi::communicator().rank()==MASTER),
                      gf(matsubara_mesh(beta,nfreq), index_mesh(nsites)),
                      gf2(gf) {}
};


TEST_F(GreensFunctionMPI, BroadCast) {
  double new_val = 15.0;
  gf(matsubara_mesh::index_type(0), index_mesh::index_type(0)) = new_val;
  if(!is_root) {
    gf_type gf3;
    gf3.broadcast(alps::mpi::communicator(), MASTER);
    // check that GF is unchanged
    ASSERT_DOUBLE_EQ(gf(matsubara_mesh::index_type(0), index_mesh::index_type(0)), new_val);
    // check that new GF is the same as old
    ASSERT_EQ(gf3, gf);
  } else {
    gf.broadcast(alps::mpi::communicator(), MASTER);
  }
}

int main(int argc, char**argv)
{
  alps::mpi::environment env(argc, argv, false);
  alps::gtest_par_xml_output tweak;
  tweak(alps::mpi::communicator().rank(), argc, argv);


  ::testing::InitGoogleTest(&argc, argv);

  Mpi_guard guard(0, "gf_new_test_mpi.dat.");

  int rc=RUN_ALL_TESTS();

  if (!guard.check_sig_files_ok(get_number_of_bcasts())) {
    MPI_Abort(MPI_COMM_WORLD, 1); // otherwise it may get stuck in MPI_Finalize().
    // downside is the test aborts, rather than reports failure!
  }

  return rc;
}