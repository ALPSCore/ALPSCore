/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */


#include <gtest/gtest.h>

#include <alps/gf/gf_base.hpp>
#include <alps/gf/mesh.hpp>
#include <alps/utilities/gtest_par_xml_output.hpp>
#include <alps/gf/tail.hpp>
#include "mpi_guard.hpp"

namespace gfns = alps::gf;

class TailedGreensFunctionMPI : public ::testing::Test
{
public:
  const double beta;
  const int nsites;
  const int nfreq ;
  const int nspins;
  int rank;
  bool is_root;
  static const int MASTER=0;
  typedef gfns::greenf<std::complex<double> , gfns::matsubara_positive_mesh, gfns::index_mesh> omega_sigma_gf;
  typedef gfns::greenf<std::complex<double>, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh> head;
  typedef gfns::greenf<double, alps::gf::index_mesh> tail;
  typedef alps::gf::matsubara_mesh<alps::gf::mesh::POSITIVE_ONLY> matsubara_mesh;
  typedef alps::gf::index_mesh index_mesh;
  omega_sigma_gf gf;
  omega_sigma_gf gf2;
  typedef omega_sigma_gf gf_type;

  TailedGreensFunctionMPI():beta(10), nsites(5), nfreq(10), nspins(2),
                      rank(alps::mpi::communicator().rank()),
                      is_root(alps::mpi::communicator().rank()==MASTER),
                      gf(matsubara_mesh(beta,nfreq), index_mesh(nsites)),
                      gf2(gf) {}
};


TEST_F(TailedGreensFunctionMPI, Broadcast){
  double new_val = 15.0;
  alps::gf::index_mesh y(nsites);
  // set value in GF
  gf(matsubara_mesh::index_type(0), index_mesh::index_type(0)) = new_val;
  // create tail
  tail tail1(y);
  tail1(alps::gf::index_mesh::index_type(0)) = 10.0;
  // create tailed GF
  gfns::gf_tail<head, tail> gft(gf);
  // attach tail
  gft.set_tail(0, tail1);
  if(!is_root) {
    gfns::gf_tail<head, tail> gf3;
    gf3.broadcast(alps::mpi::communicator(), MASTER);
    // check that GF is unchanged
    ASSERT_DOUBLE_EQ(gf(matsubara_mesh::index_type(0), index_mesh::index_type(0)).real(), new_val);
    // check that new GF is the same as old
    ASSERT_EQ(gf3, gft);
    auto tail2 = gf3.tail(0);
    ASSERT_EQ(tail1, tail2);
  } else {
    gft.broadcast(alps::mpi::communicator(), MASTER);
  }
}


int main(int argc, char**argv)
{
  alps::mpi::environment env(argc, argv, false);
  alps::gtest_par_xml_output tweak;
  tweak(alps::mpi::communicator().rank(), argc, argv);


  ::testing::InitGoogleTest(&argc, argv);

  Mpi_guard guard(0, "gf_new_tail_test_mpi.dat.");

  int rc=RUN_ALL_TESTS();

  if (!guard.check_sig_files_ok(get_number_of_bcasts())) {
    MPI_Abort(MPI_COMM_WORLD, 1); // otherwise it may get stuck in MPI_Finalize().
    // downside is the test aborts, rather than reports failure!
  }

  return rc;
}
