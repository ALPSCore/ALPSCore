/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include "four_index_gf_test.hpp"
#include "mpi_guard.hpp"

#include <alps/utilities/gtest_par_xml_output.hpp>

/*
   NOTE: This program relies on file I/O to exchange info between processes
         --- do NOT run on too many cores!
*/


static const int MASTER=0;

// Check incompatible broadcast of tails
TEST_F(FourIndexGFTest,MpiWrongTailBroadcast)
{
    namespace g=alps::gf;
    typedef g::three_index_gf<double,
                              g::momentum_index_mesh,
                              g::momentum_index_mesh,
                              g::index_mesh> density_matrix_type;

    density_matrix_type denmat=density_matrix_type(g::momentum_index_mesh(get_data_for_momentum_mesh()),
                                                   g::momentum_index_mesh(get_data_for_momentum_mesh()),
                                                   g::index_mesh(this->nspins));

    // Get the rank
    int rank=alps::mpi::communicator().rank();

    // prepare diagonal matrix
    double U=3.0;
    denmat.initialize();
    for (g::momentum_index i=g::momentum_index(0); i<denmat.mesh1().extent(); ++i) {
      denmat(i,i,g::index(0))=0.5*U;
      denmat(i,i,g::index(1))=0.5*U;
    }

    g::omega_k1_k2_sigma_gf_with_tail gft(gf);

    int order=0;
    if (rank==MASTER) { // on master only
      // attach a tail to the GF:
      gft.set_tail(order, denmat);
      // change the GF
      gft(g::matsubara_index(4),g::momentum_index(3), g::momentum_index(2), g::index(1))=std::complex<double>(7., 3.);
    }
    // slaves do not have the tail attached to their GF.

    gft.broadcast(alps::mpi::communicator(), MASTER);

    EXPECT_EQ(7, gft(g::matsubara_index(4),g::momentum_index(3), g::momentum_index(2), g::index(1)).real()) << "GF real part mismatch on rank " << rank;
    EXPECT_EQ(3, gft(g::matsubara_index(4),g::momentum_index(3), g::momentum_index(2), g::index(1)).imag()) << "GF imag part mismatch on rank " << rank;

    ASSERT_EQ(1u, gft.tail().size()) << "Tail size mismatch on rank " << rank;
    EXPECT_NEAR(0, (gft.tail(0)-denmat).norm(), 1E-8) << "Tail broadcast differs from the received on rank " << rank;
}

// for testing MPI, we need main()
int main(int argc, char**argv)
{
    alps::mpi::environment env(argc, argv, false);
    alps::gtest_par_xml_output tweak;
    tweak(alps::mpi::communicator().rank(), argc, argv);

    ::testing::InitGoogleTest(&argc, argv);

    Mpi_guard guard(MASTER,"four_index_gf_test_mismatched-tail-mpi.dat.");

    int rc=RUN_ALL_TESTS();

    if (!guard.check_sig_files_ok(get_number_of_bcasts())) {
        MPI_Abort(MPI_COMM_WORLD, 1); // otherwise it may get stuck in MPI_Finalize().
        // downside is the test aborts, rather than reports failure!
    }

    return rc;
}
