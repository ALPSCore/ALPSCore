/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include "four_index_gf_test.hpp"
#include <alps/utilities/gtest_par_xml_output.hpp>

TEST_F(FourIndexGFTest,MpiBroadcast)
{
    int rank=alps::mpi::communicator().rank();
    const int master=0;

    alps::gf::matsubara_index omega; omega=4;
    alps::gf::momentum_index i; i=2;
    alps::gf::momentum_index j=alps::gf::momentum_index(3);
    alps::gf::index sigma(1);

    gf.initialize();

    if (rank==master) {
      gf(omega,i,j,sigma)=std::complex<double>(3,4);
    }

    if (rank!=master) {
      std::complex<double> x=gf(omega,i,j,sigma);
        EXPECT_EQ(0.0, x.real());
        EXPECT_EQ(0.0, x.imag());
    }

    gf.broadcast(alps::mpi::communicator(), master);

    {
      std::complex<double> x=gf(omega,i,j,sigma);
        EXPECT_NEAR(3, x.real(),1.e-10);
        EXPECT_NEAR(4, x.imag(),1.e-10);
    }
}

TEST_F(FourIndexGFTest, MpiTailBroadcast)
{
    namespace g=alps::gf;
    typedef g::three_index_gf<double, g::momentum_index_mesh, g::momentum_index_mesh, g::index_mesh> density_matrix_type;
    density_matrix_type denmat=density_matrix_type(g::momentum_index_mesh(get_data_for_momentum_mesh()), g::momentum_index_mesh(get_data_for_momentum_mesh()),
                                                   g::index_mesh(nspins));

    int rank=alps::mpi::communicator().rank();
    const int master=0;

    // prepare diagonal matrix
    double U=3.0;
    denmat.initialize();
    for (g::momentum_index i=g::momentum_index(0); i<denmat.mesh1().extent(); ++i) {
      denmat(i,i,g::index(0))=0.5*U;
      denmat(i,i,g::index(1))=0.5*U;
    }

    g::omega_k1_k2_sigma_gf_with_tail gft(gf);
    int order=0;
    if (rank==master) { // on master only
      // attach a tail to the GF:
      gft.set_tail(order, denmat);
      // change the GF
      gft(g::matsubara_index(4),g::momentum_index(3), g::momentum_index(2), g::index(1))=std::complex<double>(7., 3.);
    } else { // on receiving ranks, make sure that the tail is there, with empty data
      gft.set_tail(order, density_matrix_type(g::momentum_index_mesh(denmat.mesh1().points()),
                                              g::momentum_index_mesh(denmat.mesh2().points()),
                                              g::index_mesh(nspins)));
    }

    // broadcast the GF with tail
    gft.broadcast(alps::mpi::communicator(), master);

    EXPECT_EQ(7, gft(g::matsubara_index(4),g::momentum_index(3), g::momentum_index(2), g::index(1)).real()) << "GF real part mismatch on rank " << rank;
    EXPECT_EQ(3, gft(g::matsubara_index(4),g::momentum_index(3), g::momentum_index(2), g::index(1)).imag()) << "GF imag part mismatch on rank " << rank;

    ASSERT_EQ(1u, gft.tail().size()) << "Tail size mismatch on rank " << rank;
    EXPECT_NEAR(0, (gft.tail(0)-denmat).norm(), 1E-8) << "Tail broadcast differs from the received on rank " << rank;
}

// if testing MPI, we need main()
int main(int argc, char**argv)
{
    alps::mpi::environment env(argc, argv, false);
    alps::gtest_par_xml_output tweak;
    tweak(alps::mpi::communicator().rank(), argc, argv);

    ::testing::InitGoogleTest(&argc, argv);
    int rc=RUN_ALL_TESTS();

    return rc;
}
