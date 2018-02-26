/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/utilities/gtest_par_xml_output.hpp>
#include "one_index_gf_test.hpp"

TEST_F(OneIndexGFTest,MpiBroadcast)
{
    int rank=alps::mpi::communicator().rank();
    const int master=0;
    
    alps::gf::matsubara_index omega; omega=4;
    gf.initialize();

    if (rank==master) {
        gf(omega)=std::complex<double>(3,4);
    }

    if (rank!=master) {
        std::complex<double> x=gf(omega);
        EXPECT_EQ(0.0, x.real());
        EXPECT_EQ(0.0, x.imag());
    }

    gf.broadcast(alps::mpi::communicator(), master);

    {
        std::complex<double> x=gf(omega);
        EXPECT_NEAR(3, x.real(),1.e-10);
        EXPECT_NEAR(4, x.imag(),1.e-10);
    }
}

// Check incompatible mesh broadcast
TEST_F(OneIndexGFTest,MpiWrongBroadcast)
{
    int rank=alps::mpi::communicator().rank();
    const int master=0;
    
    alps::gf::matsubara_index omega; omega=4;

    const int f=(rank==master)?1:2;
    alps::gf::omega_gf gf_wrong(matsubara_mesh(f*beta,f*nfreq));
    gf_wrong.initialize();
    if (rank==master) {
        gf_wrong(omega)=std::complex<double>(3,4);
    }

    gf_wrong.broadcast(alps::mpi::communicator(), master);
    
    {
        std::complex<double> x=gf_wrong(omega);
        EXPECT_NEAR(3, x.real(),1.e-10);
        EXPECT_NEAR(4, x.imag(),1.e-10);
    }
}


// if testing MPI, we need main()
int main(int argc, char**argv)
{
    alps::mpi::environment env(argc, argv, false);
    alps::gtest_par_xml_output tweak;
    tweak(alps::mpi::communicator().rank(), argc, argv);

    ::testing::InitGoogleTest(&argc, argv);
    int rc=RUN_ALL_TESTS();;

    return rc;
}
