#include "four_index_gf_test.hpp"

// Check incompatible mesh broadcast (NOTE: must be in a separate file/process because it messes up MPI)
TEST_F(FourIndexGFTest,MpiWrongBroadcast)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const int master=0;
    
    alps::gf::matsubara_index omega; omega=4;
    alps::gf::momentum_index i; i=2;
    alps::gf::momentum_index j=alps::gf::momentum_index(3);
    alps::gf::index sigma(1);

    const int f=(rank==master)?1:2;
    alps::gf::matsubara_gf gf_wrong(matsubara_mesh(f*beta,f*nfreq),
                                    alps::gf::momentum_index_mesh(5,1),
                                    alps::gf::momentum_index_mesh(5,1),
                                    alps::gf::index_mesh(nspins));
    gf_wrong.initialize();
    if (rank==master) {
      gf_wrong(omega,i,j,sigma)=std::complex<double>(3,4);
    }

    bool thrown=false;
    try {
        gf_wrong.broadcast_data(master,MPI_COMM_WORLD);
    } catch (...) {
        thrown=true;
    }
    EXPECT_TRUE( (rank==master && !thrown) || (rank!=master && thrown) );

    if (thrown) return;
    
    {
      std::complex<double> x=gf_wrong(omega,i,j,sigma);
        EXPECT_NEAR(3, x.real(),1.e-10);
        EXPECT_NEAR(4, x.imag(),1.e-10);
    }
}

// for testing MPI, we need main()
int main(int argc, char**argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int rc=RUN_ALL_TESTS();
    MPI_Finalize();
    return rc;
}
