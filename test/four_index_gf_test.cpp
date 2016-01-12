#include "four_index_gf_test.hpp"

TEST_F(FourIndexGFTest,access)
{
    alps::gf::matsubara_index omega; omega=4;
    alps::gf::momentum_index i; i=2;
    alps::gf::momentum_index j=alps::gf::momentum_index(3);
    alps::gf::index sigma(1);

    gf(omega, i,j,sigma)=std::complex<double>(3,4);
    std::complex<double> x=gf(omega,i,j,sigma);
    EXPECT_EQ(3, x.real());
    EXPECT_EQ(4, x.imag());
}

TEST_F(FourIndexGFTest,init)
{
    alps::gf::matsubara_index omega; omega=4;
    alps::gf::momentum_index i; i=2;
    alps::gf::momentum_index j=alps::gf::momentum_index(3);
    alps::gf::index sigma(1);

    gf.initialize();
    std::complex<double> x=gf(omega,i,j,sigma);
    EXPECT_EQ(0, x.real());
    EXPECT_EQ(0, x.imag());
}

TEST_F(FourIndexGFTest,scaling)
{
    alps::gf::matsubara_index omega; omega=4;
    alps::gf::momentum_index i; i=2;
    alps::gf::momentum_index j=alps::gf::momentum_index(3);
    alps::gf::index sigma(1);

    gf(omega,i,j,sigma)=std::complex<double>(3,4);
    gf *= 2.;
    std::complex<double> x=gf(omega,i,j,sigma);
    EXPECT_NEAR(6, x.real(),1.e-10);
    EXPECT_NEAR(8, x.imag(),1.e-10);

    alps::gf::matsubara_gf gf1=gf/2;
    std::complex<double> x1=gf1(omega,i,j,sigma);
    EXPECT_NEAR(3, x1.real(),1.e-10);
    EXPECT_NEAR(4, x1.imag(),1.e-10);
}

TEST_F(FourIndexGFTest,saveload)
{
    namespace g=alps::gf;
    {
        alps::hdf5::archive oar("gf.h5","w");
        gf(g::matsubara_index(4),g::momentum_index(3), g::momentum_index(2), g::index(1))=std::complex<double>(7., 3.);
        gf.save(oar,"/gf");
    }
    {
        alps::hdf5::archive iar("gf.h5");
        gf2.load(iar,"/gf");
    }
    EXPECT_EQ(7, gf2(g::matsubara_index(4),g::momentum_index(3), g::momentum_index(2), g::index(1)).real());
    EXPECT_EQ(3, gf2(g::matsubara_index(4),g::momentum_index(3), g::momentum_index(2), g::index(1)).imag());
    {
        alps::hdf5::archive oar("gf.h5","rw");
        oar["/gf/version/major"]<<7;
        EXPECT_THROW(gf2.load(oar,"/gf"),std::runtime_error);
    }
    EXPECT_EQ(7, gf2(g::matsubara_index(4),g::momentum_index(3), g::momentum_index(2), g::index(1)).real());
    EXPECT_EQ(3, gf2(g::matsubara_index(4),g::momentum_index(3), g::momentum_index(2), g::index(1)).imag());
    
    
    //boost::filesystem::remove("g5.h5");
}

TEST_F(FourIndexGFTest,MpiBroadcast)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
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

    gf.broadcast(master,MPI_COMM_WORLD);

    {
      std::complex<double> x=gf(omega,i,j,sigma);
        EXPECT_NEAR(3, x.real(),1.e-10);
        EXPECT_NEAR(4, x.imag(),1.e-10);
    }
}

TEST_F(FourIndexGFTest, tail)
{
    namespace g=alps::gf;
    typedef g::three_index_gf<double, g::momentum_index_mesh, g::momentum_index_mesh, g::index_mesh> density_matrix_type;
    density_matrix_type denmat=density_matrix_type(g::momentum_index_mesh(get_data_for_momentum_mesh()),g::momentum_index_mesh(get_data_for_momentum_mesh()),
                                                   g::index_mesh(nspins));

    // prepare diagonal matrix
    double U=3.0;
    denmat.initialize();
    for (g::momentum_index i=g::momentum_index(0); i<denmat.mesh1().extent(); ++i) {
        denmat(i,i,g::index(0))=0.5*U;
        denmat(i,i,g::index(1))=0.5*U;
    }

    // Attach a tail to the GF
    int order=0;
    
    // FIXME: TODO: gf.set_tail(min_order, max_order, denmat, ...);
    g::omega_k1_k2_sigma_gf_with_tail gft(gf);
    gft.set_tail(order, denmat)
    // .set_tail(order+1, other_gf) ....
        ;
    
    EXPECT_NEAR((denmat-gft.tail(order)).norm(), 0, 1.e-8);
}

TEST_F(FourIndexGFTest, TailSaveLoad)
{
    namespace g=alps::gf;
    typedef g::three_index_gf<double, g::momentum_index_mesh, g::momentum_index_mesh, g::index_mesh> density_matrix_type;
    density_matrix_type denmat=density_matrix_type(g::momentum_index_mesh(get_data_for_momentum_mesh()), g::momentum_index_mesh(get_data_for_momentum_mesh()),
                                                   g::index_mesh(nspins));

    // prepare diagonal matrix
    double U=3.0;
    denmat.initialize();
    for (g::momentum_index i=g::momentum_index(0); i<denmat.mesh1().extent(); ++i) {
        denmat(i,i,g::index(0))=0.5*U;
        denmat(i,i,g::index(1))=0.5*U;
    }

    // Attach a tail to the GF
    int order=0;
    
    // FIXME: TODO: gf.set_tail(min_order, max_order, denmat, ...);
    g::omega_k1_k2_sigma_gf_with_tail gft(gf);
    g::omega_k1_k2_sigma_gf_with_tail gft2(gft);
    EXPECT_EQ(g::TAIL_NOT_SET,gft.min_tail_order());
    EXPECT_EQ(g::TAIL_NOT_SET,gft.max_tail_order());

    gft.set_tail(order, denmat);

    EXPECT_EQ(0,gft.min_tail_order());
    EXPECT_EQ(0,gft.max_tail_order());
    EXPECT_EQ(0,(denmat-gft.tail(0)).norm());
    {
        alps::hdf5::archive oar("gft.h5","w");
        gft(g::matsubara_index(4),g::momentum_index(3), g::momentum_index(2), g::index(1))=std::complex<double>(7., 3.);
        gft.save(oar,"/gft");
    }
    {
        alps::hdf5::archive iar("gft.h5");
        
        gft2.load(iar,"/gft");
    }
    EXPECT_EQ(gft2.tail().size(), gft.tail().size()) << "Tail size mismatch";
    EXPECT_NEAR(0, (gft.tail(0)-gft2.tail(0)).norm(), 1E-8)<<"Tail loaded differs from tail stored"; 

    EXPECT_EQ(7, gft2(g::matsubara_index(4),g::momentum_index(3), g::momentum_index(2), g::index(1)).real()) << "GF real part mismatch";
    EXPECT_EQ(3, gft2(g::matsubara_index(4),g::momentum_index(3), g::momentum_index(2), g::index(1)).imag()) << "GF imag part mismatch";
}

TEST_F(FourIndexGFTest, MpiTailBroadcast)
{
    namespace g=alps::gf;
    typedef g::three_index_gf<double, g::momentum_index_mesh, g::momentum_index_mesh, g::index_mesh> density_matrix_type;
    density_matrix_type denmat=density_matrix_type(g::momentum_index_mesh(get_data_for_momentum_mesh()), g::momentum_index_mesh(get_data_for_momentum_mesh()),
                                                   g::index_mesh(nspins));

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
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
    gft.broadcast(master,MPI_COMM_WORLD);

    EXPECT_EQ(7, gft(g::matsubara_index(4),g::momentum_index(3), g::momentum_index(2), g::index(1)).real()) << "GF real part mismatch on rank " << rank;
    EXPECT_EQ(3, gft(g::matsubara_index(4),g::momentum_index(3), g::momentum_index(2), g::index(1)).imag()) << "GF imag part mismatch on rank " << rank;

    ASSERT_EQ(1, gft.tail().size()) << "Tail size mismatch on rank " << rank;
    EXPECT_NEAR(0, (gft.tail(0)-denmat).norm(), 1E-8) << "Tail broadcast differs from the received on rank " << rank; 
}


TEST_F(FourIndexGFTest,print)
{
  std::stringstream gf_stream;
  gf_stream<<gf;

  std::stringstream gf_stream_by_hand;
  gf_stream_by_hand<<alps::gf::matsubara_positive_mesh(beta,nfreq)
  <<alps::gf::momentum_index_mesh(get_data_for_momentum_mesh())
  <<alps::gf::momentum_index_mesh(get_data_for_momentum_mesh())
      <<alps::gf::index_mesh(nspins);
  for(int i=0;i<nfreq;++i){
    gf_stream_by_hand<<(2*i+1)*M_PI/beta<<" 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "<<std::endl;
  }
  EXPECT_EQ(gf_stream_by_hand.str(), gf_stream.str());
}



// Service function to run tests with MPI
int run_tests()
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Run all non-MPI tests by master
    ::testing::GTEST_FLAG(filter) = "-*.Mpi*";
    if (rank==0) {
        int rc=RUN_ALL_TESTS();
        if (rc!=0) return rc;
    }
    // Run all MPI tests now
    ::testing::GTEST_FLAG(filter) = "*.Mpi*";
    int rc=RUN_ALL_TESTS();
    return rc;
}


// if testing MPI, we need main()
int main(int argc, char**argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int rc=run_tests();
    MPI_Finalize();
    return rc;
}
