/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

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

TEST_F(FourIndexGFTest,negation)
{
    alps::gf::matsubara_index omega; omega=4;
    alps::gf::momentum_index i; i=2;
    alps::gf::momentum_index j=alps::gf::momentum_index(3);
    alps::gf::index sigma(1);

    gf(omega,i,j,sigma)=std::complex<double>(3,4);
    alps::gf::matsubara_gf gf_neg=-gf;

    std::complex<double> x=gf_neg(omega,i,j,sigma);
    EXPECT_NEAR(-3, x.real(),1.e-10);
    EXPECT_NEAR(-4, x.imag(),1.e-10);
}

TEST_F(FourIndexGFTest,Assign)
{
    namespace g=alps::gf;
    g::matsubara_gf other_gf(matsubara_mesh(beta, nfreq),
                             g::momentum_index_mesh(get_data_for_momentum_mesh()),
                             g::momentum_index_mesh(get_data_for_momentum_mesh()),
                             g::index_mesh(nspins*2));
    const g::matsubara_index omega(4);
    const alps::gf::momentum_index i(2), j(3);
    const g::index sigma(0);
    const std::complex<double> data(3,4);
    gf(omega,i,j,sigma)=data;
    
    gf2=gf;
    EXPECT_EQ(data, gf2(omega,i,j,sigma));
    EXPECT_NO_THROW(other_gf=gf);
    EXPECT_EQ(data, other_gf(omega,i,j,sigma));
}


TEST_F(FourIndexGFTest,saveload)
{
    namespace g=alps::gf;
    {
        alps::hdf5::archive oar("gf_4i_saveload.h5","w");
        gf(g::matsubara_index(4),g::momentum_index(3), g::momentum_index(2), g::index(1))=std::complex<double>(7., 3.);
        gf.save(oar,"/gf");
    }
    {
        alps::hdf5::archive iar("gf_4i_saveload.h5");
        gf2.load(iar,"/gf");
    }
    EXPECT_EQ(7, gf2(g::matsubara_index(4),g::momentum_index(3), g::momentum_index(2), g::index(1)).real());
    EXPECT_EQ(3, gf2(g::matsubara_index(4),g::momentum_index(3), g::momentum_index(2), g::index(1)).imag());
    {
        alps::hdf5::archive oar("gf_4i_saveload.h5","rw");
        oar["/gf/version/major"]<<7;
        EXPECT_THROW(gf2.load(oar,"/gf"),std::runtime_error);
    }
    EXPECT_EQ(7, gf2(g::matsubara_index(4),g::momentum_index(3), g::momentum_index(2), g::index(1)).real());
    EXPECT_EQ(3, gf2(g::matsubara_index(4),g::momentum_index(3), g::momentum_index(2), g::index(1)).imag());
    
    
    //boost::filesystem::remove("g5.h5");
}

TEST_F(FourIndexGFTest,saveloadstream)
{
    namespace g=alps::gf;
    {
        alps::hdf5::archive oar("gf_4i_stream.h5","w");
        gf(g::matsubara_index(4),g::momentum_index(3), g::momentum_index(2), g::index(1))=std::complex<double>(7., 3.);
        oar["/gf"] << gf;
    }
    {
        alps::hdf5::archive iar("gf_4i_stream.h5");
        iar["/gf"] >> gf2;
    }
    EXPECT_EQ(7, gf2(g::matsubara_index(4),g::momentum_index(3), g::momentum_index(2), g::index(1)).real());
    EXPECT_EQ(3, gf2(g::matsubara_index(4),g::momentum_index(3), g::momentum_index(2), g::index(1)).imag());
    {
        alps::hdf5::archive oar("gf_4i_stream.h5","rw");
        oar["/gf/version/major"]<<7;
        EXPECT_THROW(oar["/gf"]>>gf2, std::runtime_error);
    }
    EXPECT_EQ(7, gf2(g::matsubara_index(4),g::momentum_index(3), g::momentum_index(2), g::index(1)).real());
    EXPECT_EQ(3, gf2(g::matsubara_index(4),g::momentum_index(3), g::momentum_index(2), g::index(1)).imag());
    
    
    //boost::filesystem::remove("gf_stream.h5");
}


TEST_F(FourIndexGFTest, tail)
{
    namespace g=alps::gf;
    typedef g::three_index_gf<double, g::momentum_index_mesh, g::momentum_index_mesh, g::index_mesh> density_matrix_type;
    density_matrix_type denmat=density_matrix_type(g::momentum_index_mesh(get_data_for_momentum_mesh()),g::momentum_index_mesh(get_data_for_momentum_mesh()),
                                                   g::index_mesh(nspins));

    // Assign something to GF
    const g::matsubara_index omega(4);
    const alps::gf::momentum_index i(2), j(3);
    const g::index sigma(0);
    const std::complex<double> data(3,4);
    gf(omega,i,j,sigma)=data;

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

    // Check tail assignment
    g::omega_k1_k2_sigma_gf_with_tail gft2(gf2);
    gft2=gft;
    EXPECT_NEAR((gft2-gf).norm(), 0, 1E-8);
    EXPECT_NEAR((denmat-gft2.tail(order)).norm(), 0, 1.e-8);

    // Check mismatched-mesh tail assignment
    g::matsubara_gf other_gf(matsubara_mesh(beta,nfreq),
                             g::momentum_index_mesh(get_data_for_momentum_mesh()),
                             g::momentum_index_mesh(get_data_for_momentum_mesh()),
                             g::index_mesh(nspins*2));
    g::omega_k1_k2_sigma_gf_with_tail other_gft(other_gf);
    EXPECT_NO_THROW(other_gft=gft);
    EXPECT_EQ(gft, other_gft);
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
        alps::hdf5::archive oar("gf_4i_tail.h5","w");
        gft(g::matsubara_index(4),g::momentum_index(3), g::momentum_index(2), g::index(1))=std::complex<double>(7., 3.);
        oar["/gft"] << gft;
    }
    {
        alps::hdf5::archive iar("gf_4i_tail.h5");
        
        iar["/gft"] >> gft2;
    }
    EXPECT_EQ(gft2.tail().size(), gft.tail().size()) << "Tail size mismatch";
    EXPECT_NEAR(0, (gft.tail(0)-gft2.tail(0)).norm(), 1E-8)<<"Tail loaded differs from tail stored"; 

    EXPECT_EQ(7, gft2(g::matsubara_index(4),g::momentum_index(3), g::momentum_index(2), g::index(1)).real()) << "GF real part mismatch";
    EXPECT_EQ(3, gft2(g::matsubara_index(4),g::momentum_index(3), g::momentum_index(2), g::index(1)).imag()) << "GF imag part mismatch";
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

TEST_F(FourIndexGFTest, DefaultConstructive)
{
    gf_type gf_empty;
    EXPECT_TRUE(gf_empty.is_empty());
    {
        alps::hdf5::archive oar("gf_4i_defconstr.h5","w");
        oar["/gf"] << gf;
    }
    {
        alps::hdf5::archive iar("gf_4i_defconstr.h5");
        iar["/gf"] >> gf_empty;
    }
    EXPECT_FALSE(gf_empty.is_empty());
}

#ifndef NDEBUG
TEST_F(FourIndexGFTest, DefaultConstructiveAccess) {
    gf_type gf_empty;
    EXPECT_ANY_THROW(gf_empty.norm());
    EXPECT_ANY_THROW(gf_empty*1.0);
    EXPECT_ANY_THROW(-gf_empty);
}
#endif
