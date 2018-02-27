/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include "gtest/gtest.h"

#include <alps/gf/gf.hpp>

class TwoIndexGFTest : public ::testing::Test
{
  public:
    const double beta;
    const int nsites;
    const int nfreq ;
    const int nspins;
    alps::gf::omega_sigma_gf gf;
    alps::gf::omega_sigma_gf gf2;
    typedef alps::gf::matsubara_mesh<alps::gf::mesh::POSITIVE_ONLY> matsubara_mesh;
    typedef alps::gf::omega_sigma_gf gf_type;

    TwoIndexGFTest():beta(10), nsites(4), nfreq(10), nspins(2),
             gf(matsubara_mesh(beta,nfreq),
                alps::gf::index_mesh(nspins)),
             gf2(gf) {}
};
    

TEST_F(TwoIndexGFTest,access)
{
    alps::gf::matsubara_index omega; omega=4;
    alps::gf::index sigma(1);

    gf(omega, sigma)=std::complex<double>(3,4);
    std::complex<double> x=gf(omega,sigma);
    EXPECT_EQ(3, x.real());
    EXPECT_EQ(4, x.imag());
}

TEST_F(TwoIndexGFTest,init)
{
    alps::gf::matsubara_index omega; omega=4;
    alps::gf::index sigma(1);

    gf.initialize();
    std::complex<double> x=gf(omega,sigma);
    EXPECT_EQ(0, x.real());
    EXPECT_EQ(0, x.imag());
}

TEST_F(TwoIndexGFTest,scaling)
{
    alps::gf::matsubara_index omega; omega=4;
    alps::gf::index sigma(1);

    gf(omega,sigma)=std::complex<double>(3,4);
    gf *= 2.;
    std::complex<double> x=gf(omega,sigma);
    EXPECT_NEAR(6, x.real(),1.e-10);
    EXPECT_NEAR(8, x.imag(),1.e-10);

    alps::gf::omega_sigma_gf gf1=gf / 2;
    std::complex<double> x1=gf1(omega,sigma);
    EXPECT_NEAR(3, x1.real(),1.e-10);
    EXPECT_NEAR(4, x1.imag(),1.e-10);
}

TEST_F(TwoIndexGFTest,negation)
{
    alps::gf::matsubara_index omega; omega=4;
    alps::gf::index sigma(1);

    gf(omega,sigma)=std::complex<double>(3,4);
    alps::gf::omega_sigma_gf gf_neg=-gf;

    std::complex<double> x=gf_neg(omega,sigma);
    EXPECT_NEAR(-3, x.real(),1.e-10);
    EXPECT_NEAR(-4, x.imag(),1.e-10);
}

TEST_F(TwoIndexGFTest,saveload)
{
    namespace g=alps::gf;
    {
        alps::hdf5::archive oar("gf_2i_saveload.h5","w");
        gf(g::matsubara_index(4), g::index(1))=std::complex<double>(7., 3.);
        gf.save(oar,"/gf");
    }
    {
        alps::hdf5::archive iar("gf_2i_saveload.h5");
        gf2.load(iar,"/gf");
    }
    EXPECT_EQ(7, gf2(g::matsubara_index(4), g::index(1)).real());
    EXPECT_EQ(3, gf2(g::matsubara_index(4), g::index(1)).imag());
    {
        alps::hdf5::archive oar("gf_2i_saveload.h5","rw");
        oar["/gf/version/major"]<<7;
        EXPECT_THROW(gf2.load(oar,"/gf"),std::runtime_error);
    }
    EXPECT_EQ(7, gf2(g::matsubara_index(4), g::index(1)).real());
    EXPECT_EQ(3, gf2(g::matsubara_index(4), g::index(1)).imag());
}

TEST_F(TwoIndexGFTest,saveloadstream)
{
    namespace g=alps::gf;
    {
        alps::hdf5::archive oar("gf_2i_saveloadstr.h5","w");
        gf(g::matsubara_index(4), g::index(1))=std::complex<double>(7., 3.);
        oar["/gf"] << gf;
    }
    {
        alps::hdf5::archive iar("gf_2i_saveloadstr.h5");
        iar["/gf"] >> gf2;
    }
    EXPECT_EQ(7, gf2(g::matsubara_index(4), g::index(1)).real());
    EXPECT_EQ(3, gf2(g::matsubara_index(4), g::index(1)).imag());
    {
        alps::hdf5::archive oar("gf_2i_saveloadstr.h5","rw");
        oar["/gf/version/major"]<<7;
        EXPECT_THROW(oar["/gf"]>>gf2, std::runtime_error);
    }
    EXPECT_EQ(7, gf2(g::matsubara_index(4), g::index(1)).real());
    EXPECT_EQ(3, gf2(g::matsubara_index(4), g::index(1)).imag());
}



TEST_F(TwoIndexGFTest,print)
{
  std::stringstream gf_stream;
  gf_stream<<gf;

  std::stringstream gf_stream_by_hand;
  gf_stream_by_hand<<matsubara_mesh(beta,nfreq)<<alps::gf::index_mesh(2);
  for(int i=0;i<nfreq;++i){
    gf_stream_by_hand<<(2*i+1)*M_PI/beta<<" 0 0 0 0 "<<std::endl;
  }
  EXPECT_EQ(gf_stream_by_hand.str(), gf_stream.str());
}

TEST_F(TwoIndexGFTest,printindexmesh)
{
  std::stringstream gf_stream;
  int nao=6;
  alps::gf::index_mesh mesh(nao);
  alps::gf::two_index_gf<double, alps::gf::index_mesh,alps::gf::index_mesh> gf2(mesh,mesh);
  gf2.initialize();
  gf_stream<<gf2;

  std::stringstream gf_stream_by_hand;
  gf_stream_by_hand<<alps::gf::index_mesh(nao)<<alps::gf::index_mesh(nao);
  for(int i=0;i<nao;++i){
    gf_stream_by_hand<<i<<" ";
    for(int j=0;j<nao;++j){
      gf_stream_by_hand<<"0 ";
    }
    gf_stream_by_hand<<std::endl;
  }
  EXPECT_EQ(gf_stream_by_hand.str(), gf_stream.str());
}


TEST_F(TwoIndexGFTest, tail)
{
    namespace g=alps::gf;
    typedef g::one_index_gf<double, g::index_mesh> density_matrix_type;
    density_matrix_type denmat=density_matrix_type(g::index_mesh(nspins));

    // prepare diagonal matrix
    double U=3.0;
    denmat.initialize();
    denmat(g::index(0))=0.5*U;
    denmat(g::index(1))=0.5*U;

    // Attach a tail to the GF
    int order=0;

    g::omega_sigma_gf_with_tail gft(gf);
    gft.set_tail(order, denmat);

    EXPECT_NEAR((denmat-gft.tail(order)).norm(), 0, 1.e-8);
}

// FIXME: does not test the validity of print output
TEST_F(TwoIndexGFTest, tailPrint)
{
    namespace g=alps::gf;
    typedef g::one_index_gf<double, g::index_mesh> density_matrix_type;
    density_matrix_type denmat=density_matrix_type(g::index_mesh(nspins));

    // prepare diagonal matrix
    double U=3.0;
    denmat.initialize();
    denmat(g::index(0))=0.5*U;
    denmat(g::index(1))=0.5*U;

    // Attach a tail to the GF
    int order=0;

    g::omega_sigma_gf_with_tail gft(gf);
    gft.set_tail(order, denmat);

    std::ostringstream outs;
    ASSERT_NO_THROW(outs << gft.tail(0));
}

TEST_F(TwoIndexGFTest, TailSaveLoad)
{
    namespace g=alps::gf;
    typedef g::one_index_gf<double, g::index_mesh> density_matrix_type;
    density_matrix_type denmat=density_matrix_type(g::index_mesh(nspins));

    // prepare diagonal matrix
    double U=3.0;
    denmat.initialize();
    denmat(g::index(0))=0.5*U;
    denmat(g::index(1))=0.5*U;

    // Attach a tail to the GF
    int order=0;

    g::omega_sigma_gf_with_tail gft(gf);
    g::omega_sigma_gf_with_tail gft2(gft);
    EXPECT_EQ(g::TAIL_NOT_SET,gft.min_tail_order());
    EXPECT_EQ(g::TAIL_NOT_SET,gft.max_tail_order());

    gft.set_tail(order, denmat);

    EXPECT_EQ(0,gft.min_tail_order());
    EXPECT_EQ(0,gft.max_tail_order());
    EXPECT_EQ(0,(denmat-gft.tail(0)).norm());
    {
        alps::hdf5::archive oar("gf_2i_tailsaveloadX.h5","w");
        gft(g::matsubara_index(4),g::index(1))=std::complex<double>(7., 3.);
        oar["/gft"] << gft;
    }
    {
        alps::hdf5::archive iar("gf_2i_tailsaveloadX.h5");

        iar["/gft"] >> gft2;
    }
    EXPECT_EQ(gft2.tail().size(), gft.tail().size()) << "Tail size mismatch";
    EXPECT_NEAR(0, (gft.tail(0)-gft2.tail(0)).norm(), 1E-8)<<"Tail loaded differs from tail stored";

    EXPECT_EQ(7, gft2(g::matsubara_index(4), g::index(1)).real()) << "GF real part mismatch";
    EXPECT_EQ(3, gft2(g::matsubara_index(4), g::index(1)).imag()) << "GF imag part mismatch";

}

TEST_F(TwoIndexGFTest,Assign)
{
    namespace g=alps::gf;
    g::omega_sigma_gf other_gf(matsubara_mesh(beta, nfreq*2), g::index_mesh(nspins));
    const g::matsubara_index omega(4);
    const g::index sigma(0);
    const std::complex<double> data(3,4);
    gf(omega,sigma)=data;
    
    gf2=gf;
    EXPECT_EQ(data, gf2(omega,sigma));
    EXPECT_NO_THROW(other_gf = gf);
    EXPECT_EQ(data, other_gf(omega,sigma));
}

TEST_F(TwoIndexGFTest, DefaultConstructive)
{
    gf_type gf_empty;
    EXPECT_TRUE(gf_empty.is_empty());
    {
        alps::hdf5::archive oar("gf_2i_defconstr.h5","w");
        oar["/gf"] << gf;
    }
    {
        alps::hdf5::archive iar("gf_2i_defconstr.h5");
        iar["/gf"] >> gf_empty;
    }
    EXPECT_FALSE(gf_empty.is_empty());
}

TEST_F(TwoIndexGFTest, ops)
{
  namespace g=alps::gf;
  typedef g::one_index_gf<double, g::index_mesh> density_matrix_type;
  density_matrix_type denmat=density_matrix_type(g::index_mesh(nspins));

  // prepare diagonal matrix
  double U=3.0;
  denmat.initialize();
  denmat(g::index(0))=0.5*U;
  denmat(g::index(1))=0.5*U;

  // Attach a tail to the GF
  int order=0;

  g::omega_sigma_gf_with_tail gft(gf);
  gft.set_tail(order, denmat);

  EXPECT_NEAR((denmat-gft.tail(order)).norm(), 0, 1.e-8);
}

#ifndef NDEBUG
TEST_F(TwoIndexGFTest, DefaultConstructiveAccess) {
  gf_type gf_empty;
  EXPECT_ANY_THROW(gf_empty.norm());
  EXPECT_ANY_THROW(gf_empty*1.0);
  EXPECT_ANY_THROW(-gf_empty);
}
#endif
