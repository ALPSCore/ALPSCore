/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include "one_index_gf_test.hpp"
#include "alps/gf/grid.hpp"

TEST_F(OneIndexGFTest,access)
{
    alps::gf::matsubara_index omega; omega=4;

    gf(omega)=std::complex<double>(3,4);
    std::complex<double> x=gf(omega);
    EXPECT_EQ(3, x.real());
    EXPECT_EQ(4, x.imag());
}

TEST_F(OneIndexGFTest,init)
{
    alps::gf::matsubara_index omega; omega=4;

    gf.initialize();
    std::complex<double> x=gf(omega);
    EXPECT_EQ(0, x.real());
    EXPECT_EQ(0, x.imag());
}

TEST_F(OneIndexGFTest,saveload)
{
    namespace g=alps::gf;
    {
        alps::hdf5::archive oar("gf_1i_saveload.h5","w");
        gf(g::matsubara_index(4))=std::complex<double>(7., 3.);
        gf.save(oar,"/gf");
    }
    {
        alps::hdf5::archive iar("gf_1i_saveload.h5");
        gf2.load(iar,"/gf");
    }
    EXPECT_EQ(7, gf2(g::matsubara_index(4)).real());
    EXPECT_EQ(3, gf2(g::matsubara_index(4)).imag());
    {
        alps::hdf5::archive oar("gf_1i_saveload.h5","rw");
        oar["/gf/version/major"]<<7;
        EXPECT_THROW(gf2.load(oar,"/gf"),std::runtime_error);
    }
    EXPECT_EQ(7, gf2(g::matsubara_index(4)).real());
    EXPECT_EQ(3, gf2(g::matsubara_index(4)).imag());
}

TEST_F(OneIndexGFTest,saveloadstream)
{
    namespace g=alps::gf;
    {
        alps::hdf5::archive oar("gf_1i_saveloadstr.h5","w");
        gf(g::matsubara_index(4))=std::complex<double>(7., 3.);
        oar["/gf"] << gf;
    }
    {
        alps::hdf5::archive iar("gf_1i_saveloadstr.h5");
        iar["/gf"] >> gf2;
    }
    EXPECT_EQ(7, gf2(g::matsubara_index(4)).real());
    EXPECT_EQ(3, gf2(g::matsubara_index(4)).imag());
    {
        alps::hdf5::archive oar("gf_1i_saveloadstr.h5","rw");
        oar["/gf/version/major"]<<7;
        EXPECT_THROW(oar["/gf"]>>gf2, std::runtime_error);
    }
    EXPECT_EQ(7, gf2(g::matsubara_index(4)).real());
    EXPECT_EQ(3, gf2(g::matsubara_index(4)).imag());
}



TEST_F(OneIndexGFTest,print)
{
  std::stringstream gf_stream;
  gf_stream<<gf;

  std::stringstream gf_stream_by_hand;
  gf_stream_by_hand<<matsubara_mesh(beta,nfreq);
  for(int i=0;i<nfreq;++i){
    gf_stream_by_hand<<(2*i+1)*M_PI/beta<<" 0 0 "<<std::endl;
  }
  EXPECT_EQ(gf_stream_by_hand.str(), gf_stream.str());
}

TEST_F(OneIndexGFTest,scaling)
{
    alps::gf::matsubara_index omega; omega=4;

    gf(omega)=std::complex<double>(3,4);
    gf *= 2.;
    std::complex<double> x=gf(omega);
    EXPECT_NEAR(6, x.real(),1.e-10);
    EXPECT_NEAR(8, x.imag(),1.e-10);

    alps::gf::omega_gf gf1=gf/2;
    std::complex<double> x1=gf1(omega);
    EXPECT_NEAR(3, x1.real(),1.e-10);
    EXPECT_NEAR(4, x1.imag(),1.e-10);
}

TEST_F(OneIndexGFTest,negation)
{
    alps::gf::matsubara_index omega; omega=4;

    gf(omega)=std::complex<double>(3,4);
    alps::gf::omega_gf gf_neg=-gf;

    std::complex<double> x=gf_neg(omega);
    EXPECT_NEAR(-3, x.real(),1.e-10);
    EXPECT_NEAR(-4, x.imag(),1.e-10);
}

TEST_F(OneIndexGFTest,Assign)
{
    namespace g=alps::gf;
    g::omega_gf other_gf(matsubara_mesh(beta, nfreq*2));
    const g::matsubara_index omega(4);
    const std::complex<double> data(3,4);
    gf(omega)=data;
    
    gf2=gf;
    EXPECT_EQ(data, gf2(omega));

    EXPECT_NO_THROW(other_gf=gf);
    EXPECT_EQ(data, other_gf(omega));
}

TEST_F(OneIndexGFTest, RealFreq) {
    namespace g=alps::gf;
    double Emin = -5;
    double Emax = 5;
    int nfreq = 20;

    alps::gf::grid::linear_real_frequency_grid grid(Emin, Emax, nfreq);
    g::one_index_gf<std::complex<double>, g::real_frequency_mesh> other_gf((g::real_frequency_mesh(grid)));
    g::real_freq_index omega(4);
    const std::complex<double> data(3,4);
    const std::complex<double> data2(0,0);
    other_gf(omega)=data;
    EXPECT_EQ(data, other_gf(omega));
    EXPECT_EQ(data2, other_gf(++omega));
}

TEST_F(OneIndexGFTest, Legendre) {
    namespace g=alps::gf;
    int nl = 20;

    g::one_index_gf<std::complex<double>, g::legendre_mesh> other_gf(g::legendre_mesh(20.0, nl));
    g::legendre_index il(4);
    const std::complex<double> data(3,4);
    const std::complex<double> data2(0,0);
    other_gf(il)=data;
    EXPECT_EQ(data, other_gf(il));
    EXPECT_EQ(data2, other_gf(++il));
}

TEST_F(OneIndexGFTest, DefaultConstructiveReadHDF5)
{
    gf_type gf_empty;
    EXPECT_TRUE(gf_empty.is_empty());
    {
        alps::hdf5::archive oar("gf_1i_defconstr.h5","w");
        oar["/gf"] << gf;
    }
    {
        alps::hdf5::archive iar("gf_1i_defconstr.h5");
        iar["/gf"] >> gf_empty;
    }
    EXPECT_FALSE(gf_empty.is_empty());
}

#ifndef NDEBUG
TEST_F(OneIndexGFTest, DefaultConstructiveAccess) {
    gf_type gf_empty;
    EXPECT_ANY_THROW(gf_empty.norm());
    EXPECT_ANY_THROW(gf_empty*1.0);
    EXPECT_ANY_THROW(-gf_empty);
}
#endif