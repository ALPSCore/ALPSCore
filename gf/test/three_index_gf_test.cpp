/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include "gtest/gtest.h"
#include <alps/gf/gf.hpp>
#include "gf_test.hpp"

class ThreeIndexGFTest : public ::testing::Test
{
  public:
    const double beta;
    const int nsites;
    const int nfreq ;
    const int nspins;
    typedef alps::gf::omega_k_sigma_gf gf_type;
    gf_type gf;
    gf_type gf2;

    ThreeIndexGFTest():beta(10), nsites(4), nfreq(10), nspins(2),
             gf(alps::gf::matsubara_positive_mesh(beta,nfreq),
                alps::gf::momentum_index_mesh(get_data_for_momentum_mesh()),
                alps::gf::index_mesh(nspins)),
             gf2(gf) {}
};

TEST_F(ThreeIndexGFTest,access)
{
    alps::gf::matsubara_index omega; omega=4;
    alps::gf::momentum_index i; i=2;
    alps::gf::index sigma(1);

    gf(omega, i,sigma)=std::complex<double>(3,4);
    std::complex<double> x=gf(omega,i,sigma);
    EXPECT_EQ(3, x.real());
    EXPECT_EQ(4, x.imag());
}

TEST_F(ThreeIndexGFTest,init)
{
    alps::gf::matsubara_index omega; omega=4;
    alps::gf::momentum_index i; i=2;
    alps::gf::index sigma(1);

    gf.initialize();
    std::complex<double> x=gf(omega,i,sigma);
    EXPECT_EQ(0, x.real());
    EXPECT_EQ(0, x.imag());
}

TEST_F(ThreeIndexGFTest,saveload)
{
    namespace g=alps::gf;
    {
        alps::hdf5::archive oar("gf_3i_saveload.h5","w");
        gf(g::matsubara_index(4),g::momentum_index(3), g::index(1))=std::complex<double>(7., 3.);
        gf.save(oar,"/gf");
    }
    {
        alps::hdf5::archive iar("gf_3i_saveload.h5");
        gf2.load(iar,"/gf");
    }
    EXPECT_EQ(7, gf2(g::matsubara_index(4),g::momentum_index(3), g::index(1)).real());
    EXPECT_EQ(3, gf2(g::matsubara_index(4),g::momentum_index(3), g::index(1)).imag());
    {
        alps::hdf5::archive oar("gf_3i_saveload.h5","rw");
        oar["/gf/version/major"]<<7;
        EXPECT_THROW(gf2.load(oar,"/gf"),std::runtime_error);
    }
    EXPECT_EQ(7, gf2(g::matsubara_index(4),g::momentum_index(3), g::index(1)).real());
    EXPECT_EQ(3, gf2(g::matsubara_index(4),g::momentum_index(3), g::index(1)).imag());
}

TEST_F(ThreeIndexGFTest,saveloadstream)
{
    namespace g=alps::gf;
    {
        alps::hdf5::archive oar("gf_3i_saveloadstr.h5","w");
        gf(g::matsubara_index(4),g::momentum_index(3), g::index(1))=std::complex<double>(7., 3.);
        oar["/gf"] << gf;
    }
    {
        alps::hdf5::archive iar("gf_3i_saveloadstr.h5");
        iar["/gf"] >> gf2;
    }
    EXPECT_EQ(7, gf2(g::matsubara_index(4),g::momentum_index(3), g::index(1)).real());
    EXPECT_EQ(3, gf2(g::matsubara_index(4),g::momentum_index(3), g::index(1)).imag());
    {
        alps::hdf5::archive oar("gf_3i_saveloadstr.h5","rw");
        oar["/gf/version/major"]<<7;
        EXPECT_THROW(oar["/gf"]>>gf2, std::runtime_error);
    }
    EXPECT_EQ(7, gf2(g::matsubara_index(4),g::momentum_index(3), g::index(1)).real());
    EXPECT_EQ(3, gf2(g::matsubara_index(4),g::momentum_index(3), g::index(1)).imag());
}

TEST_F(ThreeIndexGFTest, tail)
{
    namespace g=alps::gf;
    typedef g::two_index_gf<double, g::momentum_index_mesh, g::index_mesh> density_matrix_type;
    density_matrix_type denmat=density_matrix_type(g::momentum_index_mesh(get_data_for_momentum_mesh()),
                                      g::index_mesh(nspins));

    // prepare diagonal matrix
    double U=3.0;
    denmat.initialize();
    for (g::momentum_index i=g::momentum_index(0); i<denmat.mesh1().extent(); ++i) {
        denmat(i,g::index(0))=0.5*U;
        denmat(i,g::index(1))=0.5*U;
    }

    // Attach a tail to the GF
    int order=0;
    
    // FIXME: TODO: gf.set_tail(min_order, max_order, denmat, ...);
    g::omega_k_sigma_gf_with_tail gft(gf);
    gft.set_tail(order, denmat)
    // .set_tail(order+1, other_gf) ....
        ;
    
    EXPECT_NEAR((denmat-gft.tail(order)).norm(), 0, 1.e-8);

    /* The following does not compile, as expected:
       
    typedef g::three_index_gf<double, g::momentum_index_mesh, g::momentum_index_mesh, g::index_mesh> some_matrix_type;
    typedef g::three_index_gf_with_tail<some_matrix_type, g::two_index_gf<double, g::momentum_index_mesh, g::index_mesh> > some_matrix_type_with_tail;
    some_matrix_type* ptr=0;
    some_matrix_type_with_tail wrong(*ptr);
    */
    

    /* The following does not compile, as expected:

    typedef g::three_index_gf<double, g::itime_mesh, g::itime_mesh, g::index_mesh> some_gf_type;
    typedef g::three_index_gf_with_tail<some_gf_type, g::two_index_gf<double, g::momentum_index_mesh, g::index_mesh> > some_gf_type_with_tail;
    some_gf_type_with_tail wrong(*(some_gf_type*)0);
    */
}

TEST_F(ThreeIndexGFTest, TailSaveLoad)
{
    namespace g=alps::gf;
    typedef g::two_index_gf<double, g::momentum_index_mesh, g::index_mesh> density_matrix_type;
    density_matrix_type denmat=density_matrix_type(g::momentum_index_mesh(get_data_for_momentum_mesh()),
                                      g::index_mesh(nspins));

    // prepare diagonal matrix
    double U=3.0;
    denmat.initialize();
    for (g::momentum_index i=g::momentum_index(0); i<denmat.mesh1().extent(); ++i) {
        denmat(i,g::index(0))=0.5*U;
        denmat(i,g::index(1))=0.5*U;
    }

    // Attach a tail to the GF
    int order=0;
    
    // FIXME: TODO: gf.set_tail(min_order, max_order, denmat, ...);
    g::omega_k_sigma_gf_with_tail gft(gf);
    g::omega_k_sigma_gf_with_tail gft2(gft);
    EXPECT_EQ(g::TAIL_NOT_SET,gft.min_tail_order());
    EXPECT_EQ(g::TAIL_NOT_SET,gft.max_tail_order());

    gft.set_tail(order, denmat);

    EXPECT_EQ(0,gft.min_tail_order());
    EXPECT_EQ(0,gft.max_tail_order());
    EXPECT_EQ(0,(denmat-gft.tail(0)).norm());
    {
        alps::hdf5::archive oar("gf_3i_tail.h5","w");
        gft(g::matsubara_index(4),g::momentum_index(3), g::index(1))=std::complex<double>(7., 3.);
        oar["/gft"] << gft;
    }
    {
        alps::hdf5::archive iar("gf_3i_tail.h5");
        iar["/gft"] >> gft2;
    }
    EXPECT_EQ(gft2.tail().size(), gft.tail().size()) << "Tail size mismatch";
    EXPECT_NEAR(0, (gft.tail(0)-gft2.tail(0)).norm(), 1E-8)<<"Tail loaded differs from tail stored"; 

    EXPECT_EQ(7, gft2(g::matsubara_index(4),g::momentum_index(3), g::index(1)).real()) << "GF real part mismatch";
    EXPECT_EQ(3, gft2(g::matsubara_index(4),g::momentum_index(3), g::index(1)).imag()) << "GF imag part mismatch";
    
}

TEST_F(ThreeIndexGFTest,EqOperators)
{
    namespace g=alps::gf;

    for (g::matsubara_index om=g::matsubara_index(0); om<gf.mesh1().extent(); ++om) {
        for (g::momentum_index ii=g::momentum_index(0); ii<gf.mesh2().extent(); ++ii) {
            for (g::index sig=g::index(0); sig<gf.mesh3().extent(); ++sig) {
                std::complex<double> v1(1+om()+ii(), 1+sig());
                std::complex<double> v2=1./v1;
                gf(om,ii,sig)=v1;
                gf2(om,ii,sig)=v2;
            }
        }
    }


    gf_type g_plus=gf; g_plus+=gf2;
    gf_type g_minus=gf; g_minus-=gf2;

    const double tol=1E-8;
                    
    for (g::matsubara_index om=g::matsubara_index(0); om<gf.mesh1().extent(); ++om) {
        for (g::momentum_index ii=g::momentum_index(0); ii<gf.mesh2().extent(); ++ii) {
            for (g::index sig=g::index(0); sig<gf.mesh3().extent(); ++sig) {

                std::complex<double> v1(1+om()+ii(), 1+sig());
                std::complex<double> v2=1./v1;
                
                std::complex<double> r1=v1+v2;
                std::complex<double> r2=v1-v2;
                
                ASSERT_NEAR(r1.real(),g_plus(om,ii,sig).real(),tol);
                ASSERT_NEAR(r1.imag(),g_plus(om,ii,sig).imag(),tol);

                ASSERT_NEAR(r2.real(),g_minus(om,ii,sig).real(),tol);
                ASSERT_NEAR(r2.imag(),g_minus(om,ii,sig).imag(),tol);
            }
        }
    }
}

TEST_F(ThreeIndexGFTest,Assign)
{
    namespace g=alps::gf;

    gf_type other_gf_beta(g::matsubara_positive_mesh(2*beta,nfreq),
                          g::momentum_index_mesh(get_data_for_momentum_mesh()),
                          g::index_mesh(nspins));
    
    gf_type other_gf_nfreq(g::matsubara_positive_mesh(beta,nfreq*2),
                          g::momentum_index_mesh(get_data_for_momentum_mesh()),
                          g::index_mesh(nspins));
    
    gf_type other_gf_nspins(g::matsubara_positive_mesh(beta,nfreq*2),
                            g::momentum_index_mesh(get_data_for_momentum_mesh()),
                            g::index_mesh(2*nspins));
    
    const alps::gf::matsubara_index omega(4);
    const alps::gf::momentum_index i(2);
    const alps::gf::index sigma(1);

    const std::complex<double> data(3,4);
    gf(omega,i,sigma)=data;

    gf2=gf;
    EXPECT_EQ(data, gf2(omega,i,sigma));
    
    EXPECT_NO_THROW(other_gf_beta=gf);
    EXPECT_EQ(data, other_gf_beta(omega,i,sigma));
    
    EXPECT_NO_THROW(other_gf_nfreq=gf);
    EXPECT_EQ(data, other_gf_nfreq(omega,i,sigma));

    EXPECT_NO_THROW(other_gf_nspins=gf);
    EXPECT_EQ(data, other_gf_nspins(omega,i,sigma));
}


TEST_F(ThreeIndexGFTest,Operators)
{
    namespace g=alps::gf;

    for (g::matsubara_index om=g::matsubara_index(0); om<gf.mesh1().extent(); ++om) {
        for (g::momentum_index ii=g::momentum_index(0); ii<gf.mesh2().extent(); ++ii) {
            for (g::index sig=g::index(0); sig<gf.mesh3().extent(); ++sig) {
                std::complex<double> v1(1+om()+ii(), 1+sig());
                std::complex<double> v2=1./v1;
                gf(om,ii,sig)=v1;
                gf2(om,ii,sig)=v2;
            }
        }
    }


    gf_type g_plus=gf+gf2;
    gf_type g_minus=gf-gf2;

    const double tol=1E-8;
                    
    for (g::matsubara_index om=g::matsubara_index(0); om<gf.mesh1().extent(); ++om) {
        for (g::momentum_index ii=g::momentum_index(0); ii<gf.mesh2().extent(); ++ii) {
            for (g::index sig=g::index(0); sig<gf.mesh3().extent(); ++sig) {

                std::complex<double> v1(1+om()+ii(), 1+sig());
                std::complex<double> v2=1./v1;
                
                std::complex<double> r1=v1+v2;
                std::complex<double> r2=v1-v2;
                
                ASSERT_NEAR(r1.real(),g_plus(om,ii,sig).real(),tol);
                ASSERT_NEAR(r1.imag(),g_plus(om,ii,sig).imag(),tol);

                ASSERT_NEAR(r2.real(),g_minus(om,ii,sig).real(),tol);
                ASSERT_NEAR(r2.imag(),g_minus(om,ii,sig).imag(),tol);
            }
        }
    }
}

TEST_F(ThreeIndexGFTest,scaling)
{
    alps::gf::momentum_index i; i=2;
    alps::gf::index sigma(1);
    alps::gf::matsubara_index omega; omega=4;

    gf(omega,i,sigma)=std::complex<double>(3,4);
    gf *= 2.;
    std::complex<double> x=gf(omega,i,sigma);
    EXPECT_NEAR(6, x.real(),1.e-10);
    EXPECT_NEAR(8, x.imag(),1.e-10);

    gf_type gf1=gf/2;
    std::complex<double> x1=gf1(omega,i,sigma);
    EXPECT_NEAR(3, x1.real(),1.e-10);
    EXPECT_NEAR(4, x1.imag(),1.e-10);
}

TEST_F(ThreeIndexGFTest,negation)
{
    alps::gf::momentum_index i; i=2;
    alps::gf::index sigma(1);
    alps::gf::matsubara_index omega; omega=4;

    gf(omega,i,sigma)=std::complex<double>(3,4);
    gf_type gf_neg=-gf;

    std::complex<double> x=gf_neg(omega,i,sigma);
    EXPECT_NEAR(-3, x.real(),1.e-10);
    EXPECT_NEAR(-4, x.imag(),1.e-10);
}

TEST_F(ThreeIndexGFTest,print)
{
  std::stringstream gf_stream;
  gf_stream<<gf;

  std::stringstream gf_stream_by_hand;
  gf_stream_by_hand<<alps::gf::matsubara_positive_mesh(beta,nfreq)
      <<alps::gf::momentum_index_mesh(get_data_for_momentum_mesh())
      <<alps::gf::index_mesh(nspins);
  for(int i=0;i<nfreq;++i){
    gf_stream_by_hand<<(2*i+1)*M_PI/beta<<" 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "<<std::endl;
  }
  EXPECT_EQ(gf_stream_by_hand.str(), gf_stream.str());
}

// FIXME: does not test the validity of print output
TEST_F(ThreeIndexGFTest, tailPrint)
{
    namespace g=alps::gf;
    typedef g::two_index_gf<double, g::momentum_index_mesh, g::index_mesh> density_matrix_type;
    density_matrix_type denmat=density_matrix_type(g::momentum_index_mesh(get_data_for_momentum_mesh()),
                                      g::index_mesh(nspins));

    // prepare diagonal matrix
    double U=3.0;
    denmat.initialize();
    for (g::momentum_index i=g::momentum_index(0); i<denmat.mesh1().extent(); ++i) {
        denmat(i,g::index(0))=0.5*U;
        denmat(i,g::index(1))=0.5*U;
    }

    // Attach a tail to the GF
    int order=0;
    
    // FIXME: TODO: gf.set_tail(min_order, max_order, denmat, ...);
    g::omega_k_sigma_gf_with_tail gft(gf);
    gft.set_tail(order, denmat)
    // .set_tail(order+1, other_gf) ....
        ;

    std::ostringstream outs;
    outs << gft.tail(0);
    std::cout << "Output is:\n" << outs.str() << std::endl;
}

TEST_F(ThreeIndexGFTest, DefaultConstructive)
{
    gf_type gf_empty;
    EXPECT_TRUE(gf_empty.is_empty());
    {
        alps::hdf5::archive oar("gf_3i_defconstr.h5","w");
        oar["/gf"] << gf;
    }
    {
        alps::hdf5::archive iar("gf_3i_defconstr.h5");
        iar["/gf"] >> gf_empty;
    }
    EXPECT_FALSE(gf_empty.is_empty());
}

TEST_F(ThreeIndexGFTest, DefaultConstructiveAssign)
{
    gf_type gf_empty;
    gf_type gf_empty2;
    gf_type gf_empty3 = gf;
    EXPECT_TRUE(gf_empty.is_empty());
    EXPECT_TRUE(gf_empty2.is_empty());
    EXPECT_FALSE(gf_empty3.is_empty());
    EXPECT_FALSE(gf_empty3.data().size()==0);
    EXPECT_NO_THROW(gf_empty = gf_empty2);
    EXPECT_NO_THROW(gf_empty3 = gf_empty);
    EXPECT_TRUE(gf_empty.is_empty());
    EXPECT_TRUE(gf_empty3.is_empty());
    EXPECT_TRUE(gf_empty3.data().size()==0);
}

#ifndef NDEBUG
TEST_F(ThreeIndexGFTest, DefaultConstructiveAccess) {
    gf_type gf_empty;
    EXPECT_ANY_THROW(gf_empty.norm());
    EXPECT_ANY_THROW(gf_empty*1.0);
    EXPECT_ANY_THROW(-gf_empty);
}
#endif
