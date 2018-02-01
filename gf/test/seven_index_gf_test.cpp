/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include "gtest/gtest.h"
#include <alps/gf/gf.hpp>
#include "gf_test.hpp"

class SevenIndexGFTest : public ::testing::Test
{
  public:
    const int nindex;

    typedef alps::gf::seven_index_gf<std::complex<double>, alps::gf::index_mesh,alps::gf::index_mesh, alps::gf::index_mesh,alps::gf::index_mesh,alps::gf::index_mesh,alps::gf::index_mesh,alps::gf::index_mesh>  gf_type;
    gf_type gf;
    gf_type gf2;

    SevenIndexGFTest():nindex(7),
             gf(alps::gf::index_mesh(nindex),
                alps::gf::index_mesh(nindex),
                alps::gf::index_mesh(nindex),
                alps::gf::index_mesh(nindex),
                alps::gf::index_mesh(nindex),
                alps::gf::index_mesh(nindex),
                alps::gf::index_mesh(nindex)),
             gf2(gf) {}
};

TEST_F(SevenIndexGFTest,access)
{
    alps::gf::index i1(1),i2(3),i3(4),i4(2),i5(6),i6(0),i7(6);

    gf(i1,i2,i3,i4,i5,i6,i7)=std::complex<double>(3,4);
    std::complex<double> x=gf(i1,i2,i3,i4,i5,i6,i7);
    EXPECT_EQ(3, x.real());
    EXPECT_EQ(4, x.imag());
}

TEST_F(SevenIndexGFTest,init)
{
    alps::gf::index i1(1),i2(3),i3(4),i4(2),i5(6),i6(0),i7(6);

    gf.initialize();
    std::complex<double> x=gf(i1,i2,i3,i4,i5,i6,i7);
    EXPECT_EQ(0, x.real());
    EXPECT_EQ(0, x.imag());
}

TEST_F(SevenIndexGFTest,saveload)
{
    namespace g=alps::gf;
    alps::gf::index i1(1),i2(3),i3(4),i4(2),i5(6),i6(0),i7(6);
    {
        alps::hdf5::archive oar("gf_7i_saveload.h5","w");
        gf(i1,i2,i3,i4,i5,i6,i7)=std::complex<double>(7., 3.);
        gf.save(oar,"/gf");
    }
    {
        alps::hdf5::archive iar("gf_7i_saveload.h5");
        gf2.load(iar,"/gf");
    }
    EXPECT_EQ(7, gf2(i1,i2,i3,i4,i5,i6,i7).real());
    EXPECT_EQ(3, gf2(i1,i2,i3,i4,i5,i6,i7).imag());
    {
        alps::hdf5::archive oar("gf_7i_saveload.h5","rw");
        oar["/gf/version/major"]<<7;
        EXPECT_THROW(gf2.load(oar,"/gf"),std::runtime_error);
    }
}

TEST_F(SevenIndexGFTest,saveloadstream)
{
    namespace g=alps::gf;
    alps::gf::index i1(1),i2(3),i3(4),i4(2),i5(6),i6(0),i7(6);
    {
        alps::hdf5::archive oar("gf_7i_stream.h5","w");
        gf(i1,i2,i3,i4,i5,i6,i7)=std::complex<double>(7., 3.);
        oar["/gf"] << gf;
    }
    {
        alps::hdf5::archive iar("gf_7i_stream.h5");
        iar["/gf"] >> gf2;
    }
    EXPECT_EQ(7, gf2(i1,i2,i3,i4,i5,i6,i7).real());
    EXPECT_EQ(3, gf2(i1,i2,i3,i4,i5,i6,i7).imag());
    {
        alps::hdf5::archive oar("gf_7i_stream.h5","rw");
        oar["/gf/version/major"]<<7;
        EXPECT_THROW(oar["/gf"] >> gf2, std::runtime_error);
    }
}



TEST_F(SevenIndexGFTest,EqOperators)
{
    namespace g=alps::gf;
    alps::gf::index i1(1),i2(3),i3(4),i4(2),i5(6),i6(0),i7(6);

    for (g::index i1=g::index(0); i1<gf.mesh1().extent(); ++i1) {
      for (g::index i2=g::index(0); i2<gf.mesh2().extent(); ++i2) {
        for (g::index i3=g::index(0); i3<gf.mesh3().extent(); ++i3) {
          for (g::index i4=g::index(0); i4<gf.mesh4().extent(); ++i4) {
            for (g::index i5=g::index(0); i5<gf.mesh5().extent(); ++i5) {
              for (g::index i6=g::index(0); i6<gf.mesh6().extent(); ++i6) {
                for (g::index i7=g::index(0); i7<gf.mesh7().extent(); ++i7) {
                std::complex<double> v1(1+i1()+2*i2()+3*i3()+4*i4(), 1+i5()+2*i6()+3*i7());
                std::complex<double> v2=1./v1;
                gf(i1,i2,i3,i4,i5,i6,i7)=v1;
                gf2(i1,i2,i3,i4,i5,i6,i7)=v2;
                }
              }
            }
          }
        }
      }
    }

    gf_type g_plus=gf; g_plus+=gf2;
    gf_type g_minus=gf; g_minus-=gf2;

    const double tol=1E-8;
                    
    for (g::index i1=g::index(0); i1<gf.mesh1().extent(); ++i1) {
      for (g::index i2=g::index(0); i2<gf.mesh2().extent(); ++i2) {
        for (g::index i3=g::index(0); i3<gf.mesh3().extent(); ++i3) {
          for (g::index i4=g::index(0); i4<gf.mesh4().extent(); ++i4) {
            for (g::index i5=g::index(0); i5<gf.mesh5().extent(); ++i5) {
              for (g::index i6=g::index(0); i6<gf.mesh6().extent(); ++i6) {
                for (g::index i7=g::index(0); i7<gf.mesh7().extent(); ++i7) {
                  std::complex<double> v1(1+i1()+2*i2()+3*i3()+4*i4(), 1+i5()+2*i6()+3*i7());
                  std::complex<double> v2=1./v1;
                
                std::complex<double> r1=v1+v2;
                std::complex<double> r2=v1-v2;
                
                ASSERT_NEAR(r1.real(),g_plus(i1,i2,i3,i4,i5,i6,i7).real(),tol);
                ASSERT_NEAR(r1.imag(),g_plus(i1,i2,i3,i4,i5,i6,i7).imag(),tol);

                ASSERT_NEAR(r2.real(),g_minus(i1,i2,i3,i4,i5,i6,i7).real(),tol);
                ASSERT_NEAR(r2.imag(),g_minus(i1,i2,i3,i4,i5,i6,i7).imag(),tol);
                }
              }
            }
          }
        }
      }
    }
}

TEST_F(SevenIndexGFTest,Assign)
{
    namespace g=alps::gf;
    alps::gf::index i1(1),i2(3),i3(4),i4(2),i5(6),i6(0),i7(6);

    const std::complex<double> data(3,4);
    gf(i1,i2,i3,i4,i5,i6,i7)=data;

    gf2=gf;
    EXPECT_EQ(data, gf2(i1,i2,i3,i4,i5,i6,i7));
    
}


TEST_F(SevenIndexGFTest,Operators)
{
    namespace g=alps::gf;

    namespace g=alps::gf;
    alps::gf::index i1(1),i2(3),i3(4),i4(2),i5(6),i6(0),i7(6);

    for (g::index i1=g::index(0); i1<gf.mesh1().extent(); ++i1) {
      for (g::index i2=g::index(0); i2<gf.mesh2().extent(); ++i2) {
        for (g::index i3=g::index(0); i3<gf.mesh3().extent(); ++i3) {
          for (g::index i4=g::index(0); i4<gf.mesh4().extent(); ++i4) {
            for (g::index i5=g::index(0); i5<gf.mesh5().extent(); ++i5) {
              for (g::index i6=g::index(0); i6<gf.mesh6().extent(); ++i6) {
                for (g::index i7=g::index(0); i7<gf.mesh7().extent(); ++i7) {
                std::complex<double> v1(1+i1()+2*i2()+3*i3()+4*i4(), 1+i5()+2*i6()+3*i7());
                std::complex<double> v2=1./v1;
                gf(i1,i2,i3,i4,i5,i6,i7)=v1;
                gf2(i1,i2,i3,i4,i5,i6,i7)=v2;
                }
              }
            }
          }
        }
      }
    }


    gf_type g_plus=gf+gf2;
    gf_type g_minus=gf-gf2;

    const double tol=1E-8;
                    
    for (g::index i1=g::index(0); i1<gf.mesh1().extent(); ++i1) {
      for (g::index i2=g::index(0); i2<gf.mesh2().extent(); ++i2) {
        for (g::index i3=g::index(0); i3<gf.mesh3().extent(); ++i3) {
          for (g::index i4=g::index(0); i4<gf.mesh4().extent(); ++i4) {
            for (g::index i5=g::index(0); i5<gf.mesh5().extent(); ++i5) {
              for (g::index i6=g::index(0); i6<gf.mesh6().extent(); ++i6) {
                for (g::index i7=g::index(0); i7<gf.mesh7().extent(); ++i7) {
                  std::complex<double> v1(1+i1()+2*i2()+3*i3()+4*i4(), 1+i5()+2*i6()+3*i7());
                  std::complex<double> v2=1./v1;
                
                std::complex<double> r1=v1+v2;
                std::complex<double> r2=v1-v2;
                
                ASSERT_NEAR(r1.real(),g_plus(i1,i2,i3,i4,i5,i6,i7).real(),tol);
                ASSERT_NEAR(r1.imag(),g_plus(i1,i2,i3,i4,i5,i6,i7).imag(),tol);

                ASSERT_NEAR(r2.real(),g_minus(i1,i2,i3,i4,i5,i6,i7).real(),tol);
                ASSERT_NEAR(r2.imag(),g_minus(i1,i2,i3,i4,i5,i6,i7).imag(),tol);
                }
              }
            }
          }
        }
      }
    }
}
