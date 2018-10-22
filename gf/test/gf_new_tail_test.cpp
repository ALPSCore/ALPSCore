/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <gtest/gtest.h>

#include <alps/gf/tail.hpp>
#include <alps/gf/mesh.hpp>


namespace gfns=alps::gf;

class GreensFunctionTailTest : public ::testing::Test
{
public:
  const double beta;
  const int nsites;
  const int nfreq ;
  const int nspins;
  typedef gfns::greenf<std::complex<double> , gfns::matsubara_positive_mesh, gfns::index_mesh> omega_sigma_gf;
  typedef gfns::greenf<std::complex<double>, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh> head;
  typedef gfns::greenf<double, alps::gf::index_mesh> tail;
  omega_sigma_gf gf;
  omega_sigma_gf gf2;
  typedef alps::gf::matsubara_mesh<alps::gf::mesh::POSITIVE_ONLY> matsubara_mesh;
  typedef omega_sigma_gf gf_type;

  GreensFunctionTailTest():beta(10), nsites(4), nfreq(10), nspins(2),
                   gf(matsubara_mesh(beta,nfreq),
                      alps::gf::index_mesh(nspins)),
                   gf2(gf) {}
};


TEST_F(GreensFunctionTailTest, InitializationTest){
  typedef gfns::greenf<double, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh, alps::gf::itime_mesh, alps::gf::legendre_mesh> head;
  typedef gfns::greenf<double, alps::gf::index_mesh, alps::gf::itime_mesh, alps::gf::legendre_mesh> tail;
  gfns::gf_tail<head, tail> g;
}

TEST_F(GreensFunctionTailTest, AssignTest){
  alps::gf::matsubara_positive_mesh x(100, 10);
  alps::gf::index_mesh y(10);
  alps::gf::itime_mesh z(100, 10);
  typedef gfns::greenf<double, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh, alps::gf::itime_mesh> head;
  typedef gfns::greenf<double, alps::gf::index_mesh, alps::gf::itime_mesh> tail;
  gfns::greenf<double, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh, alps::gf::itime_mesh> g(x,y,z);
  g(alps::gf::matsubara_positive_mesh::index_type(0), alps::gf::index_mesh::index_type(0), alps::gf::itime_mesh::index_type(0)) = 10.0;
  gfns::gf_tail<head, tail> g2(g);
  gfns::gf_tail<head, tail> g3(g);
  ASSERT_EQ(g.data(), g2.data());
  ASSERT_EQ(g2, g3);
  tail t1(y,z);
  tail t2(y,z);
  t1(alps::gf::index_mesh::index_type(0), alps::gf::itime_mesh::index_type(3)) = 1.0;
  t2(alps::gf::index_mesh::index_type(0), alps::gf::itime_mesh::index_type(3)) = 1.0;
  g2.set_tail(0, t1);
  ASSERT_NE(g2, g3);
  g3.set_tail(0, t2);
  ASSERT_EQ(g2, g3);
}

TEST_F(GreensFunctionTailTest, SetTail){
  alps::gf::index_mesh y(nspins);
  gfns::gf_tail<head, tail> g2 (gf);
  tail tail1(y);
  tail1(alps::gf::index_mesh::index_type(0)) = 10.0;
  g2.set_tail(0, tail1);
  auto tail2 = g2.tail(0);
  ASSERT_EQ(tail1, tail2);
}

TEST_F(GreensFunctionTailTest, TailSaveLoad)
{
  typedef gfns::greenf<std::complex<double>, gfns::matsubara_mesh<gfns::mesh::POSITIVE_ONLY>, gfns::index_mesh> omega_sigma_gf;
  typedef gfns::gf_tail<omega_sigma_gf, gfns::greenf<double, gfns::index_mesh> > omega_sigma_gf_with_tail;
  typedef gfns::greenf<double, gfns::index_mesh> density_matrix_type;
  density_matrix_type denmat = density_matrix_type(gfns::index_mesh(nspins));

  omega_sigma_gf gf(gfns::matsubara_positive_mesh(beta,nfreq), alps::gf::index_mesh(nspins));

  // prepare diagonal matrix
  double U=3.0;
  denmat.initialize();
  denmat(gfns::index(0))=0.5*U;
  denmat(gfns::index(1))=0.5*U;

  // Attach a tail to the GF
  int order=0;

  omega_sigma_gf_with_tail gft(gf);
  omega_sigma_gf_with_tail gft2(gft);
  EXPECT_EQ(gfns::TAIL_NOT_SET,gft.min_tail_order());
  EXPECT_EQ(gfns::TAIL_NOT_SET,gft.max_tail_order());

  gft.set_tail(order, denmat);

  EXPECT_EQ(0,gft.min_tail_order());
  EXPECT_EQ(0,gft.max_tail_order());
  EXPECT_EQ(0,(denmat - gft.tail(0)).norm());
  {
    alps::hdf5::archive oar("gf_2i_tailsaveload.h5","w");
    gft(gfns::matsubara_index(4),gfns::index(1))=std::complex<double>(7., 3.);
    oar["/gft"] << gft;
  }
  {
    alps::hdf5::archive iar("gf_2i_tailsaveload.h5");

    iar["/gft"] >> gft2;
  }
  EXPECT_EQ(gft2.tail().size(), gft.tail().size()) << "Tail size mismatch";
  EXPECT_NEAR(0, (gft.tail(0)-gft2.tail(0)).norm(), 1E-8)<<"Tail loaded differs from tail stored";

  EXPECT_EQ(7, gft2(gfns::matsubara_index(4), gfns::index(1)).real()) << "GF real part mismatch";
  EXPECT_EQ(3, gft2(gfns::matsubara_index(4), gfns::index(1)).imag()) << "GF imag part mismatch";

}
