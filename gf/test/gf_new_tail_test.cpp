//
// Created by iskakoff on 08/12/17.
//

#include <gtest/gtest.h>

#include <alps/gf_new/gf_tail.h>
#include <alps/gf/mesh.hpp>


namespace g=alps::gf;

class GreensFunctionTailTest : public ::testing::Test
{
public:
  const double beta;
  const int nsites;
  const int nfreq ;
  const int nspins;
  typedef g::greenf<std::complex<double> , g::matsubara_positive_mesh, g::index_mesh> omega_sigma_gf;
  typedef g::greenf<std::complex<double>, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh> head;
  typedef g::greenf<double, alps::gf::index_mesh> tail;
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
  typedef g::greenf<double, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh, alps::gf::itime_mesh, alps::gf::legendre_mesh> head;
  typedef g::greenf<double, alps::gf::index_mesh, alps::gf::itime_mesh, alps::gf::legendre_mesh> tail;
  g::gf_tail<head, tail> g;
}

TEST_F(GreensFunctionTailTest, AssignTest){
  alps::gf::matsubara_positive_mesh x(100, 10);
  alps::gf::index_mesh y(10);
  alps::gf::itime_mesh z(100, 10);
  typedef g::greenf<double, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh, alps::gf::itime_mesh> head;
  typedef g::greenf<double, alps::gf::index_mesh, alps::gf::itime_mesh> tail;
  g::greenf<double, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh, alps::gf::itime_mesh> g(x,y,z);
  g::gf_tail<head, tail> g2(g);
  ASSERT_EQ(g.data(), g2.data());
}

TEST_F(GreensFunctionTailTest, SetTail){
  alps::gf::index_mesh y(nspins);
  g::gf_tail<head, tail> g2 (gf);
  tail tail1(y);
  tail1(alps::gf::index_mesh::index_type(0)) = 10.0;
  g2.set_tail(0, tail1);
  auto tail2 = g2.tail(0);
  ASSERT_EQ(tail1, tail2);
}

TEST_F(GreensFunctionTailTest, TailSaveLoad)
{
  typedef g::greenf<std::complex<double>, g::matsubara_mesh<g::mesh::POSITIVE_ONLY>, g::index_mesh> omega_sigma_gf;
  typedef g::greenf<std::complex<double>, g::index_mesh> one_index_gf;
  typedef g::gf_tail<omega_sigma_gf, g::greenf<double, g::index_mesh> > omega_sigma_gf_with_tail;
  typedef g::greenf<double, g::index_mesh> density_matrix_type;
  density_matrix_type denmat = density_matrix_type(g::index_mesh(nspins));

  omega_sigma_gf gf(g::matsubara_positive_mesh(beta,nfreq), alps::gf::index_mesh(nspins));

  // prepare diagonal matrix
  double U=3.0;
  denmat.initialize();
  denmat(g::index(0))=0.5*U;
  denmat(g::index(1))=0.5*U;

  // Attach a tail to the GF
  int order=0;

  omega_sigma_gf_with_tail gft(gf);
  omega_sigma_gf_with_tail gft2(gft);
  EXPECT_EQ(g::TAIL_NOT_SET,gft.min_tail_order());
  EXPECT_EQ(g::TAIL_NOT_SET,gft.max_tail_order());

  gft.set_tail(order, denmat);

  EXPECT_EQ(0,gft.min_tail_order());
  EXPECT_EQ(0,gft.max_tail_order());
  EXPECT_EQ(0,(denmat - gft.tail(0)).norm());
  {
    alps::hdf5::archive oar("gf_2i_tailsaveload.h5","w");
    gft(g::matsubara_index(4),g::index(1))=std::complex<double>(7., 3.);
    oar["/gft"] << gft;
  }
  {
    alps::hdf5::archive iar("gf_2i_tailsaveload.h5");

    iar["/gft"] >> gft2;
  }
  EXPECT_EQ(gft2.tail().size(), gft.tail().size()) << "Tail size mismatch";
  EXPECT_NEAR(0, (gft.tail(0)-gft2.tail(0)).norm(), 1E-8)<<"Tail loaded differs from tail stored";

  EXPECT_EQ(7, gft2(g::matsubara_index(4), g::index(1)).real()) << "GF real part mismatch";
  EXPECT_EQ(3, gft2(g::matsubara_index(4), g::index(1)).imag()) << "GF imag part mismatch";

}
