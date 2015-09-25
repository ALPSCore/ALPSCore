#include "gtest/gtest.h"
#include "alps/gf/gf.hpp"
#include "alps/gf/tail.hpp"


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

TEST_F(TwoIndexGFTest,saveload)
{
    namespace g=alps::gf;
    {
        alps::hdf5::archive oar("gf.h5","w");
        gf(g::matsubara_index(4), g::index(1))=std::complex<double>(7., 3.);
        gf.save(oar,"/gf");
    }
    {
        alps::hdf5::archive iar("gf.h5");
        gf2.load(iar,"/gf");
    }
    EXPECT_EQ(7, gf2(g::matsubara_index(4), g::index(1)).real());
    EXPECT_EQ(3, gf2(g::matsubara_index(4), g::index(1)).imag());
    {
        alps::hdf5::archive oar("gf.h5","rw");
        oar["/gf/version/major"]<<7;
        EXPECT_THROW(gf2.load(oar,"/gf"),std::runtime_error);
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
    EXPECT_EQ((g::omega_sigma_gf_with_tail::TAIL_NOT_SET+0),gft.min_tail_order());
    EXPECT_EQ((g::omega_sigma_gf_with_tail::TAIL_NOT_SET+0),gft.max_tail_order());

    gft.set_tail(order, denmat);

    EXPECT_EQ(0,gft.min_tail_order());
    EXPECT_EQ(0,gft.max_tail_order());
    EXPECT_EQ(0,(denmat-gft.tail(0)).norm());
    {
        alps::hdf5::archive oar("gft.h5","w");
        gft(g::matsubara_index(4),g::index(1))=std::complex<double>(7., 3.);
        gft.save(oar,"/gft");
    }
    {
        alps::hdf5::archive iar("gft.h5");

        gft2.load(iar,"/gft");
    }
    EXPECT_EQ(gft2.tail().size(), gft.tail().size()) << "Tail size mismatch";
    EXPECT_NEAR(0, (gft.tail(0)-gft2.tail(0)).norm(), 1E-8)<<"Tail loaded differs from tail stored";

    EXPECT_EQ(7, gft2(g::matsubara_index(4), g::index(1)).real()) << "GF real part mismatch";
    EXPECT_EQ(3, gft2(g::matsubara_index(4), g::index(1)).imag()) << "GF imag part mismatch";

}

