#include "gtest/gtest.h"
#include "alps/gf/gf.hpp"
#include "alps/gf/tail.hpp"


class OneIndexGFTest : public ::testing::Test
{
  public:
    const double beta;
    const int nfreq ;
    alps::gf::omega_gf gf;
    alps::gf::omega_gf gf2;
    typedef alps::gf::matsubara_mesh<alps::gf::mesh::POSITIVE_ONLY> matsubara_mesh;

    OneIndexGFTest():beta(10), nfreq(10),
             gf(matsubara_mesh(beta,nfreq)),
             gf2(gf) {}
};
    

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
        alps::hdf5::archive oar("gf.h5","w");
        gf(g::matsubara_index(4))=std::complex<double>(7., 3.);
        gf.save(oar,"/gf");
    }
    {
        alps::hdf5::archive iar("gf.h5");
        gf2.load(iar,"/gf");
    }
    EXPECT_EQ(7, gf2(g::matsubara_index(4)).real());
    EXPECT_EQ(3, gf2(g::matsubara_index(4)).imag());
    {
        alps::hdf5::archive oar("gf.h5","rw");
        oar["/gf/version/major"]<<7;
        EXPECT_THROW(gf2.load(oar,"/gf"),std::runtime_error);
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
    gf_stream_by_hand<<(2*i+1)*M_PI/beta<<" 0 0"<<std::endl;
  }
  EXPECT_EQ(gf_stream_by_hand.str(), gf_stream.str());
}
