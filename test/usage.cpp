#include "gtest/gtest.h"
#include "alps/gf/gf.hpp"

/// This generates some "outside" data to fill the mesh: 4 2-d points
alps::gf::momentum_index_mesh::container_type get_data_for_mesh()
{
    alps::gf::momentum_index_mesh::container_type points(boost::extents[4][2]);
    points[0][0]=0; points[0][1]=0; 
    points[1][0]=M_PI; points[1][1]=M_PI;
    points[2][0]=M_PI; points[2][1]=0; 
    points[3][0]=0; points[3][1]=M_PI;

    return points;
}


class TestGF : public ::testing::Test
{
  public:
    const double beta;
    const int nsites;
    const int nfreq ;
    const int nspins;
    alps::gf::matsubara_gf gf;
    alps::gf::matsubara_gf gf2;
    typedef alps::gf::matsubara_mesh<alps::gf::mesh::POSITIVE_ONLY> matsubara_mesh;

    TestGF():beta(10), nsites(4), nfreq(10), nspins(2),
             gf(matsubara_mesh(beta,nfreq),
                alps::gf::momentum_index_mesh(get_data_for_mesh()),
                alps::gf::momentum_index_mesh(get_data_for_mesh()),
                alps::gf::index_mesh(nspins)),
             gf2(gf) {}
};
    

TEST_F(TestGF,access)
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

TEST_F(TestGF,init)
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

TEST_F(TestGF,saveload)
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


class ItimeTestGF : public ::testing::Test
{
  public:
    const double beta;
    const int nsites;
    const int ntau;
    const int nspins;
    alps::gf::itime_gf gf;
    alps::gf::itime_gf gf2;

    ItimeTestGF():beta(10), nsites(4), ntau(10), nspins(2),
             gf(alps::gf::itime_mesh(beta,ntau),
                alps::gf::momentum_index_mesh(get_data_for_mesh()),
                alps::gf::momentum_index_mesh(get_data_for_mesh()),
                alps::gf::index_mesh(nspins)),
             gf2(gf) {}
};
    

TEST_F(ItimeTestGF,access)
{
    alps::gf::itime_index tau; tau=4;
    alps::gf::momentum_index i; i=2;
    alps::gf::momentum_index j=alps::gf::momentum_index(3);
    alps::gf::index sigma(1);

    gf(tau, i,j,sigma)=7.;
    double x=gf(tau,i,j,sigma);
    EXPECT_EQ(7, x);
}

TEST_F(ItimeTestGF,init)
{
    alps::gf::itime_index tau; tau=4;
    alps::gf::momentum_index i; i=2;
    alps::gf::momentum_index j=alps::gf::momentum_index(3);
    alps::gf::index sigma(1);

    gf.initialize();
    double x=gf(tau,i,j,sigma);
    EXPECT_EQ(0, x);
}

TEST_F(ItimeTestGF,saveload)
{
    namespace g=alps::gf;
    {
        alps::hdf5::archive oar("gf.h5","w");
        gf(g::itime_index(4),g::momentum_index(3), g::momentum_index(2), g::index(1))=6.;
        gf.save(oar,"/gf");
    }
    {
        alps::hdf5::archive iar("gf.h5");
        gf2.load(iar,"/gf");
    }
    EXPECT_EQ(6., gf2(g::itime_index(4),g::momentum_index(3), g::momentum_index(2), g::index(1)));
    {
        alps::hdf5::archive oar("gf.h5","rw");
        oar["/gf/version/major"]<<7;
        EXPECT_THROW(gf2.load(oar,"/gf"),std::runtime_error);
    }
    EXPECT_EQ(6., gf2(g::itime_index(4),g::momentum_index(3), g::momentum_index(2), g::index(1)));
    
    
    //boost::filesystem::remove("g5.h5");
}

TEST(Index, UnaryAndComparisonOperators){
    alps::gf::matsubara_mesh<alps::gf::mesh::POSITIVE_NEGATIVE> mesh(5.0, 20);
    alps::gf::matsubara_index omega; omega=5;
   alps::gf::matsubara_index omega5=omega++;
   alps::gf::matsubara_index omega7=++omega;
   EXPECT_EQ(7, omega);
   EXPECT_EQ(5, omega5);
   EXPECT_EQ(7, omega7);
   
   omega+=1;
   EXPECT_EQ(8, omega);
   omega-=3;
   EXPECT_EQ(5, omega);
   --omega;
   EXPECT_EQ(4, omega);
   EXPECT_LT(omega,5);
   EXPECT_LE(omega,4);
   EXPECT_GT(omega,3);
   EXPECT_GE(omega,4);

   EXPECT_GT(5,omega);
   EXPECT_GE(4,omega);
   EXPECT_LT(3,omega);
   EXPECT_LE(4,omega);
}

TEST(Index, BinaryOperators){
    alps::gf::matsubara_mesh<alps::gf::mesh::POSITIVE_NEGATIVE> mesh(5.0, 20);
    alps::gf::matsubara_index omega; omega=5;
   alps::gf::matsubara_index omegaprime=omega+11;
   alps::gf::matsubara_index omegaprime1=11+omega;
   alps::gf::matsubara_index omegaprime2=omega-11;

   EXPECT_EQ(5, omega);

   EXPECT_EQ(16, omegaprime);
   EXPECT_EQ(16, omegaprime1);

   EXPECT_EQ(-6, omegaprime2);
}

// TEST(Index, SizeofTest) {
//     alps::gf::matsubara_index omega(5);
//     std::cout << "sizeof(omega)=" << sizeof(omega) << std::endl;
// }

class ThreeIndexTestGF : public ::testing::Test
{
  public:
    const double beta;
    const int nsites;
    const int nfreq ;
    const int nspins;
    alps::gf::omega_k_sigma_gf gf;
    alps::gf::omega_k_sigma_gf gf2;

    ThreeIndexTestGF():beta(10), nsites(4), nfreq(10), nspins(2),
             gf(alps::gf::matsubara_positive_mesh(beta,nfreq),
                alps::gf::momentum_index_mesh(get_data_for_mesh()),
                alps::gf::index_mesh(nspins)),
             gf2(gf) {}
};
TEST_F(ThreeIndexTestGF,access)
{
    alps::gf::matsubara_index omega; omega=4;
    alps::gf::momentum_index i; i=2;
    alps::gf::index sigma(1);

    gf(omega, i,sigma)=std::complex<double>(3,4);
    std::complex<double> x=gf(omega,i,sigma);
    EXPECT_EQ(3, x.real());
    EXPECT_EQ(4, x.imag());
}

TEST_F(ThreeIndexTestGF,init)
{
    alps::gf::matsubara_index omega; omega=4;
    alps::gf::momentum_index i; i=2;
    alps::gf::index sigma(1);

    gf.initialize();
    std::complex<double> x=gf(omega,i,sigma);
    EXPECT_EQ(0, x.real());
    EXPECT_EQ(0, x.imag());
}

TEST_F(ThreeIndexTestGF,saveload)
{
    namespace g=alps::gf;
    {
        alps::hdf5::archive oar("gf.h5","w");
        gf(g::matsubara_index(4),g::momentum_index(3), g::index(1))=std::complex<double>(7., 3.);
        gf.save(oar,"/gf");
    }
    {
        alps::hdf5::archive iar("gf.h5");
        gf2.load(iar,"/gf");
    }
    EXPECT_EQ(7, gf2(g::matsubara_index(4),g::momentum_index(3), g::index(1)).real());
    EXPECT_EQ(3, gf2(g::matsubara_index(4),g::momentum_index(3), g::index(1)).imag());
    {
        alps::hdf5::archive oar("gf.h5","rw");
        oar["/gf/version/major"]<<7;
        EXPECT_THROW(gf2.load(oar,"/gf"),std::runtime_error);
    }
    EXPECT_EQ(7, gf2(g::matsubara_index(4),g::momentum_index(3), g::index(1)).real());
    EXPECT_EQ(3, gf2(g::matsubara_index(4),g::momentum_index(3), g::index(1)).imag());

}

