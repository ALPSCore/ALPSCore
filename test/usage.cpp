#include "gtest/gtest.h"
#include "alps/gf/gf.hpp"
#include "alps/gf/tail.hpp"

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

TEST(Mesh,CompareMatsubara) {
    alps::gf::matsubara_mesh<alps::gf::mesh::POSITIVE_NEGATIVE> mesh1(5.0, 20);
    alps::gf::matsubara_mesh<alps::gf::mesh::POSITIVE_NEGATIVE> mesh2(5.0, 20);
    alps::gf::matsubara_mesh<alps::gf::mesh::POSITIVE_NEGATIVE> mesh3(4.0, 20);

    EXPECT_TRUE(mesh1==mesh2);
    EXPECT_TRUE(mesh1!=mesh3);
    
    EXPECT_FALSE(mesh1==mesh3);
    EXPECT_FALSE(mesh1!=mesh2);
    
}

TEST(Mesh,CompareITime) {
    alps::gf::itime_mesh mesh1(5.0, 20);
    alps::gf::itime_mesh mesh2(5.0, 20);
    alps::gf::itime_mesh mesh3(4.0, 20);

    EXPECT_TRUE(mesh1==mesh2);
    EXPECT_TRUE(mesh1!=mesh3);
    
    EXPECT_FALSE(mesh1==mesh3);
    EXPECT_FALSE(mesh1!=mesh2);
}

TEST(Mesh,CompareMomentum) {
    alps::gf::momentum_index_mesh::container_type points1(boost::extents[20][3]);
    alps::gf::momentum_index_mesh::container_type points2(boost::extents[20][3]);
    alps::gf::momentum_index_mesh::container_type points3(boost::extents[20][3]);
    alps::gf::momentum_index_mesh::container_type points4(boost::extents[3][20]);
    for (int i=0; i<points1.num_elements(); ++i) {
        *(points1.origin()+i)=i;
        *(points2.origin()+i)=i;
        *(points3.origin()+i)=i+1;
        *(points4.origin()+i)=i;
    }
    
    alps::gf::momentum_index_mesh mesh1(points1);
    alps::gf::momentum_index_mesh mesh2(points2);
    alps::gf::momentum_index_mesh mesh3(points3);
    alps::gf::momentum_index_mesh mesh4(points4);

    EXPECT_TRUE(mesh1==mesh2);
    EXPECT_TRUE(mesh1!=mesh3);
    EXPECT_TRUE(mesh1!=mesh4);
    
    EXPECT_FALSE(mesh1==mesh3);
    EXPECT_FALSE(mesh1!=mesh2);
    EXPECT_FALSE(mesh1==mesh4);
}


TEST(Mesh,CompareRealSpace) {
    alps::gf::real_space_index_mesh::container_type points1(boost::extents[20][3]);
    alps::gf::real_space_index_mesh::container_type points2(boost::extents[20][3]);
    alps::gf::real_space_index_mesh::container_type points3(boost::extents[20][3]);
    alps::gf::real_space_index_mesh::container_type points4(boost::extents[3][20]);
    for (int i=0; i<points1.num_elements(); ++i) {
        *(points1.origin()+i)=i;
        *(points2.origin()+i)=i;
        *(points3.origin()+i)=i+1;
        *(points4.origin()+i)=i;
    }
    
    alps::gf::real_space_index_mesh mesh1(points1);
    alps::gf::real_space_index_mesh mesh2(points2);
    alps::gf::real_space_index_mesh mesh3(points3);
    alps::gf::real_space_index_mesh mesh4(points4);

    EXPECT_TRUE(mesh1==mesh2);
    EXPECT_TRUE(mesh1!=mesh3);
    EXPECT_TRUE(mesh1!=mesh4);
    
    EXPECT_FALSE(mesh1==mesh3);
    EXPECT_FALSE(mesh1!=mesh2);
    EXPECT_FALSE(mesh1==mesh4);
}

TEST(Mesh,CompareIndex) {
    alps::gf::index_mesh mesh1(20);
    alps::gf::index_mesh mesh2(20);
    alps::gf::index_mesh mesh3(19);

    EXPECT_TRUE(mesh1==mesh2);
    EXPECT_TRUE(mesh1!=mesh3);
    
    EXPECT_FALSE(mesh1==mesh3);
    EXPECT_FALSE(mesh1!=mesh2);
}



class ThreeIndexTestGF : public ::testing::Test
{
  public:
    const double beta;
    const int nsites;
    const int nfreq ;
    const int nspins;
    typedef alps::gf::omega_k_sigma_gf gf_type;
    gf_type gf;
    gf_type gf2;

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

TEST_F(ThreeIndexTestGF, tail)
{
    namespace g=alps::gf;
    typedef g::two_index_gf<double, g::momentum_index_mesh, g::index_mesh> density_matrix_type;
    density_matrix_type denmat=density_matrix_type(g::momentum_index_mesh(get_data_for_mesh()),
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
}

TEST_F(ThreeIndexTestGF,EqOperators)
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

TEST_F(ThreeIndexTestGF,Operators)
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

