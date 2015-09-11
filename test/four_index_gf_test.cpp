#include "gtest/gtest.h"
#include "alps/gf/gf.hpp"
#include "alps/gf/tail.hpp"

/// This generates some "outside" data to fill the mesh: 4 2-d points
inline alps::gf::momentum_index_mesh::container_type get_data_for_momentum_mesh()
{
    alps::gf::momentum_index_mesh::container_type points(boost::extents[4][2]);
    points[0][0]=0; points[0][1]=0; 
    points[1][0]=M_PI; points[1][1]=M_PI;
    points[2][0]=M_PI; points[2][1]=0; 
    points[3][0]=0; points[3][1]=M_PI;

    return points;
}


class FourIndexGFTest : public ::testing::Test
{
  public:
    const double beta;
    const int nsites;
    const int nfreq ;
    const int nspins;
    alps::gf::matsubara_gf gf;
    alps::gf::matsubara_gf gf2;
    typedef alps::gf::matsubara_mesh<alps::gf::mesh::POSITIVE_ONLY> matsubara_mesh;

    FourIndexGFTest():beta(10), nsites(4), nfreq(10), nspins(2),
             gf(matsubara_mesh(beta,nfreq),
                alps::gf::momentum_index_mesh(get_data_for_momentum_mesh()),
                alps::gf::momentum_index_mesh(get_data_for_momentum_mesh()),
                alps::gf::index_mesh(nspins)),
             gf2(gf) {}
};
    

TEST_F(FourIndexGFTest,access)
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

TEST_F(FourIndexGFTest,init)
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

TEST_F(FourIndexGFTest,saveload)
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


TEST_F(FourIndexGFTest, tail)
{
    namespace g=alps::gf;
    typedef g::three_index_gf<double, g::momentum_index_mesh, g::momentum_index_mesh, g::index_mesh> density_matrix_type;
    density_matrix_type denmat=density_matrix_type(g::momentum_index_mesh(get_data_for_momentum_mesh()),g::momentum_index_mesh(get_data_for_momentum_mesh()),
                                                   g::index_mesh(nspins));

    // prepare diagonal matrix
    double U=3.0;
    denmat.initialize();
    for (g::momentum_index i=g::momentum_index(0); i<denmat.mesh1().extent(); ++i) {
        denmat(i,i,g::index(0))=0.5*U;
        denmat(i,i,g::index(1))=0.5*U;
    }

    // Attach a tail to the GF
    int order=0;
    
    // FIXME: TODO: gf.set_tail(min_order, max_order, denmat, ...);
    g::omega_k1_k2_sigma_gf_with_tail gft(gf);
    gft.set_tail(order, denmat)
    // .set_tail(order+1, other_gf) ....
        ;
    
    EXPECT_NEAR((denmat-gft.tail(order)).norm(), 0, 1.e-8);
}

TEST_F(FourIndexGFTest, TailSaveLoad)
{
    namespace g=alps::gf;
    typedef g::three_index_gf<double, g::momentum_index_mesh, g::momentum_index_mesh, g::index_mesh> density_matrix_type;
    density_matrix_type denmat=density_matrix_type(g::momentum_index_mesh(get_data_for_momentum_mesh()), g::momentum_index_mesh(get_data_for_momentum_mesh()),
                                                   g::index_mesh(nspins));

    // prepare diagonal matrix
    double U=3.0;
    denmat.initialize();
    for (g::momentum_index i=g::momentum_index(0); i<denmat.mesh1().extent(); ++i) {
        denmat(i,i,g::index(0))=0.5*U;
        denmat(i,i,g::index(1))=0.5*U;
    }

    // Attach a tail to the GF
    int order=0;
    
    // FIXME: TODO: gf.set_tail(min_order, max_order, denmat, ...);
    g::omega_k1_k2_sigma_gf_with_tail gft(gf);
    g::omega_k1_k2_sigma_gf_with_tail gft2(gft);
    EXPECT_EQ((g::omega_k1_k2_sigma_gf_with_tail::TAIL_NOT_SET+0),gft.min_tail_order());
    EXPECT_EQ((g::omega_k1_k2_sigma_gf_with_tail::TAIL_NOT_SET+0),gft.max_tail_order());

    gft.set_tail(order, denmat);

    EXPECT_EQ(0,gft.min_tail_order());
    EXPECT_EQ(0,gft.max_tail_order());
    EXPECT_EQ(0,(denmat-gft.tail(0)).norm());
    {
        alps::hdf5::archive oar("gft.h5","w");
        gft(g::matsubara_index(4),g::momentum_index(3), g::momentum_index(2), g::index(1))=std::complex<double>(7., 3.);
        gft.save(oar,"/gft");
    }
    {
        alps::hdf5::archive iar("gft.h5");
        
        gft2.load(iar,"/gft");
    }
    EXPECT_EQ(gft2.tail().size(), gft.tail().size()) << "Tail size mismatch";
    EXPECT_NEAR(0, (gft.tail(0)-gft2.tail(0)).norm(), 1E-8)<<"Tail loaded differs from tail stored"; 

    EXPECT_EQ(7, gft2(g::matsubara_index(4),g::momentum_index(3), g::momentum_index(2), g::index(1)).real()) << "GF real part mismatch";
    EXPECT_EQ(3, gft2(g::matsubara_index(4),g::momentum_index(3), g::momentum_index(2), g::index(1)).imag()) << "GF imag part mismatch";
}



