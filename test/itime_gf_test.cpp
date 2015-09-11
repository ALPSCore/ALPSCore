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
                alps::gf::momentum_index_mesh(get_data_for_momentum_mesh()),
                alps::gf::momentum_index_mesh(get_data_for_momentum_mesh()),
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


