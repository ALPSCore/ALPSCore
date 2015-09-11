#include "gtest/gtest.h"
#include "alps/gf/gf.hpp"
#include "alps/gf/tail.hpp"

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

