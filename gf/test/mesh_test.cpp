/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include "gtest/gtest.h"
#include "alps/gf/mesh.hpp"
#include "alps/gf/grid.hpp"
#include "alps/gf/piecewise_polynomial.hpp"
#include "gf_test.hpp"

#include <boost/filesystem/operations.hpp>

TEST(Mesh, RealFrequencyLoadSave) {
    namespace g=alps::gf;
    g::grid::linear_real_frequency_grid grid(-3,3,100);
    g::real_frequency_mesh mesh1(grid);
    g::real_frequency_mesh mesh2;
    {
        alps::hdf5::archive oar("gf.h5","w");
        mesh1.save(oar,"/gf");
    }
    {
        alps::hdf5::archive iar("gf.h5");
        g::real_frequency_mesh mesh;
        mesh2.load(iar,"/gf");
    }
    EXPECT_EQ(mesh1, mesh2);
    boost::filesystem::remove("gf.h5");
}

TEST(Mesh, RealFrequencyMeshQuadric) {
    double spread = 5.0;
    int nfreq = 41;
    alps::gf::grid::quadratic_real_frequency_grid grid(spread, nfreq);
    alps::gf::real_frequency_mesh mesh1(grid);
    EXPECT_EQ(mesh1.extent(), nfreq);
}

TEST(Mesh, RealFrequencyMeshLogarithmic) {
    double tmax = 5, tmin = 0.001;
    int nfreq = 41;
    alps::gf::grid::logarithmic_real_frequency_grid grid(tmin, tmax, nfreq);
    alps::gf::real_frequency_mesh mesh1(grid);
    EXPECT_EQ(mesh1.extent(), nfreq);
}

TEST(Mesh, RealFrequencyMeshLinear) {
    double Emin = -5;
    double Emax = 5;
    int nfreq = 20;
    alps::gf::grid::linear_real_frequency_grid grid(Emin, Emax, nfreq);
    alps::gf::real_frequency_mesh mesh1(grid);
    EXPECT_EQ(mesh1.extent(), nfreq);
}

TEST(Mesh, BosonicMatsubara) {
    alps::gf::matsubara_mesh<alps::gf::mesh::POSITIVE_NEGATIVE> mesh1(5.0, 20, alps::gf::statistics::BOSONIC);
    EXPECT_EQ(mesh1.statistics(), alps::gf::statistics::BOSONIC);
}

TEST(Mesh,SwapMatsubara) {
    alps::gf::matsubara_mesh<alps::gf::mesh::POSITIVE_NEGATIVE> mesh1(5.0, 20);
    alps::gf::matsubara_mesh<alps::gf::mesh::POSITIVE_NEGATIVE> mesh2(7.0, 40);
    mesh1.swap(mesh2);

    EXPECT_EQ(mesh1.beta(), 7.0);
    EXPECT_EQ(mesh2.beta(), 5.0);
}

TEST(Mesh,SwapMatsubaraCheckStatistics) {
    alps::gf::matsubara_mesh<alps::gf::mesh::POSITIVE_NEGATIVE> mesh1(5.0, 20, alps::gf::statistics::BOSONIC);
    alps::gf::matsubara_mesh<alps::gf::mesh::POSITIVE_NEGATIVE> mesh2(7.0, 40, alps::gf::statistics::FERMIONIC);

    EXPECT_ANY_THROW(mesh1.swap(mesh2));
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

TEST(Mesh,PowerMeshHasRightSize) {
  alps::gf::power_mesh mesh1(20, 12, 16);
  EXPECT_EQ(417, mesh1.points().size());
  alps::gf::power_mesh mesh2(20, 12, 256);
  EXPECT_EQ(6657, mesh2.points().size());

}
TEST(Mesh,ComparePower) {
  alps::gf::power_mesh mesh1(12, 16, 20);
  alps::gf::power_mesh mesh2(12, 16, 20);
  alps::gf::power_mesh mesh3(11, 16, 20);
  alps::gf::power_mesh mesh4(12, 12, 20);
  alps::gf::power_mesh mesh5(12, 16, 22);

  EXPECT_TRUE(mesh1==mesh2);
  EXPECT_TRUE(mesh1!=mesh3);
  EXPECT_TRUE(mesh1!=mesh4);
  EXPECT_TRUE(mesh1!=mesh5);

  EXPECT_FALSE(mesh1==mesh3);
  EXPECT_FALSE(mesh1!=mesh2);
}

TEST(Mesh,CompareMomentum) {
  alps::gf::momentum_index_mesh::container_type points1(boost::extents[20][3]);
  alps::gf::momentum_index_mesh::container_type points2(boost::extents[20][3]);
  alps::gf::momentum_index_mesh::container_type points3(boost::extents[20][3]);
  alps::gf::momentum_index_mesh::container_type points4(boost::extents[3][20]);
  for (std::size_t i=0; i<points1.num_elements(); ++i) {
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
  for (std::size_t i=0; i<points1.num_elements(); ++i) {
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

TEST(Mesh,Legendre) {
    alps::gf::legendre_mesh mesh_f(5.0, 20, alps::gf::statistics::FERMIONIC);
    EXPECT_EQ(mesh_f.statistics(), alps::gf::statistics::FERMIONIC);

    alps::gf::legendre_mesh mesh_b(5.0, 20, alps::gf::statistics::BOSONIC);
    EXPECT_EQ(mesh_b.statistics(), alps::gf::statistics::BOSONIC);
}

TEST(Mesh,CompareLegendre) {
    alps::gf::legendre_mesh mesh_f1(5.0, 20, alps::gf::statistics::FERMIONIC);
    alps::gf::legendre_mesh mesh_f2(1.0, 20, alps::gf::statistics::FERMIONIC);
    alps::gf::legendre_mesh mesh_f3(5.0, 20, alps::gf::statistics::FERMIONIC);

    alps::gf::legendre_mesh mesh_b1(5.0, 20, alps::gf::statistics::BOSONIC);
    alps::gf::legendre_mesh mesh_b2(1.0, 20, alps::gf::statistics::BOSONIC);
    alps::gf::legendre_mesh mesh_b3(5.0, 20, alps::gf::statistics::BOSONIC);

    EXPECT_FALSE(mesh_f1==mesh_f2);
    EXPECT_TRUE(mesh_f1==mesh_f3);

    EXPECT_FALSE(mesh_b1==mesh_b2);
    EXPECT_TRUE(mesh_b1==mesh_b3);

    EXPECT_FALSE(mesh_f1==mesh_b1);
}

TEST(Mesh, LegendreLoadSave) {
    namespace g=alps::gf;
    g::legendre_mesh mesh1(5.0, 20, alps::gf::statistics::FERMIONIC);
    g::legendre_mesh mesh2;
    {
        alps::hdf5::archive oar("gf.h5","w");
        mesh1.save(oar,"/gf");
    }
    {
        alps::hdf5::archive iar("gf.h5");
        g::legendre_mesh mesh;
        mesh2.load(iar,"/gf");
    }
    EXPECT_EQ(mesh1, mesh2);
    boost::filesystem::remove("gf.h5");
}

TEST(Mesh,SwapLegendre) {
    alps::gf::legendre_mesh mesh_1(5.0, 20, alps::gf::statistics::FERMIONIC);
    alps::gf::legendre_mesh mesh_1r(mesh_1);
    alps::gf::legendre_mesh mesh_2(10.0, 40, alps::gf::statistics::FERMIONIC);
    alps::gf::legendre_mesh mesh_2r(mesh_2);

    mesh_1.swap(mesh_2);
    EXPECT_EQ(mesh_1, mesh_2r);
    EXPECT_EQ(mesh_2, mesh_1r);
}

TEST(Mesh,SwapLegendreDifferentStatistics) {
    alps::gf::legendre_mesh mesh_1(5.0, 20, alps::gf::statistics::BOSONIC);
    alps::gf::legendre_mesh mesh_1r(mesh_1);
    alps::gf::legendre_mesh mesh_2(10.0, 40, alps::gf::statistics::FERMIONIC);
    alps::gf::legendre_mesh mesh_2r(mesh_2);

    EXPECT_THROW(mesh_1.swap(mesh_2), std::runtime_error);
}

TEST(Mesh,PrintMatsubaraMeshHeader) {
  double beta=5.;
  int n=20;
  {
    std::stringstream header_line;
    header_line << "# MATSUBARA mesh: N: "<<n<<" beta: "<<beta<<" statistics: "<<"FERMIONIC"<<" POSITIVE_ONLY"<<std::endl;
    alps::gf::matsubara_mesh<alps::gf::mesh::POSITIVE_ONLY> mesh1(beta, n);
    std::stringstream header_line_from_mesh;
    header_line_from_mesh << mesh1;
    EXPECT_EQ(header_line.str(), header_line_from_mesh.str());
  }
  {
    std::stringstream header_line;
    header_line << "# MATSUBARA mesh: N: "<<n<<" beta: "<<beta<<" statistics: "<<"FERMIONIC"<<" POSITIVE_NEGATIVE"<<std::endl;
    alps::gf::matsubara_mesh<alps::gf::mesh::POSITIVE_NEGATIVE> mesh1(beta, n);
    std::stringstream header_line_from_mesh;
    header_line_from_mesh << mesh1;
    EXPECT_EQ(header_line.str(), header_line_from_mesh.str());
  }

}

TEST(Mesh,PrintImagTimeMeshHeader) {
  double beta=5.;
  int ntau=200;
  std::stringstream header_line;
  header_line << "# IMAGINARY_TIME mesh: N: "<<ntau<<" beta: "<<beta<<" statistics: "<<"FERMIONIC"<<std::endl;
  alps::gf::itime_mesh mesh1(beta, ntau);
  std::stringstream header_line_from_mesh;
  header_line_from_mesh << mesh1;
  EXPECT_EQ(header_line.str(), header_line_from_mesh.str());
}
TEST(Mesh,PrintPowerMeshHeader) {
  double beta=5.;
  int power=12;
  int uniform=16;
  int ntau=417;
  std::stringstream header_line;
  header_line << "# POWER mesh: power: "<<power<<" uniform: "<<uniform<<" N: "<<ntau<<" beta: "<<beta<<" statistics: "<<"FERMIONIC"<<std::endl;
  alps::gf::power_mesh mesh1(beta, power, uniform);
  std::stringstream header_line_from_mesh;
  header_line_from_mesh << mesh1;
  EXPECT_EQ(header_line.str(), header_line_from_mesh.str());
}

TEST(Mesh,PrintMomentumMeshHeader) {
  alps::gf::momentum_index_mesh::container_type data=get_data_for_momentum_mesh();
  std::stringstream header_line;
  header_line << "# MOMENTUM_INDEX mesh: N: "<<data.shape()[0]<<" dimension: "<<data.shape()[1]<<" points: ";
  for(std::size_t i=0;i<data.shape()[0];++i){
    header_line<<"(";
    for(std::size_t d=0;d<data.shape()[1]-1;++d){ header_line<<data[i][d]<<","; } header_line<<data[i][data.shape()[1]-1]<<") ";
  }
  header_line<<std::endl;
  alps::gf::momentum_index_mesh mesh1(data);
  std::stringstream header_line_from_mesh;
  header_line_from_mesh << mesh1;
  EXPECT_EQ(header_line.str(), header_line_from_mesh.str());
}
TEST(Mesh,PrintRealSpaceMeshHeader) {
  alps::gf::real_space_index_mesh::container_type data=get_data_for_real_space_mesh();
  std::stringstream header_line;
  header_line << "# REAL_SPACE_INDEX mesh: N: "<<data.shape()[0]<<" dimension: "<<data.shape()[1]<<" points: ";
  for(std::size_t i=0;i<data.shape()[0];++i){
    header_line<<"(";
    for(std::size_t d=0;d<data.shape()[1]-1;++d){ header_line<<data[i][d]<<","; } header_line<<data[i][data.shape()[1]-1]<<") ";
  }
  header_line<<std::endl;
  alps::gf::real_space_index_mesh mesh1(data);
  std::stringstream header_line_from_mesh;
  header_line_from_mesh << mesh1;
  EXPECT_EQ(header_line.str(), header_line_from_mesh.str());
}
TEST(Mesh,PrintIndexMeshHeader) {
  int n=2;
  std::stringstream header_line;
  header_line << "# INDEX mesh: N: "<<n<<std::endl;
  alps::gf::index_mesh mesh1(2);
  std::stringstream header_line_from_mesh;
  header_line_from_mesh << mesh1;
  EXPECT_EQ(header_line.str(), header_line_from_mesh.str());
}
TEST(Mesh,PowerWeightsAddUpToOne) {
  alps::gf::power_mesh mesh1(20, 12, 16);
  //check that the integration weights add up to 1:
  double sum=0;
  for(int i=0;i<mesh1.extent();++i) sum+=mesh1.weights()[i];
  EXPECT_NEAR(1, sum, 1.e-10);
}

TEST(Mesh,SwapNumericalMesh) {
    const int n_section = 2, k = 3;
    const double beta = 100.0;
    typedef double Scalar;
    typedef alps::gf::piecewise_polynomial<Scalar,k> pp_type;

    std::vector<double> section_edges(n_section+1);
    section_edges[0] = -1.0;
    section_edges[1] =  0.0;
    section_edges[2] =  1.0;

    boost::multi_array<Scalar,2> coeff(boost::extents[n_section][k+1]);
    std::fill(coeff.origin(), coeff.origin()+coeff.num_elements(), 0.0);

    pp_type p(n_section, section_edges, coeff);

    std::vector<pp_type> basis_functions;
    basis_functions.push_back(p);
    basis_functions.push_back(p);

    std::vector<pp_type> basis_functions2;
    basis_functions2.push_back(p);

    alps::gf::numerical_mesh<double,k> mesh1(beta, basis_functions, alps::gf::statistics::FERMIONIC);
    alps::gf::numerical_mesh<double,k> mesh2(beta, basis_functions2, alps::gf::statistics::FERMIONIC);
    alps::gf::numerical_mesh<double,k> mesh3(beta, basis_functions2, alps::gf::statistics::BOSONIC);

    mesh1.swap(mesh2);
    ASSERT_TRUE(mesh1.extent()==1);
    ASSERT_TRUE(mesh2.extent()==2);

    ASSERT_THROW(mesh1.swap(mesh3), std::runtime_error);
}

TEST(Mesh,NumericalMeshSave) {
    const int n_section = 2, k = 3;
    const double beta = 100.0;
    typedef double Scalar;
    typedef alps::gf::piecewise_polynomial<Scalar,k> pp_type;

    std::vector<double> section_edges(n_section+1);
    section_edges[0] = -1.0;
    section_edges[1] =  0.0;
    section_edges[2] =  1.0;

    boost::multi_array<Scalar,2> coeff(boost::extents[n_section][k+1]);
    std::fill(coeff.origin(), coeff.origin()+coeff.num_elements(), 0.0);

    pp_type p(n_section, section_edges, coeff);
    std::vector<pp_type> basis_functions;
    basis_functions.push_back(p);
    basis_functions.push_back(p);

    alps::gf::numerical_mesh<double,k> mesh1(beta, basis_functions, alps::gf::statistics::FERMIONIC);
    {
        alps::hdf5::archive oar("nm.h5","w");
        mesh1.save(oar,"/nm");

    }

    alps::gf::numerical_mesh<double,k> mesh2;
    {
        alps::hdf5::archive iar("nm.h5");
        mesh2.load(iar,"/nm");
    }
    boost::filesystem::remove("nm.h5");
}


