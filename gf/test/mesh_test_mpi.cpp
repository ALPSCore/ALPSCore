/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include "gtest/gtest.h"
#include <alps/utilities/gtest_par_xml_output.hpp>
#include "alps/gf/mesh.hpp"
#include "alps/gf/grid.hpp"
#include "gf_test.hpp"

#include "mpi_guard.hpp"


class MeshTest : public ::testing::Test {
  protected:
    int rank_;
    bool is_root_;
  public:
    static const int MASTER=0;
    MeshTest() {
        rank_=alps::mpi::communicator().rank();
        is_root_=(rank_==MASTER);
    }
};

namespace agf=alps::gf;
// namespace gfm=agf::mesh;

TEST_F(MeshTest,MpiBcastRealFrequency) {
    typedef agf::real_frequency_mesh mesh_type;
    agf::grid::linear_real_frequency_grid grid(-3.,3.,10);
    int tmax = 2;
    double tmin = 0.00001;
    int c = -1;
    int nfreq = 4;
    agf::grid::logarithmic_real_frequency_grid grid2(tmin, tmax, c, nfreq);
    mesh_type ref_mesh(grid);
    mesh_type my_mesh(grid2);
    mesh_type* mesh_ptr= is_root_ ? &ref_mesh : &my_mesh;
    mesh_ptr->broadcast(alps::mpi::communicator(), MASTER);
    EXPECT_EQ(*mesh_ptr, ref_mesh) << "Failed at rank=" << rank_;
}

TEST_F(MeshTest,MpiBcastMatsubara) {
    typedef agf::matsubara_mesh<agf::mesh::POSITIVE_NEGATIVE> mesh_type;
    mesh_type ref_mesh(5.0, 20);
    mesh_type my_mesh(1,1);
    mesh_type* mesh_ptr= is_root_ ? &ref_mesh : &my_mesh;

    mesh_ptr->broadcast(alps::mpi::communicator(), MASTER);

    EXPECT_EQ(*mesh_ptr, ref_mesh) << "Failed at rank=" << rank_;
}

TEST_F(MeshTest,MpiBcastMatsubara2) {
  typedef agf::matsubara_mesh<agf::mesh::POSITIVE_NEGATIVE> mesh_type_1;
  typedef agf::matsubara_mesh<agf::mesh::POSITIVE_ONLY> mesh_type_2;
  mesh_type_1 ref_mesh(5.0, 20,alps::gf::statistics::FERMIONIC);
  mesh_type_1 same_mesh(1,1,alps::gf::statistics::FERMIONIC);
  mesh_type_1 diff_stat(1,1,alps::gf::statistics::BOSONIC);
  mesh_type_2 diff_mesh(1,1,alps::gf::statistics::FERMIONIC);
  mesh_type_2 diff_mesh_stat(1,1,alps::gf::statistics::BOSONIC);
  if(rank_ == MASTER) {
    ref_mesh.broadcast(alps::mpi::communicator(), MASTER);
    ref_mesh.broadcast(alps::mpi::communicator(), MASTER);
    ref_mesh.broadcast(alps::mpi::communicator(), MASTER);
    ref_mesh.broadcast(alps::mpi::communicator(), MASTER);
  } else {
    same_mesh.broadcast(alps::mpi::communicator(), MASTER);
    EXPECT_EQ(same_mesh, ref_mesh);
    diff_stat.broadcast(alps::mpi::communicator(), MASTER);
    EXPECT_EQ(diff_stat, ref_mesh);
    EXPECT_ANY_THROW(diff_mesh.broadcast(alps::mpi::communicator(), MASTER));
    EXPECT_ANY_THROW(diff_mesh_stat.broadcast(alps::mpi::communicator(), MASTER));
  }
}

TEST_F(MeshTest,MpiBcastITime) {
    agf::itime_mesh ref_mesh(5.0, 20);
    agf::itime_mesh my_mesh(1,1);
    agf::itime_mesh* mesh_ptr= is_root_? &ref_mesh:&my_mesh;
    mesh_ptr->broadcast(alps::mpi::communicator(), MASTER);

    EXPECT_EQ(*mesh_ptr, ref_mesh) << "Failed at rank=" << rank_;
}

TEST_F(MeshTest,MpiBcastMomentum) {
    typedef agf::momentum_index_mesh mesh_type;
    mesh_type::container_type points(boost::extents[20][3]);
    for (std::size_t i=0; i<points.num_elements(); ++i) {
        *(points.origin()+i)=i;
    }

    mesh_type ref_mesh(points);
    mesh_type my_mesh=mesh_type((mesh_type::container_type()));
    mesh_type* mesh_ptr= is_root_ ? &ref_mesh : &my_mesh;

    mesh_ptr->broadcast(alps::mpi::communicator(), MASTER);

    EXPECT_EQ(*mesh_ptr, ref_mesh) << "Failed at rank=" << rank_;
}

TEST_F(MeshTest,MpiBcastRealspace) {
    typedef agf::real_space_index_mesh mesh_type;
    mesh_type::container_type points(boost::extents[20][3]);
    for (std::size_t i=0; i<points.num_elements(); ++i) {
        *(points.origin()+i)=i;
    }

    mesh_type ref_mesh(points);
    mesh_type my_mesh=mesh_type((mesh_type::container_type()));
    mesh_type* mesh_ptr= is_root_ ? &ref_mesh : &my_mesh;

    mesh_ptr->broadcast(alps::mpi::communicator(), MASTER);

    EXPECT_EQ(*mesh_ptr, ref_mesh) << "Failed at rank=" << rank_;
}

TEST_F(MeshTest,MpiBcastIndex) {
    typedef agf::index_mesh mesh_type;
    mesh_type ref_mesh(20);
    mesh_type my_mesh(1);
    mesh_type* mesh_ptr= is_root_ ? &ref_mesh : &my_mesh;

    mesh_ptr->broadcast(alps::mpi::communicator(), MASTER);

    EXPECT_EQ(*mesh_ptr, ref_mesh) << "Failed at rank=" << rank_;
}

TEST_F(MeshTest,MpiBcastLegendre) {
    typedef agf::legendre_mesh mesh_type;
    const double beta = 5.0;
    mesh_type ref_mesh(beta, 20);
    mesh_type my_mesh(beta, 1);
    mesh_type* mesh_ptr= is_root_ ? &ref_mesh : &my_mesh;

    mesh_ptr->broadcast(alps::mpi::communicator(), MASTER);

    EXPECT_EQ(*mesh_ptr, ref_mesh) << "Failed at rank=" << rank_;
}

TEST_F(MeshTest,MpiBcastNumericalMesh) {
    typedef agf::numerical_mesh<double> mesh_type;
    const double beta = 5.0;

    const int n_section = 2, k = 3, dim = 10;
    typedef double Scalar;
    typedef alps::gf::piecewise_polynomial<Scalar> pp_type;

    std::vector<double> section_edges(n_section+1);
    section_edges[0] = -1.0;
    section_edges[1] =  0.0;
    section_edges[2] =  1.0;
    boost::multi_array<Scalar,2> coeff(boost::extents[n_section][k+1]);

    std::vector<pp_type> bf, bf2;
    for (int l = 0; l < dim; ++l){
        std::fill(coeff.origin(), coeff.origin()+coeff.num_elements(), 0.0);
        bf.push_back(pp_type(n_section, section_edges, coeff));

        std::fill(coeff.origin(), coeff.origin()+coeff.num_elements(), 1.0);
        bf2.push_back(pp_type(n_section, section_edges, coeff));
    }

    mesh_type ref_mesh(beta, bf);
    mesh_type my_mesh(beta, bf2);
    mesh_type* mesh_ptr= is_root_ ? &ref_mesh : &my_mesh;

    mesh_ptr->broadcast(alps::mpi::communicator(), MASTER);

    EXPECT_EQ(*mesh_ptr, ref_mesh) << "Failed at rank=" << rank_;
}


// for testing MPI, we need main()
int main(int argc, char**argv)
{
    alps::mpi::environment env(argc, argv, false);
    alps::gtest_par_xml_output tweak;
    tweak(alps::mpi::communicator().rank(), argc, argv);

    ::testing::InitGoogleTest(&argc, argv);

    Mpi_guard guard(0, "mesh_test_mpi.dat.");

    int rc=RUN_ALL_TESTS();

    if (!guard.check_sig_files_ok(get_number_of_bcasts())) {
        MPI_Abort(MPI_COMM_WORLD, 1); // otherwise it may get stuck in MPI_Finalize().
        // downside is the test aborts, rather than reports failure!
    }

    return rc;
}
