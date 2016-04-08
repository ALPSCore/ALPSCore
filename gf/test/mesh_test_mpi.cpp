#include "gtest/gtest.h"
#include "alps/gf/mesh.hpp"
#include "gf_test.hpp"

#include "mpi_guard.hpp"


class MeshTest : public ::testing::Test {
  protected:
    int rank_;
    bool is_root_;
  public:
    static const int MASTER=0;
    MeshTest() {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        is_root_=(rank_==MASTER);
    }
};

namespace agf=alps::gf;
// namespace gfm=agf::mesh;

TEST_F(MeshTest,MpiBcastMatsubara) {
    typedef agf::matsubara_mesh<agf::mesh::POSITIVE_NEGATIVE> mesh_type;
    mesh_type ref_mesh(5.0, 20);
    mesh_type my_mesh(1,1);
    mesh_type* mesh_ptr= is_root_ ? &ref_mesh : &my_mesh;

    mesh_ptr->broadcast(MASTER,MPI_COMM_WORLD);

    EXPECT_EQ(*mesh_ptr, ref_mesh) << "Failed at rank=" << rank_;
}

TEST_F(MeshTest,MpiBcastITime) {
    agf::itime_mesh ref_mesh(5.0, 20);
    agf::itime_mesh my_mesh(1,1);
    agf::itime_mesh* mesh_ptr= is_root_? &ref_mesh:&my_mesh;
    mesh_ptr->broadcast(MASTER,MPI_COMM_WORLD);

    EXPECT_EQ(*mesh_ptr, ref_mesh) << "Failed at rank=" << rank_;
}

TEST_F(MeshTest,MpiBcastMomentum) {
    typedef agf::momentum_index_mesh mesh_type;
    mesh_type::container_type points(boost::extents[20][3]);
    for (int i=0; i<points.num_elements(); ++i) {
        *(points.origin()+i)=i;
    }

    mesh_type ref_mesh(points);
    mesh_type my_mesh=mesh_type((mesh_type::container_type()));
    mesh_type* mesh_ptr= is_root_ ? &ref_mesh : &my_mesh;

    mesh_ptr->broadcast(MASTER,MPI_COMM_WORLD);

    EXPECT_EQ(*mesh_ptr, ref_mesh) << "Failed at rank=" << rank_;
}

TEST_F(MeshTest,MpiBcastRealspace) {
    typedef agf::real_space_index_mesh mesh_type;
    mesh_type::container_type points(boost::extents[20][3]);
    for (int i=0; i<points.num_elements(); ++i) {
        *(points.origin()+i)=i;
    }

    mesh_type ref_mesh(points);
    mesh_type my_mesh=mesh_type((mesh_type::container_type()));
    mesh_type* mesh_ptr= is_root_ ? &ref_mesh : &my_mesh;

    mesh_ptr->broadcast(MASTER,MPI_COMM_WORLD);

    EXPECT_EQ(*mesh_ptr, ref_mesh) << "Failed at rank=" << rank_;
}

TEST_F(MeshTest,MpiBcastIndex) {
    typedef agf::index_mesh mesh_type;
    mesh_type ref_mesh(20);
    mesh_type my_mesh(1);
    mesh_type* mesh_ptr= is_root_ ? &ref_mesh : &my_mesh;

    mesh_ptr->broadcast(MASTER,MPI_COMM_WORLD);

    EXPECT_EQ(*mesh_ptr, ref_mesh) << "Failed at rank=" << rank_;
}


// for testing MPI, we need main()
int main(int argc, char**argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);

    Mpi_guard guard(0, "mesh_test_mpi.dat.");

    int rc=RUN_ALL_TESTS();

    if (!guard.check_sig_files_ok(get_number_of_bcasts())) {
        MPI_Abort(MPI_COMM_WORLD, 1); // otherwise it may get stuck in MPI_Finalize().
        // downside is the test aborts, rather than reports failure!
    }

    MPI_Finalize();
    return rc;
}
