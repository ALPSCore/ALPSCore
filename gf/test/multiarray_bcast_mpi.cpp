/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/gf/gf.hpp>

#include "gtest/gtest.h"
#include "mpi_guard.hpp"
#include <alps/utilities/gtest_par_xml_output.hpp>

#define ARRAY_EXTENTS boost::extents[2][3][5][7]

class GfMultiArrayTest : public ::testing::Test {
    protected:
    int rank_;
    bool is_root_;
    typedef boost::multi_array<std::complex<double>,4> data_type;
    data_type ref_data_;
    boost::array<data_type::size_type, 4> ref_shape_;

    public:
    static const int MASTER=0;
    static const int BASE=1;

    GfMultiArrayTest() : ref_data_(ARRAY_EXTENTS) {
        rank_=alps::mpi::communicator().rank();
        is_root_=(rank_==MASTER);

        for (data_type::index i=0;
             static_cast<data_type::size_type>(i)<ref_data_.num_elements();
             ++i) {
            *(ref_data_.origin()+i)=std::complex<double>(i+0.5,i-0.5);
        }
        std::copy(ref_data_.shape(), ref_data_.shape()+ref_data_.num_dimensions(), ref_shape_.begin());
        ref_data_.reindex(BASE);
    }
};

TEST_F(GfMultiArrayTest, MpiBroadcast) {
    data_type mydata;
    if (is_root_) {
        mydata.resize(ARRAY_EXTENTS);
        mydata=ref_data_;
        mydata.reindex(BASE);
    } else {
        mydata.reindex(5); // set a strange base on receiving ranks
    }

    alps::gf::detail::broadcast(alps::mpi::communicator(), mydata, MASTER);

    // Compare with ref, element-by-element
    for (std::size_t i0=0; i0<ref_shape_[0]; ++i0) {
        for (std::size_t i1=0; i1<ref_shape_[1]; ++i1) {
            for (std::size_t i2=0; i2<ref_shape_[2]; ++i2) {
                for (std::size_t i3=0; i3<ref_shape_[3]; ++i3) {
                    ASSERT_EQ(ref_data_[BASE+i0][BASE+i1][BASE+i2][BASE+i3].real(),
                              mydata[BASE+i0][BASE+i1][BASE+i2][BASE+i3].real())
                        << "The reference and the broadcast arrays differ on rank #" << rank_;
                    ASSERT_EQ(ref_data_[BASE+i0][BASE+i1][BASE+i2][BASE+i3].imag(),
                              mydata[BASE+i0][BASE+i1][BASE+i2][BASE+i3].imag())
                        << "The reference and the broadcast arrays differ on rank #" << rank_;
                }
            }
        }
    }
}

// This test should result in program crash (call to MPI_Abort())
TEST_F(GfMultiArrayTest, DISABLED_MpiBroadcastDimMismatch) {
    typedef boost::multi_array<std::complex<double>, 2> mis_data_type;
    mis_data_type mydata(boost::extents[2][3]);

    if (is_root_) {
        data_type root_data;
        root_data.resize(ARRAY_EXTENTS);
        root_data=ref_data_;
        alps::gf::detail::broadcast(alps::mpi::communicator(), root_data, MASTER);
    } else {
        alps::gf::detail::broadcast(alps::mpi::communicator(), mydata, MASTER);
        FAIL() << "This point should not be reachable.";
    }
}


// for testing MPI, we need main()
int main(int argc, char**argv)
{
    alps::mpi::environment env(argc, argv, false);
    alps::gtest_par_xml_output tweak;
    tweak(alps::mpi::communicator().rank(), argc, argv);

    ::testing::InitGoogleTest(&argc, argv);

    Mpi_guard guard(0, "multiarray_bcast_mpi.dat.");
    // std::cout << "pid=" << getpid() << " rank=" << guard.rank() << std::endl;
    // sleep(20);

    int rc=RUN_ALL_TESTS();
    if (!guard.check_sig_files_ok(get_number_of_bcasts())) {
        MPI_Abort(MPI_COMM_WORLD, 1); // otherwise it may get stuck in MPI_Finalize().
        // downside is the test aborts, rather than reports failure!
    }

    return rc;
}
