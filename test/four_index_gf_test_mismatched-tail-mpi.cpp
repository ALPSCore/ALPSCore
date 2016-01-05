#include "four_index_gf_test.hpp"

#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <boost/lexical_cast.hpp>

/* NOTE: This program should be compiled as a separate executable
         and run as a separate MPI process set, because it potentially
         messes up MPI library state.

   NOTE: This program relies on file I/O to exchange info between processes
         --- do NOT run on too many cores!
*/

#include <mpi.h>

/* The following piece of code is a primitive DIY MPI-correctness checker. */

static int Number_of_bcasts=0; //< Number of broadcasts performed.

// We intercept MPI_Bcast using PMPI interface.
extern "C" int MPI_Bcast(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);
extern "C" int PMPI_Bcast(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);

int MPI_Bcast(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
{
    ++Number_of_bcasts;
    return PMPI_Bcast(buffer, count, datatype, root, comm);
}

// Check incompatible broadcast of tails
TEST_F(FourIndexGFTest,MpiWrongTailBroadcast)
{
    namespace g=alps::gf;
    typedef g::three_index_gf<double,
                              g::momentum_index_mesh,
                              g::momentum_index_mesh,
                              g::index_mesh> density_matrix_type;

    density_matrix_type denmat=density_matrix_type(g::momentum_index_mesh(get_data_for_momentum_mesh()),
                                                   g::momentum_index_mesh(get_data_for_momentum_mesh()),
                                                   g::index_mesh(this->nspins));

    // MPI stuff.
    // Get the rank
    int rank,nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    const int master=0;
    // As MPI will be likely messed up, we communicate via a file.
    const std::string sig_fname_base="four_index_gf_test_mismatched-tail-mpi.dat."; // FIXME? use unique name?
    const std::string sig_fname=sig_fname_base+boost::lexical_cast<std::string>(rank);
    std::remove(sig_fname.c_str());
    std::fstream sig_fs;
    if (rank==master) {
        sig_fs.open(sig_fname.c_str(), std::ios::out); // master creates file to signal slaves to wait
        if (!sig_fs) {
            std::cerr << "Cannot open communication file " << sig_fname << std::endl;
            MPI_Abort(MPI_COMM_WORLD,1);
            return;
        }
        sig_fs << "?";
        sig_fs.close();
    }
    MPI_Barrier(MPI_COMM_WORLD); // MPI is still ok here, we can synchronize
    

    // prepare diagonal matrix
    double U=3.0;
    denmat.initialize();
    for (g::momentum_index i=g::momentum_index(0); i<denmat.mesh1().extent(); ++i) {
      denmat(i,i,g::index(0))=0.5*U;
      denmat(i,i,g::index(1))=0.5*U;
    }

    g::omega_k1_k2_sigma_gf_with_tail gft(gf);
    
    int order=0;
    if (rank==master) { // on master only
      // attach a tail to the GF:
      gft.set_tail(order, denmat);
      // change the GF
      gft(g::matsubara_index(4),g::momentum_index(3), g::momentum_index(2), g::index(1))=std::complex<double>(7., 3.);
    } else { // on receiving ranks, make sure that the tail is there, with empty data
        // DEBUG!!! Temporarily disable the check
      // gft.set_tail(order, density_matrix_type(g::momentum_index_mesh(denmat.mesh1().points()),
      //                                         g::momentum_index_mesh(denmat.mesh2().points()),
      //                                         g::index_mesh(this->nspins)));
    }

    // broadcast the GF with tail
    gft.broadcast_data(master,MPI_COMM_WORLD);

    EXPECT_EQ(7, gft(g::matsubara_index(4),g::momentum_index(3), g::momentum_index(2), g::index(1)).real()) << "GF real part mismatch on rank " << rank;
    EXPECT_EQ(3, gft(g::matsubara_index(4),g::momentum_index(3), g::momentum_index(2), g::index(1)).imag()) << "GF imag part mismatch on rank " << rank;

    // DEBUG!!! Temporarily disable the check
    // ASSERT_EQ(1, gft.tail().size()) << "Tail size mismatch on rank " << rank;
    // EXPECT_NEAR(0, (gft.tail(0)-denmat).norm(), 1E-8) << "Tail broadcast differs from the received on rank " << rank; 


    // MPI stuff again.
    // Now we have to check the number of bcast operations.
    // As MPI may be already in an incorrect state, we rely exclusively on file I/O
    // (inefficient, but simple, and uses only basic C/C++ file operations).
    if (rank!=master) {
        // slave process: write the data...
        sig_fs.open(sig_fname.c_str(), std::ios::out);
        sig_fs << Number_of_bcasts;
        sig_fs.close();
        // ... then wait till the master's file disappears
        const std::string sig_fname_master=sig_fname_base+boost::lexical_cast<std::string>(master);
        for (;;) {
            sig_fs.open(sig_fname_master.c_str(), std::ios::in);
            if (!sig_fs) break;
            sig_fs.close();
            sleep(3);
        }
    } else {
        // master process: wait till all slaves report
        for (int i=0; i<nprocs; ++i) {
            if (i==master) continue; // don't wait on myself
            const std::string sig_fname_slave=sig_fname_base+boost::lexical_cast<std::string>(i);
            int nbc=-1;
            int itry;
            for (itry=1; itry<=100; ++itry) {
                sleep(1);
                std::ifstream sig_in_fs(sig_fname_slave.c_str());
                if (!sig_in_fs) continue;
                if (!(sig_in_fs >> nbc)) continue;
                break;
            }
            std::cout << "DEBUG: after " << itry << " tries, got info from rank #" << i << ": " << nbc << std::endl;
            EXPECT_EQ(Number_of_bcasts, nbc) << " mismatch in number of broadcasts for rank #" << i;
        }
        // All ranks info collected, remove the master's communication file
        std::remove(sig_fname.c_str());
    }
    
    // alps::gf::matsubara_index omega; omega=4;
    // alps::gf::momentum_index i; i=2;
    // alps::gf::momentum_index j=alps::gf::momentum_index(3);
    // alps::gf::index sigma(1);

    // const int f=(rank==master)?1:2;
    // alps::gf::matsubara_gf gf_wrong(matsubara_mesh(f*beta,f*nfreq),
    //                                 alps::gf::momentum_index_mesh(5,1),
    //                                 alps::gf::momentum_index_mesh(5,1),
    //                                 alps::gf::index_mesh(nspins));
    // gf_wrong.initialize();
    // if (rank==master) {
    //   gf_wrong(omega,i,j,sigma)=std::complex<double>(3,4);
    // }

    // bool thrown=false;
    // try {
    //     gf_wrong.broadcast_data(master,MPI_COMM_WORLD);
    // } catch (...) {
    //     thrown=true;
    // }
    // EXPECT_TRUE( (rank==master && !thrown) || (rank!=master && thrown) );

    // if (thrown) return;
    
    // {
    //   std::complex<double> x=gf_wrong(omega,i,j,sigma);
    //     EXPECT_NEAR(3, x.real(),1.e-10);
    //     EXPECT_NEAR(4, x.imag(),1.e-10);
    // }
}

// for testing MPI, we need main()
int main(int argc, char**argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int rc=RUN_ALL_TESTS();
    MPI_Finalize();
    return rc;
}
