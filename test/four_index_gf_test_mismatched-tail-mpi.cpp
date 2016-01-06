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

/* ^^^ End of the MPI-correctnes checker code  ^^^ */


/* The following is a simple DIY file-based, MPI-independent communication code.
   The intent is to check the number of bcast operations.
   As MPI may be in an incorrect state in the case of the mismatch,
   we rely exclusively on file I/O
   (inefficient, but simple, and uses only basic C/C++ file operations).
 */
class Mpi_guard {
    int master_, rank_, nprocs_;
    std::string sig_fname_base_, sig_fname_master_, sig_fname_;
  public:

    // Construct the object and create necessary files. MPI is assumed to be working.
    Mpi_guard(int master) : master_(master) {

        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &nprocs_);
        
        sig_fname_base_="four_index_gf_test_mismatched-tail-mpi.dat."; // FIXME? use unique name?
        sig_fname_=sig_fname_base_+boost::lexical_cast<std::string>(rank_);
        sig_fname_master_=sig_fname_base_+boost::lexical_cast<std::string>(master_);
        
        std::remove(sig_fname_.c_str());
        if (rank_==master_) {
            std::ofstream sig_fs(sig_fname_.c_str()); // master creates file to signal slaves to wait
            if (!sig_fs) {
                std::cerr << "Cannot open communication file " << sig_fname_ << std::endl;
                MPI_Abort(MPI_COMM_WORLD,1);
                return;
            }
            sig_fs << "?";
            sig_fs.close();
        }
        MPI_Barrier(MPI_COMM_WORLD); // MPI is still ok here, we can synchronize
    }

    // Check the number of broadcasts across processes. Do not rely on MPI. Return `true` if OK.
    bool check_sig_files_ok(int number_of_bcasts) {
        bool result=true;
    
        if (rank_!=master_) {
            // slave process: write the data...
            std::ofstream sig_fs(sig_fname_.c_str());
            sig_fs << number_of_bcasts;
            sig_fs.close();
            // ... then wait till the master's file disappears
            for (;;) {
                std::ifstream sig_fs(sig_fname_master_.c_str());
                if (!sig_fs) break;
                sig_fs.close();
                sleep(3);
            }
        } else {
            // master process: wait till all slaves report
            for (int i=0; i<nprocs_; ++i) {
                if (i==master_) continue; // don't wait on myself
                const std::string sig_fname_slave=sig_fname_base_+boost::lexical_cast<std::string>(i);
                int nbc=-1;
                int itry;
                for (itry=1; itry<=100; ++itry) {
                    sleep(1);
                    std::ifstream sig_in_fs(sig_fname_slave.c_str());
                    if (!sig_in_fs) continue;
                    if (!(sig_in_fs >> nbc)) continue;
                    break;
                }
                // std::cout << "DEBUG: after " << itry << " tries, got info from rank #" << i << ": " << nbc << std::endl;
                if (number_of_bcasts!=nbc) {
                    std::cout << " mismatch in number of broadcasts!"
                              << " master expects: " << number_of_bcasts
                              << " ; rank #" << i << " reports: " << nbc
                              << std::endl;
                    result=false;
                }
            }
            // All ranks info collected, remove the master's communication file
            std::remove(sig_fname_.c_str());
        }
        return result;
    }
};
/* ^^^ End of MPI-independent communication code ^^^ */


static const int MASTER=0;

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

    // Get the rank
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
    // prepare diagonal matrix
    double U=3.0;
    denmat.initialize();
    for (g::momentum_index i=g::momentum_index(0); i<denmat.mesh1().extent(); ++i) {
      denmat(i,i,g::index(0))=0.5*U;
      denmat(i,i,g::index(1))=0.5*U;
    }

    g::omega_k1_k2_sigma_gf_with_tail gft(gf);
    
    int order=0;
    if (rank==MASTER) { // on master only
      // attach a tail to the GF:
      gft.set_tail(order, denmat);
      // change the GF
      gft(g::matsubara_index(4),g::momentum_index(3), g::momentum_index(2), g::index(1))=std::complex<double>(7., 3.);
    }
    // slaves do not have the tail attached to their GF.

    // broadcast the GF with tail
    bool thrown=false;
    try {
        gft.broadcast_data(MASTER,MPI_COMM_WORLD);
    } catch (std::runtime_error exc) {
        // FIXME: verify exception message
        thrown=true;
    }
    EXPECT_TRUE( thrown ^ (rank==MASTER) );

    EXPECT_EQ(7, gft(g::matsubara_index(4),g::momentum_index(3), g::momentum_index(2), g::index(1)).real()) << "GF real part mismatch on rank " << rank;
    EXPECT_EQ(3, gft(g::matsubara_index(4),g::momentum_index(3), g::momentum_index(2), g::index(1)).imag()) << "GF imag part mismatch on rank " << rank;

    ASSERT_EQ(1, gft.tail().size()) << "Tail size mismatch on rank " << rank;
    EXPECT_NEAR(0, (gft.tail(0)-denmat).norm(), 1E-8) << "Tail broadcast differs from the received on rank " << rank;
}

// for testing MPI, we need main()
int main(int argc, char**argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);

    Mpi_guard guard(MASTER);
    
    int rc=RUN_ALL_TESTS();

    if (!guard.check_sig_files_ok(Number_of_bcasts)) {
        MPI_Abort(MPI_COMM_WORLD, 1); // otherwise it may get stuck in MPI_Finalize().
        // downside is the test aborts, rather than reports failure!
    }

    MPI_Finalize();
    return rc;
}
