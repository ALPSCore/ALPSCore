/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file mpi_guard.cpp
    @brief A simple DIY MPI correctness checker (implementation file)
*/

#include "mpi_guard.hpp"

#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <boost/lexical_cast.hpp>

static int Number_of_bcasts=0; //< Number of broadcasts performed.

// We intercept MPI_Bcast using PMPI interface.
extern "C" int MPI_Bcast(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);
extern "C" int PMPI_Bcast(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);

int MPI_Bcast(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
{
    ++Number_of_bcasts;
    return PMPI_Bcast(buffer, count, datatype, root, comm);
}

int get_number_of_bcasts()
{
    return Number_of_bcasts;
}

/* ^^^ End of the MPI-correctnes checker code  ^^^ */


/* Implementation of I/I-based communication class (see header for docs) */

Mpi_guard::Mpi_guard(int master, const std::string& basename) : master_(master), sig_fname_base_(basename) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs_);
        
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

bool Mpi_guard::check_sig_files_ok(int number_of_bcasts) {
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
