/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file mpi_guard.hpp
    @brief A simple DIY MPI correctness checker (header file)
*/

#ifndef ALPS_TESTS_MPI_GUARD_HPP_d1f0c4691e714f40be03d55dcfdffcd3
#define ALPS_TESTS_MPI_GUARD_HPP_d1f0c4691e714f40be03d55dcfdffcd3

#include <string>
#include <mpi.h>

extern int get_number_of_bcasts();

/// This class implements simple DIY file-based, MPI-independent communication code.
/** The intent is to check the number of bcast operations.
    As MPI may be in an incorrect state in the case of the mismatch,
    we rely exclusively on file I/O
    (inefficient, but simple, and uses only basic C/C++ file operations).
 */
class Mpi_guard {
    int master_, rank_, nprocs_;
    std::string sig_fname_base_, sig_fname_master_, sig_fname_;
  public:

    /// Construct the object and create necessary files. MPI is assumed to be working.
    Mpi_guard(int master, const std::string& base_file_name);
    
    // Check the number of broadcasts across processes. Do not rely on MPI. Return `true` if OK.
    bool check_sig_files_ok(int number_of_bcasts);

    // Return rank (convenience method)
    int rank() { return rank_; }
};

#endif /* ALPS_TESTS_MPI_GUARD_HPP_d1f0c4691e714f40be03d55dcfdffcd3 */
