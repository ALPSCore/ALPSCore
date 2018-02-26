/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** Support for MPI utility testing */

#include <mpi.h>

// Intercept MPI_Abort to verify it was indeed called
static int Mpi_abort_called=0;
/// This function detects MPI_Abort() call
extern "C" int MPI_Abort(MPI_Comm /*comm*/, int /*rc*/)
{
    ++Mpi_abort_called;
    return 0;
}

int get_mpi_abort_called() {
  return Mpi_abort_called;
}
