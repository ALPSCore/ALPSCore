/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
int get_mpi_abort_called();

static ::testing::AssertionResult mpi_is_up()
{
    int ini;
    MPI_Initialized(&ini);
    if (ini) {
        return ::testing::AssertionSuccess() << "MPI is initialized";
    } else {
        return ::testing::AssertionFailure() << "MPI is not yet initialized";
    }
}

static ::testing::AssertionResult mpi_is_down()
{
    int fin;
    MPI_Finalized(&fin);
    if (fin) {
        return ::testing::AssertionSuccess() << "MPI is finalized";
    } else {
        return ::testing::AssertionFailure() << "MPI is not yet finalized";
    }
}
