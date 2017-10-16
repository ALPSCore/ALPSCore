/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <alps/alea/core.hpp>
#include <alps/utilities/mpi.hpp>     /* provides mpi.h */

namespace alps { namespace alea {

// TODO: merge into MPI
namespace mpi {

struct failed_operation : std::exception { };

void checked(int retcode)
{
    if (retcode != MPI_SUCCESS)
        throw failed_operation();
}

bool is_intercomm(const alps::mpi::communicator &comm)
{
    int flag;
    checked(MPI_Comm_test_inter(comm, &flag));
    return flag;
}

}

struct mpi_reducer
    : public reducer
{
    mpi_reducer(const alps::mpi::communicator &comm, const int root)
        : comm_(&comm)
        , root_(root)
    {
        if (mpi::is_intercomm(comm))
            throw std::runtime_error("Unable to use in-place communication");
    }

    reducer::setup begin()
    {
        reducer::setup mpi_setup = { (size_t) comm_->rank(),
                                     (size_t) comm_->size(),
                                     comm_->rank() == root_ };
        return mpi_setup;
    }

    void reduce(sink<double> data) { inplace_reduce(data); }

    void reduce(sink<long> data) { inplace_reduce(data); }

    void commit() { }

protected:
    template <typename T>
    void inplace_reduce(sink<T> data)
    {
        // NO-OP in the case of empty data (strange though)
        if (data.size() == 0)
            return;

        // Extract data type and get on with it
        MPI_Datatype dtype_tag = alps::mpi::get_mpi_datatype(T());
        mpi::checked(MPI_Reduce(
                MPI_IN_PLACE, data.data(), data.size(), dtype_tag,
                MPI_SUM, root_, *comm_));
    }

private:
    const alps::mpi::communicator *comm_;
    int root_;
};


}}
