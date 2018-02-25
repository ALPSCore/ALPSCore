/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <alps/alea/core.hpp>
#include <alps/utilities/mpi.hpp>     /* provides mpi.h */

// TODO: merge into MPI
namespace alps { namespace mpi {

struct failed_operation : std::exception { };

inline void checked(int retcode)
{
    if (retcode != MPI_SUCCESS)
        throw failed_operation();
}

inline bool is_intercomm(const communicator &comm)
{
    int flag;
    checked(MPI_Comm_test_inter(comm, &flag));
    return flag;
}

}}

namespace alps { namespace alea {

namespace mpi = alps::mpi;

/**
 * In-place sum-reduction via an MPI communicator.
 */
struct mpi_reducer
    : public reducer
{

    mpi_reducer(const mpi::communicator &comm=mpi::communicator(), int root=0)
        : comm_(comm)
        , root_(root)
    {
        if (mpi::is_intercomm(comm))
            throw std::runtime_error("Unable to use in-place communication");
    }

    reducer_setup get_setup() const override
    {
        reducer_setup mpi_setup = { (size_t) comm_.rank(),
                                    (size_t) comm_.size(),
                                    am_root() };
        return mpi_setup;
    }

    long get_max(long data) const override
    {
        mpi::checked(MPI_Allreduce(MPI_IN_PLACE, &data, 1, MPI_LONG,
                                   MPI_MAX, comm_));
        return data;
    }

    void reduce(view<double> data) const override { inplace_reduce(data); }

    void reduce(view<long> data) const override { inplace_reduce(data); }

    void commit() const override { }

    const mpi::communicator &comm() const { return comm_; }

    int root() const { return root_; }

    bool am_root() const { return comm_.rank() == root_; }

protected:
    template <typename T>
    void inplace_reduce(view<T> data) const
    {
        // NO-OP in the case of empty data (strange though)
        if (data.size() == 0)
            return;

        // Extract data type and get on with it
        MPI_Datatype dtype_tag = alps::mpi::get_mpi_datatype(T());

        // In-place requires special value for sendbuf, but only on root
        const void *sendbuf = am_root() ? MPI_IN_PLACE : data.data();
        // To maintain MPI-2.0 compatibility `const` modifier should be removed
        mpi::checked(MPI_Reduce(const_cast<void*>(sendbuf), data.data(), data.size(), dtype_tag,
                                MPI_SUM, root_, comm_));
    }

private:
    alps::mpi::communicator comm_;
    int root_;
};

}}
