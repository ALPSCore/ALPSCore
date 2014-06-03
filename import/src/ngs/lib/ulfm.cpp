/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */


#include <alps/ngs/ulfm.hpp>


namespace alps {
    namespace ngs {

#ifdef ALPS_NGS_OPENMPI_ULFM


boost::mpi::communicator duplicate_comm(MPI_Comm comm) {
/*
INTERNAL:

The goal is to make sure every alive process completes the duplication
uniformly, so that we can take full ownership of the communicator (i.e. a copy
of COMM_WORLD) in subsequent recovery routines.

The barrier is non-uniform in it's completion (unlocks after all peers have
entered). If we know that the second barrier call is locally successful, all
peers in the current group have completed the duplication, but we don't know if
they have completed the barrier operation yet.

If the barrier throws, it is indistinguishable whether the barrier failed or
the communicator was revoked in the preceding step. We simply revoke the new
communicator and retry until all alive processes enter the barrier successfully.
*/

    boost::mpi::communicator c;
    while(true) {

        try {
            MPI_Barrier(comm);
            c = boost::mpi::communicator(comm, boost::mpi::comm_duplicate);
        } catch(boost::mpi::exception e) {
            if(e.error_class() != MPI_ERR_PROC_FAILED && e.error_class() != MPI_ERR_REVOKED)
                throw;
            MPI_Comm new_comm;
            OMPI_Comm_revoke(comm);
            OMPI_Comm_shrink(comm, &new_comm);
            MPI_Comm_free(&comm);
            comm = new_comm;
            continue;
        }

        try {
            c.barrier();
            break;
        } catch(boost::mpi::exception e) {
            if(e.error_class() != MPI_ERR_PROC_FAILED && e.error_class() != MPI_ERR_REVOKED)
                throw;
            OMPI_Comm_revoke(MPI_Comm(c));
        }

    }
    c = boost::mpi::communicator(comm, boost::mpi::comm_duplicate);
    return c;
}



void ulfm_result::store_failed(const MPI_Comm comm) {
/*
INTERNAL:

Store the failed ranks and old comm info for store_new().

Has to be passed the old communicator.

We can't use boost::mpi::communicator as parameter because we had an MPI error,
i.e. comm is broken!
*/

    // TODO: fixable in boost? can't use boost::mpi::communicator because it throws here
    //~ boost::mpi::communicator c = boost::mpi::communicator(comm, boost::mpi::comm_duplicate);
    //~ boost::mpi::group cgroup = c.group();
    MPI_Group c_group;
    MPI_Comm_group(comm, &c_group);
    boost::mpi::group cgroup = boost::mpi::group(c_group, false);  // whole failed communicator group
    MPI_Group_free(&c_group);

    OMPI_Comm_failure_ack(comm);
    MPI_Group ack_group;
    OMPI_Comm_failure_get_acked(comm, &ack_group);
    boost::mpi::group fgroup(ack_group, false);  // acknowledged failed communicator group

    std::vector<int> franks, cranks(fgroup.size());
    for( int i = 0; i < fgroup.size(); ++i) {
        franks.push_back(i);
    }

    fgroup.translate_ranks(franks.begin(), franks.end(), cgroup, cranks.begin());

    for(int i = 0; i < fgroup.size(); ++i)
        failed_ranks_.push_back(cranks[i]);

    MPI_Comm_rank(comm, &oldrank_);
    MPI_Comm_size(comm, &oldsize_);
}

void ulfm_result::store_new(const boost::mpi::communicator comm) {
/*
INTERNAL:

Store the translation table which maps the old ranks to the new ones.

Has to be passed the new communicator.

This function assumes that the all_gather output is sorted, i.e. the MPI
shrinking operation keeps the rank order (as OpenMPI ULFM does). If this is not
the case, apply a sort after the all_gather.
*/

    std::vector<int> ranks;
    boost::mpi::all_gather(comm, oldrank_, ranks);
        
    for(int i = 0, n = 0; n < oldsize_; ++n)
        if(i >= ranks.size() || n != ranks[i])
            translation_.insert(std::make_pair(n, -1));
        else
            translation_.insert(std::make_pair(n, i++));
}

namespace detail {
    ulfm_result shrink_recover(boost::mpi::communicator & c, boost::mpi::exception e) {
        ulfm_result res;
        res.store_failed(c);

        MPI_Comm new_comm;
        if(e.error_class() == MPI_ERR_PROC_FAILED)
            OMPI_Comm_revoke(MPI_Comm(c));                 // not collective
        OMPI_Comm_shrink(MPI_Comm(c), &new_comm);          // collective
        c = boost::mpi::communicator(new_comm, boost::mpi::comm_take_ownership);

        res.store_new(c);
        return res;
    }
}

#else // ALPS_NGS_OPENMPI_ULFM

boost::mpi::communicator duplicate_comm(MPI_Comm c) {
    return boost::mpi::communicator(c, boost::mpi::comm_duplicate);
}

namespace detail {
    ulfm_result shrink_recover(boost::mpi::communicator & c, boost::mpi::exception e) {
        return ulfm_result();
    }
}

#endif // ALPS_NGS_OPENMPI_ULFM

    }
}
