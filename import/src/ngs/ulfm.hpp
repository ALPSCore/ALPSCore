/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2012-2013 Donjan Rodic <drodic@phys.ethz.ch>                      *
 *                                                                                 *
 * This software is part of the ALPS libraries, published under the ALPS           *
 * Library License; you can use, redistribute it and/or modify it under            *
 * the terms of the license, either version 1 or (at your option) any later        *
 * version.                                                                        *
 *                                                                                 *
 * You should have received a copy of the ALPS Library License along with          *
 * the ALPS Libraries; see the file LICENSE.txt. If not, the license is also       *
 * available from http://alps.comp-phys.org/.                                      *
 *                                                                                 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        *
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT       *
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE       *
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,     *
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER     *
 * DEALINGS IN THE SOFTWARE.                                                       *
 *                                                                                 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
 
 
#ifndef ALPS_NGS_ULFM_HPP
#define ALPS_NGS_ULFM_HPP

#include <alps/ngs/config.hpp>
#include <alps/ngs/boost_mpi.hpp>
#include <boost/function.hpp>

#include "mpi-ext.h"
// TODO: mpi-ext.h should go into ALPS
// TODO: c++11 implementation of repeat_if_ranks_fail


namespace alps {
    namespace ngs {


/**
*  @brief Duplicate communicator uniformly, fault-tolerant.
*
*  This routine duplicates a communicator in a fault-tolerant manner
*  It guarantees that all processes have a consistent view of the
*  context and behave uniformly. If a broken (MPI_ERR_PROC_FAILED,
*  MPI_ERR_REVOKED) communicator gets passed, it will return a working
*  communicator for the alive peers.
*
*  This operation is usually performed on MPI_COMM_WORLD in order to
*  take ownership of the communicator and resize it when necessary.
*  The routine uses barriers and thus may slow down your program.
*
*  @param c The communicator to be duplicated.
*
*  @returns A copy of the passed communicator.
*/
ALPS_DECL boost::mpi::communicator duplicate_comm(MPI_Comm c);




class ulfm_result;

namespace detail {
    // Recovery of communicator: simply shrink.
    ulfm_result shrink_recover(boost::mpi::communicator & c, boost::mpi::exception e);
}

/**
*  @brief ULFM return object with failed ranks and translation table
*
*  This class is used to handle shrink recovery.
*/
class ulfm_result {
    public:
    friend ulfm_result detail::shrink_recover(boost::mpi::communicator & c, boost::mpi::exception e);

    ulfm_result() : oldrank_(-1), oldsize_(-1) {}

    /* Returns current ranks within recovered communicator */
    std::vector<int> get_failed() const { return failed_ranks_; }

    /* Returns translation between current communicator and last recovered one */
    std::map<int, int> get_translation() const { return translation_; };

    private:
    void store_failed(const MPI_Comm c);
    void store_new(const boost::mpi::communicator c);
    std::vector<int> failed_ranks_;
    std::map<int, int> translation_;
    int oldrank_;
    int oldsize_;
};

// Needs to be thrown by boost in case of MPI_ERR_PROC_FAILED or MPI_ERR_REVOKED -> proposal to Boost
class ulfm_exception : public boost::mpi::exception { };




/**
*  @brief Ignore a rank failure
*
*  Wrapper to ignore a MPI error of class MPI_ERR_PROC_FAILED or MPI_ERR_REVOKED
*/
#define ULFM_IGNORE_RANK_FAILURE_BEGIN                                          \
    try {

#define ULFM_IGNORE_RANK_FAILURE_END                                            \
    } catch(boost::mpi::exception e/*TODO: ulfm_exception*/) {                  \
        if(   e.error_class() != MPI_ERR_PROC_FAILED                            \
           && e.error_class() != MPI_ERR_REVOKED)                               \
            throw;                                                              \
    }

/**
*  @brief Check/trigger for rank failure
*
*  Wrapper to check for a MPI error of class MPI_ERR_PROC_FAILED or
*  MPI_ERR_REVOKED (slow!)
*/
#define ULFM_CHECK_RANK_FAILURE_BEGIN                                           \
        ULFM_IGNORE_RANK_FAILURE_BEGIN

#define ULFM_CHECK_RANK_FAILURE_END(communicator)                               \
        ULFM_IGNORE_RANK_FAILURE_END                                            \
        communicator.barrier(); /* trigger on barrier */                        



/**
*  @brief Executes a MPI function fault tolerantly.
*
*  Wrapper to indefinitely call the shrink recovery on a MPI error of class
*  MPI_ERR_PROC_FAILED or MPI_ERR_REVOKED.
*  The given function is executed and, in case of failure, the communicator is
*  recreated from those nodes that are still alive. The enclosed function(s)
*  must either track it's state externally or be stateless. The reason is that
*  some nodes may run the function an arbitrary number of times while others are
*  trying to recover: they are simply called again after an exception until
*  every rank has at least once received a valid return.
*/
#define ULFM_REPEAT_IF_RANKS_FAIL_BEGIN                                         \
    while(true) {                                                               \
        try {                                                                   \
            ULFM_CHECK_RANK_FAILURE_BEGIN

#define ULFM_REPEAT_IF_RANKS_FAIL_END(communicator)                             \
            ULFM_CHECK_RANK_FAILURE_END(communicator)                           \
            break;                                                              \
        } catch(boost::mpi::exception e/*TODO: ulfm_exception*/) {              \
            ngs::ulfm_result r = ngs::detail::shrink_recover(communicator, e);  \
                                                                                \
            if(communicator.rank() == 0) {                                      \
                std::vector<int> f = r.get_failed();                            \
                std::cerr << "Warning: failed ranks:";                          \
                for(std::vector<int>::iterator it=f.begin(); it!=f.end(); ++it) \
                    std::cerr << " " << *it;                                    \
                std::cerr << std::endl;                                         \
                                                                                \
                std::map<int, int> t = r.get_translation();                     \
                std::cerr << "Rank map:\n";                                     \
                for(std::map<int,int>::iterator it=t.begin(); it!=t.end(); ++it)\
                    std::cerr << it->first << " -> " << it->second << std::endl;\
            }                                                                   \
        }                                                                       \
    }



    }
}

#endif  // ALPS_NGS_ULFM_HPP
