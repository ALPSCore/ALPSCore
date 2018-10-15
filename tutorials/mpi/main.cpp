/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <iostream>
#include <vector>
#include "alps/utilities/mpi.hpp"

int main(int argc, char **argv)
{
    namespace am=alps::mpi; // shortcut

    // Initialize MPI environment; this calls MPI_Init():
    am::environment env;

    // Get a communicator to work with (MPI_COMM_WORLD by default):
    am::communicator comm;

    // Access communicator size and the process rank in it:
    std::cout << "My rank is " << comm.rank()
              << " out of total " << comm.size() << " processes"
              << std::endl;

    if (comm.size()<2) {
        std::cerr << "This test needs at least 2 parallel processes\n";
        return 1; // graceful return will call MPI_Finalize()
    }

    // Use any MPI function as you see appropriate; e.g., send from rank=1 to root
    const int src_rank=1;
    const int dst_rank=0;

    const int myrank=comm.rank();
    std::vector<int> message={1,2,3,4,5};
    if (myrank==src_rank) {
        message=std::vector<int>({10,20,30,40,50}); // data for the message
        // 1) instead of message.front(), one can use any int value
        // 2) the communicator object is converted to the MPI communicator
        MPI_Send(message.data(), message.size(), am::get_mpi_datatype(message.front()),
                 dst_rank, 0, comm);

        std::cout << "Rank " << myrank << " has sent\n";
    } else if (myrank==dst_rank) {
        // 1) instead of message.front(), one can use any int value
        // 2) the communicator object is converted to the MPI communicator
        MPI_Recv(message.data(), message.size(), am::get_mpi_datatype(message.front()),
                 src_rank, 0, comm, MPI_STATUS_IGNORE);

        // Checking that it worked:
        if (message == std::vector<int>({10,20,30,40,50})) {
            std::cout << "Message received successfully" << std::endl;
        } else {
            std::cout << "The message was not received correctly!" << std::endl;
        }
    }

    return 0; // MPI_Finalize() will be called automatically
}
