/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_UTILITY_GTEST_PAR_XML_OUTPUT_HPP
#define ALPS_UTILITY_GTEST_PAR_XML_OUTPUT_HPP

#include <vector>

#include "alps/config.hpp"

namespace alps {

    /** @brief Functor class to tweak (argc,argv).

        Its intended use is for unit tests that use Google Test and
        are run in parallel via MPI. Suggested use:

            int main(int argc, char**argv) {
              alps::mpi::environment env(argc, argv);
              gtest_par_xml_output tweak; // holds the memory
              tweak(argc, argv, alps::mpi::communicator().rank() ); // does tweaking
              ::testing::InitGoogleTest(&argc, argv);
              // .....
              // destructor releases the memory
            }

    */        
    class gtest_par_xml_output {
        std::vector<char*> keeper_; ///< Holds pointers to allocated argv[i]
        // Note: we could use shared_ptr<>, but why bring in yet another header for a simple task?
      public:
        virtual ~gtest_par_xml_output();
        
        /** @brief Tweaks (argc,argv) to redirect GTest XML output to different files.
            @param irank : A number added to XML output file name (usually, MPI rank)
            @param argc : main()'s argc
            @param argv : main()'s argv

            This method scans argv[] for '--gtest=xml' argument and tweaks it by adding irank
            to the output file name (preserving the last extension, if any).
        */
        void operator()(unsigned int irank, int argc, char** argv);
    };
}
#endif // ALPS_UTILITY_GTEST_PAR_XML_OUTPUT_HPP
