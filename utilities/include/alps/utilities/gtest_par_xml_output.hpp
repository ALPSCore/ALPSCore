/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_UTILITY_GTEST_PAR_XML_OUTPUT_HPP
#define ALPS_UTILITY_GTEST_PAR_XML_OUTPUT_HPP

#include <alps/config.hpp>

namespace alps {
    /** @brief Tweaks (argc,argv) to redirect GTest XML output to different files.
        @param irank : A number added to XML output file name (usually, MPI rank)
        @param argc : main()'s argc
        @param argv : main()'s argv

        This function scans argv[] for '--gtest=xml' argument and tweaks it by adding irank
        to the output file name (preserving the last extension, if any). Its intended use
        is for unit tests that use Google Test and are run in parallel via MPI. Suggested use:

        int main(int argc, char**argv) {
        boost::mpi::environment env(argc, argv);
        gtest_par_xml_output(argc, argv, boost::mpi::communicator().rank() );
        ::testing::InitGoogleTest(&argc, argv);
        // .....
        }
    */

    ALPS_DECL void gtest_par_xml_output(unsigned int irank, int argc, char** argv);
}
#endif // ALPS_UTILITY_GTEST_PAR_XML_OUTPUT_HPP
