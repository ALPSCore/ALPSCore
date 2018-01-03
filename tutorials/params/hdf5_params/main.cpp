/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <iostream>
#include "alps/params.hpp"

/**
 * This example shows how to read/write parameters from/to an .hdf5 file.
 * This format is meant to save/restore computation, and not for long
 * term storage. That is: the format may change with the version of ALPSCore,
 * and may not be forward/backward compatible.
 * 
 * @param argc the number of arguments
 * @param argv the argument array
 * @return the exit code
 */
int main(int argc, char** argv)
{
    // Creates an instance of the parameter class.
    std::cout << "Creating parameter object..." << std::endl;
    alps::params par;    
    
    
    // Since we are not interested in providing command-line help (command-line
    // arguments are ignored) we simply add the parameter values. Note that
    // the "help" parameter is already defined and is initialized to false.
    std::cout << "Defining parameters..." << std::endl;
    par["count"] = 120;
    par["val"] = 2.71;
    par["name"] = "Endurance";
    
    // Printing parameters to standard output.
    std::cout << "Parameter values" << std::endl;    
    std::cout << "----------------" << std::endl;    
    std::cout << par;
    

    // Saving parameters to an hdf5 file.
    // Note the file is opened with write permission.
    std::cout << "Writing parameters to parameters.h5..." << std::endl;
    std::string filename("parameters.h5");
    alps::hdf5::archive oar(filename, "w");
    oar["params"] << par;
    oar.close();

    // Loading parameters from the hdf5 file.
    // Note the file is opened with read-only permission.
    // To prevent problems, no process should be writing to the file while we
    // are reading.
    alps::params par2;
    alps::hdf5::archive iar(filename, "r");
    iar["params"] >> par2;
    iar.close();
    
    // Printing loaded parameters to standard output.
    // These should match the previous parameters.
    std::cout << "Loaded parameter values" << std::endl;    
    std::cout << "-----------------------" << std::endl;    
    std::cout << par2;
    return 0;
}
