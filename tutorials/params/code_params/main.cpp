/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <iostream>
#include "alps/params.hpp"

/**
 * This example shows how to prepare parameters within the code itself.
 * Command-line parameters are ignored.
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
    
    // You can read the parameter value and use it to set other parameters.
    if (par["count"].as<int>() > 100) {
        par["high"] = true;
    } else {
        par["high"] = false;
    }

    // Printing parameter to standard output.
    std::cout << "Parameter values" << std::endl;    
    std::cout << "----------------" << std::endl;    
    std::cout << par;
    return 0;
}
