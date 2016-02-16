/*
 * Copyright (C) 1998-2016 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <iostream>
#include "alps/params.hpp"

/**
 * This example shows how to read parameters from a .ini file.
 * The file is passed through the first command-line argument.
 * A couple of example .ini files are provided in this directory.
 * The parameters can also be overridden using command-line options.
 * Run the example with different arguments combinations. For example:
 * <ul>
 *   <li>./ini_params configurationA.ini</li>
 *   <li>./ini_params configurationA.ini --count 3</li>
 *   <li>./ini_params configurationB.ini</li>
 * </ul>
 * 
 * @param argc the number of arguments
 * @param argv the argument array
 * @return the exit code
 */
int main(int argc, const char* argv[])
{
    // Creates an instance of the parameter class, using the arguments to
    // initialize the values.
    std::cout << "Creating parameter object..." << std::endl;
    alps::params par(argc, argv);
    
    
    // Here we define all the parameters we are interested in, giving their
    // names, default values, and descriptions. By default, only the "help" 
    // parameter is defined, and is initialized to false.
    //
    // If an ini file entry matches "parameterName=Xxx" then the parameter 
    // is set to the value; if no ini file entry matches, the default
    // value is used. Command-line options will override the ini file entry.
    std::cout << "Defining parameters..." << std::endl;
    par.define<int>("count", 0, "Number of interconnected elements");
    par.define<double>("val", 6.28, "Value of implosion constant");
    par.define<std::string>("name", "Judas", "Name of de-construction algorithm");
    
    // If requested, we print the help message, which is constructed from the
    // information we gave when defining the parameters.
    if (par.help_requested(std::cout)) {
        return 0;
    }

    // Printing parameter to standard output.
    std::cout << "Parameter values" << std::endl;    
    std::cout << "----------------" << std::endl;    
    std::cout << par;
    return 0;
}
