/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <iostream>
#include "alps/params.hpp"

/**
 * This example shows how to read and validate parameters from command-line arguments.
 * The supported parameters are:
 * <ul>
 *   <li>--count   integer, default 0</li>
 *   <li>--val     double, default 6.28</li>
 *   <li>--name    std:string, no default, mandatory</li>
 * </ul>
 * <p>
 * Run the example with different arguments combinations. For example:
 * <ul>
 *   <li>./cmd_params --name="Superman"</li>
 *   <li>./cmd_params </li>
 *   <li>./cmd_params --count=-100  </li>
 *   <li>./cmd_params --count=200 --val=2.71 --name="Luthor" </li>
 * </ul>
 * You can omit dashes in front of arguments, as long there is an "=" sign:
 * <ul>
 *   <li>./cmd_params name="Superman"</li>
 *   <li>./cmd_params </li>
 *   <li>./cmd_params count=-100  </li>
 *   <li>./cmd_params count=200 val=2.71 name="Luthor" </li>
 * </ul>
 *
 * @param argc the number of arguments
 * @param argv the argument array
 * @return the exit code
 */
int main(int argc, char** argv)
{
    // Creates an instance of the parameter class, using the arguments to
    // initialize the values.
    std::cout << "Creating parameter object..." << std::endl;
    alps::params par(argc, argv);


    // Here we define all the parameters we are interested in, giving their
    // names, default values, and descriptions. By default, only the "help"
    // parameter is defined, and is initialized to false.
    //
    // If a command-line argument matches "--parameterName=..." then the parameter
    // is set to the value; if no command-line argument matches, the default
    // value is used. Command-line arguments that do not match any parameter are ignored.
    std::cout << "Defining parameters..." << std::endl;
    par.define<int>("count", 100, "Number of interconnected elements");
    par.define<double>("val", 6.28, "Value of implosion constant");
    par.define<std::string>("name", "Name of de-construction algorithm");

    // Parameter validation. We check each condition. If not met, we print an
    // error and flag the help to be printed later.
    // The "name" parameter must be present
    if (!par.exists("name")) {
        std::cout << "You must provide the name of the de-construction algorithm" << std::endl;
        par["help"] = true;
    }
    // This will validate that count and val have been given the proper datatype
    else if (par.has_missing(std::cout)) {
        par["help"] = true;
        par.help_requested(std::cout);
        return 1;
    }
    // The "count" parameter must be greater than zero
    if (par["count"].as<int>() <= 0) {
        std::cout << "The number of interconnected elements must be positive" << std::endl;
        par["help"] = true;
    }

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
