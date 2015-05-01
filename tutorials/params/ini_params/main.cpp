/** @file main.cpp
    @brief alps::params cmd_params
*/

#include <iostream>
#include "alps/params.hpp"

/**
 * This example shows how to read parameters from command-line arguments.
 * The supported parameters are:
 * <ul>
 *   <li>--count   integer, default 0</li>
 *   <li>--val     double, default 6.28</li>
 *   <li>--name    std:string, default "Judas"</li>
 * </ul>
 * <p>
 * Run the example with different arguments combinations. For example:
 * <ul>
 *   <li>./cmd_params</li>
 *   <li>./cmd_params --count 3 --val 2.71 --name "Superman"</li>
 *   <li>./cmd_params --count 100 --val 2.71 </li>
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
    // If a command-line argument matches "--parameterName" then the parameter 
    // is set to the value; if no command-line argument matches, the default
    // value is used. Command-line arguments that do not match any parameter are ignored.
    std::cout << "Defining parameters..." << std::endl;
    par.define<int>("count", 0, "Number of interconnected elements");
    par.define<double>("val", 6.28, "Value of implosion constant");
    par.define<std::string>("name", "Judas", "Name of de-construction algorithm");
    
    // If request, we print the help message, which is constructed from the
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
