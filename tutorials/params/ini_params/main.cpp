/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <iostream>
#include "alps/params.hpp"

/**
 * This example shows how to read parameters from a .ini file.
 * The file is passed through the first command-line argument.
 * A couple of example .ini files are provided in this directory.
 * A .ini file can have sections.
 * The parameters can also be overridden using command-line options.
 * Run the example with different arguments combinations. For example:
 * <ul>
 *   <li>./ini_params configurationA.ini</li>
 *   <li>./ini_params configurationA.ini count=3</li>
 *   <li>./ini_params configurationB.ini</li>
 *   <li>./ini_params configurationA.ini count=3 configurationB.ini</li>
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
    // If a parameter is defined without a default value, the
    // parameter must be supplied (either as an ini file entry or as a
    // command line argument).
    //
    // If an ini file entry matches "parameterName=Xxx" then the parameter
    // is set to the value; if no ini file entry matches, the default
    // value is used. Command-line options will override the ini file entry.
    //
    // An ini file can contain [sections]. A parameter name `name` in
    // a section `[section]` is referred to as `section.name`.
    //
    std::cout << "Defining parameters..." << std::endl;
    // no sections:
    par .define<int>("count", 0, "Number of interconnected elements")
        .define<double>("val", 6.28, "Value of implosion constant")
        .define<std::string>("name", "Judas", "Name of de-construction algorithm");
    // inside a section [other]:
    par .define<int>("other.count", -1, "Number of the other interconnected elements")
        .define<double>("other.val", 1.25, "Value of the other implosion constant")
        .define<std::string>("other.name", "Jack", "Name of the other de-construction algorithm");
    // a parameter with a required value:
    par.define<std::string>("user", "User name");
    // a "switch" parameter (either it is present or not):
    par.define("verbose","Be verbose");

    // If requested, we print the help message, which is constructed from the
    // information we gave when defining the parameters.
    if (par.help_requested(std::cout)) {
        return 0;
    }

    // We can also check if there are any parameters that do not have
    // a value assigned or were given as an invalid format; if so, a short message will
    // be printed to the given stream.
    if (par.has_missing(std::cout)) {
        return 1;
    }

    // Using the parameters.
    // Type checking and conversion happens automatically, when possible.
    std::string user=par["user"];
    std::cout << "Hello, " << user << "!\n";
    if (par["verbose"]) {
        std::cout << "You asked me to be verbose.\n";
    }
    std::cout << "Your parameters are:\n";
    int count=par["count"];
    double val=par["val"];
    std::cout << "count=" << count << " val=" << val;
    // Here C++ cannot guess the type to convert to, we have to help:
    std::cout << " name=" << par["name"].as<std::string>()
              << std::endl;

    // Parameters from the other section have `other.` prefix.
    // Also, we are using explicit type conversion.
    std::cout << "Your other parameters are:\n"
              << "count=" << int(par["other.count"])
              << " val="  << double(par["other.val"])
             // but calling std::string() would be ambiguous, we have to use .as<...>(...)
              << " name=" << par["other.name"].as<std::string>()
              << std::endl;

    // We can also print parameters to standard output.
    std::cout << "\nAll parameter values" << std::endl;
    std::cout << "----------------" << std::endl;
    std::cout << par;
    return 0;
}
