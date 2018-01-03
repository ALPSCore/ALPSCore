/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <iostream>
#include <alps/hdf5/archive.hpp>
#include <alps/accumulators.hpp>
#include <alps/params.hpp>

/**
   This tutorial is to demonstrate how to process results from a series 
   of MC calculations.

   The program takes a list of Ising MC output HDF5 files as arguments,
   and computes Binder cumulant from the data from each output file.
*/

int main(int argc, char** argv)
{
    if (argc<2) {
        std::cerr << "Usage: " << argv[0] << " ising_result1.h5 [ising_result2.h5 ...]\n";
        return 1;
    }

    // Define a couple of "shortcuts"
    typedef unsigned int uint;
    namespace aa=alps::accumulators;

    const uint npoints=argc-1; // number of data points available
    const char* const * fnames=argv+1; // array of file names to read

    // Print the header of the output file:
    std::cout << "# Temperature Binder_cumulant_mean Binder_cumulant_error" << std::endl;

    // Loop over all data points:
    for (uint ip=0; ip<npoints; ++ip) {
        try {
            // open the archive:
            alps::hdf5::archive ar(fnames[ip],"r");

            // read the simulation result set:
            aa::result_set results;
            ar["/simulation/results"] >> results;
        
            // read the simulation parameters:
            alps::params p;
            ar["/parameters"] >> p;

            // Extract the named results from the result set:
            const aa::result_wrapper& mag4=results["Magnetization^4"];
            const aa::result_wrapper& mag2=results["Magnetization^2"];

            // Compute Binder cumulant:
            aa::result_wrapper binder_cumulant=1-mag4/(3*mag2*mag2);

            // Output the results:
            std::cout << p["temperature"].as<double>() << " "
                      << binder_cumulant.mean<double>() << " "
                      << binder_cumulant.error<double>() << std::endl;
            
        } catch (const std::runtime_error& exc) {
            std::cerr << "Exception caught at point #" << ip
                      << ": " << exc.what()
                      << "\nSkipping the point." << std::endl;
        }
    } // end loop over points

    return 0;
}
