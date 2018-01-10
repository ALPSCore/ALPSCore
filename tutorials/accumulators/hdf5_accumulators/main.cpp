/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <iostream>
#include <alps/accumulators.hpp>
#include <alps/hdf5.hpp>

void generateData(alps::accumulators::accumulator_set &data,
        const std::string &mean, const std::string &nobin,
        const std::string &logbin, const std::string &fullbin);

/**
 * This example shows how to save are restore accumulators. This can be useful
 * when stopping and resuming a simulation.
 * 
 * @param argc the number of arguments
 * @param argv the argument array
 * @return the exit code
 */
int main(int argc, char** argv)
{
    // Create the empty set of accumulators
    std::cout << "Creating accumulator set..." << std::endl;
    alps::accumulators::accumulator_set set;

    // We will accumulate the same data with the 4 different types of
    // accumulators. We first define the labels for the accumulators.
    std::string fullbin = "fullbin";
    std::string logbin = "logbin";
    std::string nobin = "nobin";
    std::string mean = "mean";
    
    // We create the accumulators of the appropriate type and add them
    // to the set
    set << alps::accumulators::FullBinningAccumulator<double>(fullbin)
         << alps::accumulators::LogBinningAccumulator<double>(logbin)
         << alps::accumulators::NoBinningAccumulator<double>(nobin)
         << alps::accumulators::MeanAccumulator<double>(mean);

    // Generate and accumulate the data    
    std::cout << "Generating data..." << std::endl;
    generateData(set, mean, nobin, logbin, fullbin);
    
    // Printing the accumulator data
    std::cout << "Result:" << std::endl;
    std::cout << set << std::endl;
    
    // Saving all accumulators to an hdf5 file.
    // Note the file is opened with write permission.
    std::cout << "Writing accumulators to parameters.h5..." << std::endl;
    std::string filename("measurements.h5");
    alps::hdf5::archive oar(filename, "w");
    oar["measurements"] << set;
    oar.close();

    // Loading all accumulators from the hdf5 file.
    // Note the file is opened with read-only permission.
    // To prevent problems, no process should be writing to the file while we
    // are reading.
    std::cout << "Reading accumulators from parameters.h5..." << std::endl;
    alps::accumulators::accumulator_set set2;
    alps::hdf5::archive iar(filename, "r");
    iar["measurements"] >> set2;
    iar.close();
    
    // Printing the loaded data
    std::cout << "Loaded result:" << std::endl;
    std::cout << set2 << std::endl;
    
    return 0;
}

void generateData(alps::accumulators::accumulator_set &set,
        const std::string &mean, const std::string &nobin,
        const std::string &logbin, const std::string &fullbin) {
    double count = 1000;
    double i0 = 0; 
    double i1 = 0;
    double i2 = 0;
    double i3 = 0;
    for (double i = 0; i < count; ++i) {
        i0 = 0.5 * i1 + 0.3 * i2 + 0.1 * i3 + 0.1 * drand48();
        set[mean] << i0;
        set[nobin] << i0;
        set[logbin] << i0;
        set[fullbin] << i0;
        i3 = i2;
        i2 = i1;
        i1 = i0;
    }
}
