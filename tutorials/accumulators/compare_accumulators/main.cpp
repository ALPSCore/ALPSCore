/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <iostream>
#include <alps/accumulators.hpp>

void generateData(alps::accumulators::accumulator_set &data,
        const std::string &mean, const std::string &nobin,
        const std::string &logbin, const std::string &fullbin);

/**
 * This example shows the different in error estimation between the different
 * type of accumulators.
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

    // Extracts the results    
    alps::accumulators::result_set results(set);
    
    // Prints out the direct result from the accumulator.
    // The "mean" accumulator will have no error associated with it.
    // The "nobin" accumulator will have an error, but it will be underestimated.
    // The "logbin" accumulator will have the correct error, and the estimation
    //     becomes better with the bigger bin size.
    // The "fullbin" accumulator error estimation will be the same as the "logbin".
    std::cout << "Results from accumulators:" << std::endl;
    std::cout << "mean:    " << results[mean] << std::endl;
    std::cout << "nobin:   " << results[nobin] << std::endl;
    std::cout << "logbin:  " << results[logbin] << std::endl;
    std::cout << "fullbin: " << results[fullbin] << std::endl;
    
    // Prints out the result obtained by squaring the accumulator, after
    // the results are computed.
    // The "mean" accumulator will have no error associated with it.
    // The "nobin" accumulator will have an error, but it will be underestimated
    //     because of the auto-correlation and the computation.
    // The "logbin" accumulator will have an error, also underestimated but
    //     because of the computation only.
    // The "fullbin" accumulator will have the correct error.
    std::cout << "Results from squared accumulators:" << std::endl;
    std::cout << "mean:    " << results[mean] * results[mean] << std::endl;
    std::cout << "nobin:   " << results[nobin] * results[nobin] << std::endl;
    std::cout << "logbin:  " << results[logbin] * results[logbin] << std::endl;
    std::cout << "fullbin: " << results[fullbin] * results[fullbin] << std::endl;
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
