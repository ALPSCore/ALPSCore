/*
 * Copyright (C) 1998-2016 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include "ising.hpp"

#include <alps/params/convenience_params.hpp>
#include <boost/lambda/lambda.hpp>

// Defines the parameters for the ising simulation
void ising_sim::define_parameters(parameters_type & parameters) {
    // If the parameters are restored, they are already defined
    if (parameters.is_restored()) {
        return;
    }
    
    // Adds the parameters of the base class
    alps::mcbase::define_parameters(parameters);
    // Adds the convenience parameters (for save/load)
    // followed by the ising specific parameters
    alps::define_convenience_parameters(parameters)
        .description("2D ising simulation")
        .define<int>("length", 50, "size of the periodic box")
        .define<int>("sweeps", 10000, "maximum number of sweeps")
        .define<int>("thermalization", "number of sweeps for thermalization")
        .define<double>("temperature", "temperature of the system");
}

// Creates a new simulation.
// We always need the parameters and the seed as we need to pass it to
// the alps::mcbase constructor. We also initialize our internal state,
// mainly using values from the parameters.
ising_sim::ising_sim(parameters_type const & parms, std::size_t seed_offset)
    : alps::mcbase(parms, seed_offset)
    , length(parameters["length"])
    , sweeps(0)
    , thermalization_sweeps(int(parameters["thermalization"]))
    , total_sweeps(parameters["sweeps"])
    , beta(1. / parameters["temperature"].as<double>())
    , spins(length,length)
{
    // Initializes the spins
    for(int i=0; i<length; ++i) {
        for (int j=0; j<length; ++j) {
            spins(i,j) = (random() < 0.5 ? 1 : -1);
        }
    }
    
    // Adds the measurements
    measurements
        << alps::accumulators::FullBinningAccumulator<double>("Energy")
        << alps::accumulators::FullBinningAccumulator<double>("Magnetization")
        << alps::accumulators::FullBinningAccumulator<double>("Magnetization^2")
        << alps::accumulators::FullBinningAccumulator<double>("Magnetization^4")
        << alps::accumulators::FullBinningAccumulator<std::vector<double> >("Correlations1")
        << alps::accumulators::FullBinningAccumulator<std::vector<double> >("Correlations2")
        << alps::accumulators::FullBinningAccumulator<std::vector<double> >("Correlations3")
        ;
}

// Performs the calculation at each MC step;
// decides if the step is accepted.
void ising_sim::update() {
    using std::exp;
    typedef unsigned int uint;
    // Choose a spin to flip:
    uint i = int(length * random());
    uint j = int(length * random());
    // Find neighbors indices, with wrap over box boundaries:
    uint i1 = (i+1) % length;            // left
    uint i2 = (i-1+length) % length;     // right
    uint j1 = (j+1) % length;            // up
    uint j2 = (j-1+length) % length;     // down
    // Energy difference:
    double delta=2.*spins(i,j)*
                 (spins(i1,j1)+  // left
                  spins(i2,j)+  // right
                  spins(i,j1)+  // up
                  spins(i,j2)); // down
    // Step acceptance:
    if (delta<=0. || random() < exp(-beta*delta)) {
        // flip the spin
        spins(i,j) = -spins(i,j);
    }
}

// Collects the measurements at each MC step.
void ising_sim::measure() {
    sweeps++;
    if (sweeps<thermalization_sweeps) return;
    
    double tmag = 0; // magnetization
    double ten = 0; // energy
    // FIXME: all 3 correlations must converge to the same?
    std::vector<double> corr_v(length); // "vertical"
    std::vector<double> corr_h(length); // "horizontal"
    std::vector<double> corr_d(length); // "diagonal"
    
    for (int i=0; i<length; ++i) {
        for (int j=0; j<length; ++j) {
            tmag += spins(i,j);
            int i_next=(i+1)%length;
            int j_next=(j+1)%length;
            ten += -(spins(i,j)*spins(i,j_next)+
                     spins(i,j)*spins(i_next,j));
            
            for (int d = 0; d < length; ++d) {
                int i_pair=(i+d)%length;
                int j_pair=(j+d)%length;
                corr_h[d] += spins(i,j)*spins(i_pair,j);
                corr_v[d] += spins(i,j)*spins(i,j_pair);
                corr_d[d] += spins(i,j)*spins(i_pair,j_pair);
            }
        }
    }
    // pull in operator/ for vectors
    using alps::numeric::operator/;
    const double l2=length*length;
    corr_h = corr_h / l2;
    corr_v = corr_v / l2;
    corr_d = corr_d / l2;
    ten /= l2;
    tmag /= l2;

    // Accumulate the data
    measurements["Energy"] << ten;
    measurements["Magnetization"] << tmag;
    measurements["Magnetization^2"] << tmag*tmag;
    measurements["Magnetization^4"] << tmag*tmag*tmag*tmag;
    measurements["Correlations1"] << corr_h;
    measurements["Correlations2"] << corr_v;
    measurements["Correlations3"] << corr_d;
}

// Returns a number between 0.0 and 1.0 with the completion percentage
double ising_sim::fraction_completed() const {
    double f=0;
    if (sweeps >= thermalization_sweeps) {
        f=(sweeps-thermalization_sweeps)/double(total_sweeps);
    }
    return f;
}

// Saves the state to the hdf5 file
void ising_sim::save(alps::hdf5::archive & ar) const {
    // Most of the save logic is already implemented in the base class
    alps::mcbase::save(ar);
    
    // We just need to add our own internal state
    ar["checkpoint/sweeps"] << sweeps;
    ar["checkpoint/spins"] << spins;
    // The rest of the internal state is saved as part of the parameters
}

// Loads the state from the hdf5 file
void ising_sim::load(alps::hdf5::archive & ar) {
    // Most of the load logic is already implemented in the base class
    alps::mcbase::load(ar);

    // Restore the internal state that came from parameters
    length = parameters["length"];
    thermalization_sweeps = parameters["thermalization"];
    total_sweeps = parameters["sweeps"];
    beta = 1. / parameters["temperature"].as<double>();

    // Restore the rest of the state from the hdf5 file
    ar["checkpoint/sweeps"] >> sweeps;
    ar["checkpoint/spins"] >> spins;
}
