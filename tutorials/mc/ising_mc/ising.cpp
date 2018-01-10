/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
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
        .description("1D ising simulation")
        .define<int>("length", 50, "lenth of the periodic ising chain")
        .define<int>("sweeps", 1000, "maximum number of sweeps")
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
    , total_sweeps(int(parameters["sweeps"]))
    , beta(1. / double(parameters["temperature"]))
    , spins(length)
{
    // Initializes the spin vector
    for(int i = 0; i < length; ++i) {
        spins[i] = (random() < 0.5 ? 1 : -1);
    }
    
    // Adds the measurements
    measurements
        << alps::accumulators::FullBinningAccumulator<double>("Energy")
        << alps::accumulators::FullBinningAccumulator<double>("Magnetization")
        << alps::accumulators::FullBinningAccumulator<double>("Magnetization^2")
        << alps::accumulators::FullBinningAccumulator<double>("Magnetization^4")
        << alps::accumulators::FullBinningAccumulator<std::vector<double> >("Correlations")
    ;
}

// Performs the calculation at each MC step.
void ising_sim::update() {
    for (int j = 0; j < length; ++j) {
        using std::exp;
        int i = int(double(length) * random());
        int right = ( i + 1 < length ? i + 1 : 0 );
        int left = ( i - 1 < 0 ? length - 1 : i - 1 );
        double p = exp( 2. * beta * spins[i] * ( spins[right] + spins[left] ));
        if ( p >= 1. || random() < p )
            spins[i] = -spins[i];
    }
}

// Collects the measurements at each MC step.
void ising_sim::measure() {
    sweeps++;
    if (sweeps > thermalization_sweeps) {
        double tmag = 0;
        double ten = 0;
        double sign = 1;
        std::vector<double> corr(length);
        for (int i = 0; i < length; ++i) {
            tmag += spins[i];
            sign *= spins[i];
            ten += -spins[i] * spins[ i + 1 < length ? i + 1 : 0 ];
            for (int d = 0; d < length; ++d)
                corr[d] += spins[i] * spins[( i + d ) % length ];
        }
        // pull in operator/ for vectors
        using alps::numeric::operator/;
        corr = corr / double(length);
        ten /= length;
        tmag /= length;
        measurements["Energy"] << ten;
        measurements["Magnetization"] << tmag;
        measurements["Magnetization^2"] << tmag * tmag;
        measurements["Magnetization^4"] << tmag * tmag * tmag * tmag;
        measurements["Correlations"] << corr;
    }
}

// Returns a number between 0.0 and 1.0 with the completion percentage
double ising_sim::fraction_completed() const {
    return (sweeps < thermalization_sweeps ? 0. : ( sweeps - thermalization_sweeps ) / double(total_sweeps));
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
    length = int(parameters["length"]);
    thermalization_sweeps = int(parameters["thermalization"]);
    total_sweeps = int(parameters["sweeps"]);
    beta = 1. / double(parameters["temperature"]);

    // Restore the rest of the state from the hdf5 file
    ar["checkpoint/sweeps"] >> sweeps;
    ar["checkpoint/spins"] >> spins;
}
