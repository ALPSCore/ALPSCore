/*
 * Copyright (C) 1998-2016 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include "ising.hpp"

#include <alps/params/convenience_params.hpp>
#include <boost/lambda/lambda.hpp>

double ising_sim::get_energy()
{
    double en=0;
    for (int i=0; i<length; ++i) {
        for (int j=0; j<length; ++j) {
            int i_next=(i+1)%length;
            int j_next=(j+1)%length;
            en += -(spins(i,j)*spins(i,j_next)+
                    spins(i,j)*spins(i_next,j));
            
        }
    }
    return en;
}

double ising_sim::get_magnetization()
{
    double mag=0;
    for (int i=0; i<length; ++i) {
        for (int j=0; j<length; ++j) {
            mag += spins(i,j);
        }
    }
    return mag;
}



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
        .define<int>("length", "size of the periodic box")
        .define<int>("sweeps", 0, "maximum number of sweeps (0 means indefinite)")
        .define<int>("thermalization", 10000, "number of sweeps for thermalization")
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
    , current_energy(0)
    , current_magnetization(0)
{
    // Initializes the spins
    for(int i=0; i<length; ++i) {
        for (int j=0; j<length; ++j) {
            // spins(i,j) = (random() < 0.5 ? 1 : -1);
            spins(i,j) = 1;
        }
    }

    // Calculates initial magnetization and energy
    // current_magnetization=get_magnetization();
    // current_energy=get_energy();
    current_magnetization=length*length;
    current_energy=-2*length*length;
    assert(get_energy()==current_energy);
    assert(get_magnetization()==current_magnetization);
    
    // for (int i=0; i<length; ++i) {
    //     for (int j=0; j<length; ++j) {
    //         current_magnetization += spins(i,j);
    //         int i_next=(i+1)%length;
    //         int j_next=(j+1)%length;
    //         current_energy += -(spins(i,j)*spins(i,j_next)+
    //                             spins(i,j)*spins(i_next,j));
            
    //     }
    // }
    // std::cerr << "Initial energy=" << current_energy << " magnetization=" << current_magnetization << std::endl << spins; // DEBUG
    
    // Adds the measurements
    measurements
        << alps::accumulators::FullBinningAccumulator<double>("Energy")
        << alps::accumulators::FullBinningAccumulator<double>("Magnetization")
        << alps::accumulators::FullBinningAccumulator<double>("AbsMagnetization")
        << alps::accumulators::FullBinningAccumulator<double>("Magnetization^2")
        << alps::accumulators::FullBinningAccumulator<double>("Magnetization^4")
        ;
}

// Performs the calculation at each MC step;
// decides if the step is accepted.
void ising_sim::update() {
    using std::exp;
    typedef unsigned int uint;
    // Choose a spin to flip:
    uint i = uint(length * random());
    uint j = uint(length * random());
    // Find neighbors indices, with wrap over box boundaries:
    uint i1 = (i+1) % length;            // left
    uint i2 = (i-1+length) % length;     // right
    uint j1 = (j+1) % length;            // up
    uint j2 = (j-1+length) % length;     // down
    // Energy difference:
    double delta=2.*spins(i,j)*
                    (spins(i1,j)+  // left
                     spins(i2,j)+  // right
                     spins(i,j1)+  // up
                     spins(i,j2)); // down
    // Step acceptance:
    if (delta<=0. || random() < exp(-beta*delta)) {
        // update energy:
        current_energy += delta;
        // update magnetization:
        current_magnetization -= 2*spins(i,j);
        // flip the spin
        spins(i,j) = -spins(i,j);

        // std::cerr << "Accepted, Delta=" << delta << std::endl << spins; // DEBUG
    } else {
        // std::cerr << "Rejected, Delta=" << delta << std::endl; // DEBUG
    }
        
}

// Collects the measurements at each MC step.
void ising_sim::measure() {
    ++sweeps;
    if (sweeps<thermalization_sweeps) return;
    
    const double l2=length*length; // number of sites
    double tmag = current_magnetization / l2; // magnetization

    assert(get_energy()==current_energy); // DEBUG
    assert(get_magnetization()==current_magnetization); // DEBUG
    // std::cerr << "Energy=" << current_energy << " Magnetization=" << current_magnetization << std::endl; // DEBUG
    // std::cerr << "Energ1=" << get_energy() << " Magnetizatio1=" << get_magnetization() << std::endl; // DEBUG
    
    // Accumulate the data (per site)
    measurements["Energy"] << (current_energy / l2);
    measurements["Magnetization"] << tmag;
    measurements["AbsMagnetization"] << fabs(tmag);
    measurements["Magnetization^2"] << tmag*tmag;
    measurements["Magnetization^4"] << tmag*tmag*tmag*tmag;
}

// Returns a number between 0.0 and 1.0 with the completion percentage
double ising_sim::fraction_completed() const {
    double f=0;
    if (total_sweeps>0 && sweeps >= thermalization_sweeps) {
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
