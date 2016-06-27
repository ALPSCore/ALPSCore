/*
 * Copyright (C) 1998-2016 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#pragma once

#include <alps/mc/mcbase.hpp>
#include <alps/hdf5/archive.hpp>
#include <alps/hdf5/vector.hpp>
#include <boost/function.hpp>
#include <boost/filesystem/path.hpp>
#include <vector>
#include <string>
#include <iostream>

#include "storage_type.hpp"

// Simulation class for 2D Ising model (square lattice).
// We extend alps::mcbase, which is the base class of all Monte Carlo simulations.
// We define our own state, calculation functions (update/measure) and
// serialization functions (save/load)
class ising_sim : public alps::mcbase {
    // The internal state of our simulation
  private:
    int length; // the same in both dimensions
    int sweeps;
    int thermalization_sweeps;
    int total_sweeps;
    double beta;
    storage_type spins;
        
  public:
    ising_sim(parameters_type const & parms, std::size_t seed_offset = 0);

    static void define_parameters(parameters_type & parameters);

    virtual void update();
    virtual void measure();
    virtual double fraction_completed() const;

    using alps::mcbase::save;
    using alps::mcbase::load;
    virtual void save(alps::hdf5::archive & ar) const;
    virtual void load(alps::hdf5::archive & ar);
};
