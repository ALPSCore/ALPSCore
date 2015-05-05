/*
 * Copyright (C) 1998-2015 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include "ising.hpp"

#include <boost/lambda/lambda.hpp>

void ising_sim::define_parameters(parameters_type& params)
{
    params.description("Tutorial example: 1D Ising simulation")
        .define<int>("SEED", 42, "PRNG seed")
        .define<int>("L", "Simulation length")
        .define<int>("THERMALIZATION", "Number of thermalization sweeps")
        .define<int>("SWEEPS", "Total sweeps")
        .define<double>("T", "Simulation temperature")
        .define<double>("timelimit", 0, "Time limit")
        .define<std::string>("output_file", "", "Name of the output file");
    ;
}

ising_sim::ising_sim(parameters_type const & params)
    : parameters(params)
    , random(boost::mt19937(int(parameters["SEED"])), boost::uniform_real<>())
    , length(parameters["L"])
    , sweeps(0)
    , thermalization_sweeps(int(parameters["THERMALIZATION"]))
    , total_sweeps(int(parameters["SWEEPS"]))
    , beta(1. / double(parameters["T"]))
    , spins(length)
{
    for(int i = 0; i < length; ++i)
        spins[i] = (random() < 0.5 ? 1 : -1);
    measurements
        << alps::accumulators::FullBinningAccumulator<double>("Energy")
        << alps::accumulators::FullBinningAccumulator<double>("Magnetization")
        << alps::accumulators::FullBinningAccumulator<double>("Magnetization^2")
        << alps::accumulators::FullBinningAccumulator<double>("Magnetization^4")
        << alps::accumulators::FullBinningAccumulator<std::vector<double> >("Correlations")
    ;
}

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

double ising_sim::fraction_completed() const {
    return (sweeps < thermalization_sweeps ? 0. : ( sweeps - thermalization_sweeps ) / double(total_sweeps));
}

bool ising_sim::run(boost::function<bool ()> const & stop_callback) {
    bool stopped = false;
    do {
        update();
        measure();
    } while(!(stopped = stop_callback()) && fraction_completed() < 1.);
    return !stopped;
}

// TODO: implement a nice keys(m) function
ising_sim::result_names_type ising_sim::result_names() const {
    result_names_type names;
    for(accumulators_type::const_iterator it = measurements.begin(); it != measurements.end(); ++it)
        names.push_back(it->first);
    return names;
}

ising_sim::result_names_type ising_sim::unsaved_result_names() const {
    return result_names_type(); 
}

ising_sim::results_type ising_sim::collect_results() const {
    return collect_results(result_names());
}

ising_sim::results_type ising_sim::collect_results(result_names_type const & names) const {
    results_type partial_results;
    for(result_names_type::const_iterator it = names.begin(); it != names.end(); ++it)
        partial_results[*it] = measurements[*it].result();
    return partial_results;
}

void ising_sim::save(boost::filesystem::path const & filename) const {
    alps::hdf5::archive ar(filename, "w");
    ar["/simulation/realizations/0/clones/0"] << *this;
}

void ising_sim::load(boost::filesystem::path const & filename) {
    alps::hdf5::archive ar(filename);
    ar["/simulation/realizations/0/clones/0"] >> *this;
}

void ising_sim::save(alps::hdf5::archive & ar) const {
    ar["/parameters"] << parameters;

    ar["measurements"] << measurements;
    ar["checkpoint/sweeps"] << sweeps;
    ar["checkpoint/spins"] << spins;

    {
        std::ostringstream os;
        os << random.engine();
        ar["checkpoint/engine"] << os.str();
    }

}

void ising_sim::load(alps::hdf5::archive & ar) {
    ar["/parameters"] >> parameters;
    length = int(parameters["L"]);
    thermalization_sweeps = int(parameters["THERMALIZATION"]);
    total_sweeps = int(parameters["SWEEPS"]);
    beta = 1. / double(parameters["T"]);

    ar["measurements"] >> measurements;

    ar["checkpoint/sweeps"] >> sweeps;
    ar["checkpoint/spins"] >> spins;

    {
        std::string state;
        ar["checkpoint/engine"] >> state;
        std::istringstream is(state);
        is >> random.engine();
    }
}
