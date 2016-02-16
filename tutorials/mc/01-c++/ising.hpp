/*
 * Copyright (C) 1998-2016 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#pragma once

#include <alps/hdf5/archive.hpp>
#include <alps/hdf5/vector.hpp>

#include <alps/params.hpp>
#include <alps/accumulators.hpp>

#include <boost/function.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <vector>
#include <string>

class ising_sim {

    typedef alps::accumulators::accumulator_set accumulators_type;

    public:

        typedef alps::params parameters_type;
        typedef std::vector<std::string> result_names_type;
        typedef alps::accumulators::result_set results_type;

        ising_sim(parameters_type const & params);

        void update();
        void measure();
        double fraction_completed() const;
        bool run(boost::function<bool ()> const & stop_callback);

        result_names_type result_names() const;
        result_names_type unsaved_result_names() const;
        results_type collect_results() const;
        results_type collect_results(result_names_type const & names) const;

        static void define_parameters(parameters_type&);
        
        void save(boost::filesystem::path const & filename) const;
        void load(boost::filesystem::path const & filename);
        void save(alps::hdf5::archive & ar) const;
        void load(alps::hdf5::archive & ar);

    protected:

        parameters_type parameters;
        boost::variate_generator<boost::mt19937, boost::uniform_real<> > random;
        accumulators_type measurements;

    private:
        
        int length;
        int sweeps;
        int thermalization_sweeps;
        int total_sweeps;
        double beta;
        std::vector<int> spins;
};
