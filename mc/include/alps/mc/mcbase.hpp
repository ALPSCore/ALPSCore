/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#pragma once

#include <boost/function.hpp>

#include <alps/accumulators.hpp>
#include <alps/params.hpp>
#include "random01.hpp"

#include <vector>
#include <string>

// move to alps::mcbase root scope
namespace alps {

    class mcbase {

        protected:

            typedef alps::accumulators::accumulator_set observable_collection_type;

        public:

            typedef alps::params parameters_type;
            typedef std::vector<std::string> result_names_type;

            typedef alps::accumulators::result_set results_type;

            mcbase(parameters_type const & parms, std::size_t seed_offset = 0);

            static parameters_type& define_parameters(parameters_type & parameters);

            virtual void update() = 0;
            virtual void measure() = 0;
            virtual double fraction_completed() const = 0;
            bool run(boost::function<bool ()> const & stop_callback);

            result_names_type result_names() const;
            result_names_type unsaved_result_names() const;
            results_type collect_results() const;
            results_type collect_results(result_names_type const & names) const;

            void save(std::string const & filename) const;
            void load(std::string const & filename);
            virtual void save(alps::hdf5::archive & ar) const;
            virtual void load(alps::hdf5::archive & ar);

        protected:

            parameters_type parameters;
            // parameters_type & params; // TODO: deprecated, remove!
            alps::random01 random;
            observable_collection_type measurements;
    };

    
}

