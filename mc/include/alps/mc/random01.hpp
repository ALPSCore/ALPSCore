/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#pragma once

#include <alps/hdf5/archive.hpp>

#include <boost/random.hpp>

#include <string>
#include <sstream>

namespace alps {

    struct random01 : public boost::variate_generator<boost::mt19937, boost::uniform_01<double> > {
        random01(int seed = 42)
            : boost::variate_generator<boost::mt19937, boost::uniform_01<double> >(boost::mt19937(seed), boost::uniform_01<double>())
        {}

        void save(alps::hdf5::archive & ar) const { // TODO: move this to hdf5 archive!
            std::ostringstream os;
            os << this->engine();
            ar["engine"] << os.str();
        }

        void load(alps::hdf5::archive & ar) { // TODO: move this to hdf5 archive!
            std::string state;
            ar["engine"] >> state;
            std::istringstream is(state);
            is >> this->engine();
        }
    };

}
