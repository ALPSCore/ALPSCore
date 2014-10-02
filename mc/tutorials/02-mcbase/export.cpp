/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#define PY_ARRAY_UNIQUE_SYMBOL isingsim_PyArrayHandle

#include "ising.hpp"

#include <alps/ngs/detail/export_sim_to_python.hpp>

BOOST_PYTHON_MODULE(ising_c) {
    ALPS_EXPORT_SIM_TO_PYTHON(sim, ising_sim);
}
