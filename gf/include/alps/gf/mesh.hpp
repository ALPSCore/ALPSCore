/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once
#include <complex>
#include <cassert>
#include <array>
#include <type_traits>
#include <boost/multi_array.hpp>
#include <boost/operators.hpp>

#include <alps/hdf5.hpp>

#ifdef ALPS_HAVE_MPI
#include <alps/gf/mpi_bcast.hpp>
#endif

#include <alps/gf/mesh/index.hpp>
#include <alps/gf/mesh/frequency_meshes.hpp>
#include <alps/gf/mesh/time_meshes.hpp>
#include <alps/gf/mesh/index_meshes.hpp>
#include <alps/gf/mesh/legendre_mesh.hpp>
#include <alps/gf/mesh/numerical_mesh.hpp>
#include <alps/gf/mesh/mesh_base.hpp>
#include <alps/gf/mesh/chebyshev_mesh.hpp>

#include"flagcheck.hpp"

namespace alps {namespace gf {
  namespace detail {
    /// Print a 2D double boost::multi_array (for printing 2D meshes)
    /** @todo FIXME: Use a proper mesh-specific method instead, see operator<<(momentum_realspace_index_mesh) */
    std::ostream& operator<<(std::ostream& s, const boost::multi_array<double, 1>& data);

  }
}}
