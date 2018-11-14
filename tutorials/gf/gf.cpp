/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <fstream>
#include <limits>

#include <alps/hdf5.hpp>
#include <alps/gf/gf.hpp>

// machine precision
static const double tol = std::numeric_limits<double>::epsilon();
// Define shortcuts for meshes and mesh indices types
using matsubara_mesh_t  = alps::gf::matsubara_positive_mesh;
using index_mesh_t      = alps::gf::index_mesh;
using itime_mesh_t      = alps::gf::itime_mesh;
using legendre_mesh_t   = alps::gf::legendre_mesh;
using chebyshev_mesh_t  = alps::gf::chebyshev_mesh;

using matsubara_index_t = alps::gf::matsubara_positive_mesh::index_type;
using index_index_t     = alps::gf::index_mesh::index_type;
using itime_index_t     = alps::gf::itime_mesh::index_type;
using legendre_index_t  = alps::gf::legendre_mesh::index_type;
using chebyshev_index_t = alps::gf::chebyshev_mesh::index_type;

// Initialize Green's function on the specific grids
void init_data(alps::gf::greenf<double, matsubara_mesh_t, index_mesh_t, itime_mesh_t, legendre_mesh_t, chebyshev_mesh_t> &g) {
  // Dimensions for Green's functions
  size_t N1 = 10;
  size_t N2 = 100;
  size_t N3 = 50;
  size_t N4 = 400;
  size_t N5 = 20;

  double beta = 10.0;

  // Define meshes for the Green's function object
  matsubara_mesh_t m1(beta, N1);
  index_mesh_t     m2(N2);
  itime_mesh_t     m3(beta, N3);
  legendre_mesh_t  m4(beta, N4);
  chebyshev_mesh_t m5(beta, N5);

  // Change the shape of the Green's function
  g.reshape(m1, m2, m3, m4, m5);
  std::cout<<"Green's function object of the total size "<<g.data().size()<<" has been created"<<std::endl;
}

// Perform I/O routines with Green's functions
void greens_function_io(alps::gf::greenf<double, itime_mesh_t, legendre_mesh_t, chebyshev_mesh_t> &g_copy,
                        alps::gf::greenf_view<double, legendre_mesh_t, chebyshev_mesh_t> &g_view) {
  // Save Green's functions into the file
  std::ofstream f1("green.dat");
  f1<<g_copy;
  f1.close();
  f1.open("green_view.dat");
  f1<<g_view;

  // Save/load Green's functions into hdf5
  alps::hdf5::archive ar("gf.h5", "w");

  // save Green's function object into group 'GF'
  ar["GF"]<<g_copy;
  // save Green's function view into group 'GF_VIEW'
  ar["GF_VIEW"]<<g_view;

  // create copy of the view object
  alps::gf::greenf<double, legendre_mesh_t, chebyshev_mesh_t> g_view_copy = g_view;

  // change view object
  for (legendre_index_t i(0); i < g_view.mesh1().extent(); ++i) {
    for (chebyshev_index_t j(0); j < g_view.mesh2().extent(); ++j) {
      g_view(i, j) = i()*10+j();
    }
  }

  // restore view by loading it from hdf5
  ar["GF_VIEW"] >> g_view;

  // check that the data has been restored
  if((g_view - g_view_copy).norm() > tol) {
    throw std::logic_error("Something wrong");
  }
}

// Perform basic arithmetics on the Green's function object
void basic_arithmetics(alps::gf::greenf_view<double, legendre_mesh_t, chebyshev_mesh_t> &g_view) {
  // create copy of the view object
  alps::gf::greenf<double, legendre_mesh_t, chebyshev_mesh_t> g_view_copy = g_view;

  // Multiply by scalar
  g_view *= 2.0;

  // check the results
  for (legendre_index_t i(0); i < g_view.mesh1().extent(); ++i) {
    for (chebyshev_index_t j(0); j < g_view.mesh2().extent(); ++j) {
      if(std::abs(g_view(i, j) - 2.0*g_view_copy(i, j) ) > tol) {
        throw std::logic_error("Something wrong");
      }
    }
  }

  // Addition
  alps::gf::greenf<double, legendre_mesh_t, chebyshev_mesh_t> g_a_plus_b = g_view + g_view_copy;
  for (legendre_index_t i(0); i < g_view.mesh1().extent(); ++i) {
    for (chebyshev_index_t j(0); j < g_view.mesh2().extent(); ++j) {
      if(std::abs(g_view(i, j) + g_view_copy(i, j) - g_a_plus_b(i, j) ) > tol) {
        throw std::logic_error("Something wrong");
      }
    }
  }
  // asignment
  g_view = g_view_copy;
  for (legendre_index_t i(0); i < g_view.mesh1().extent(); ++i) {
    for (chebyshev_index_t j(0); j < g_view.mesh2().extent(); ++j) {
      if(std::abs(g_view(i, j) - g_view_copy(i, j) ) > tol) {
        throw std::logic_error("Something wrong");
      }
    }
  }

  // Inplace addition
  g_view += g_a_plus_b;
  for (legendre_index_t i(0); i < g_view.mesh1().extent(); ++i) {
    for (chebyshev_index_t j(0); j < g_view.mesh2().extent(); ++j) {
      if(std::abs(g_view(i, j) - g_a_plus_b(i, j) - g_view_copy(i, j) ) > tol) {
        throw std::logic_error("Something wrong");
      }
    }
  }

  // max norm
  if((g_view - g_a_plus_b - g_view_copy).norm() > tol)
    throw std::logic_error("Something wrong");

}

int main() {
  // Define empty 5-dimensional Green's function object
  alps::gf::greenf<double,matsubara_mesh_t, index_mesh_t, itime_mesh_t, legendre_mesh_t, chebyshev_mesh_t> g;
  // Init 5-dimensional Green's function
  init_data(g);

  // create view of the Green's function object for the first three indices equal to (0,0,1)
  alps::gf::greenf_view<double, legendre_mesh_t, chebyshev_mesh_t> g_view = g(matsubara_index_t(0), index_index_t(0), itime_index_t(1));
  std::cout<<"Green's function view of the total size "<<g_view.data().size()<<" has been created"<<std::endl;

  matsubara_mesh_t m1 = g.mesh1();
  index_mesh_t     m2 = g.mesh2();
  itime_mesh_t     m3 = g.mesh3();
  legendre_mesh_t  m4 = g.mesh4();
  chebyshev_mesh_t m5 = g.mesh5();

  // work with the view object
  for (legendre_index_t i(0); i < m4.extent(); ++i) {
    for (chebyshev_index_t j(0); j < m5.extent(); ++j) {
      g_view(i, j) = i()*10+j();
    }
  }

  // Check that we changed the original data using max norm
  if((g_view - g(matsubara_index_t(0), index_index_t(0), itime_index_t(1))).norm() > tol ) {
    throw std::logic_error("Something wrong");
  }

  // create copy of the part of the Green's function object for the first three indices equal to (0,0)
  alps::gf::greenf<double, itime_mesh_t, legendre_mesh_t, chebyshev_mesh_t> g_copy = g(matsubara_index_t(0), index_index_t(0));
  std::cout<<"Copy of the part of the Green's function object of the total size "<<g_copy.data().size()<<" has been created"<<std::endl;

  for (legendre_index_t i(0); i < m4.extent(); ++i) {
    for (chebyshev_index_t j(0); j < m5.extent(); ++j) {
      // check that it is the same as view
      if(std::abs(g_view(i, j) - g_copy(itime_index_t(1), i, j)) > tol ) {
        throw std::logic_error("Something wrong");
      }
      // change the value
      g_copy(itime_index_t(1), i, j) += i()*4 + 5 + j();
    }
  }

  // check that it does not affect original
  if((g_view - g_copy(itime_index_t(1)) ).norm() < tol ) {
    throw std::logic_error("Something wrong");
  }

  // perform basic I/O operations
  greens_function_io(g_copy, g_view);

  // perform basic arithmetic operations
  basic_arithmetics(g_view);

}
