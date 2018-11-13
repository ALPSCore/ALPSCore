/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <fstream>

#include <alps/hdf5.hpp>
#include <alps/gf/gf.hpp>


// Define shortcuts for meshes types

using matsubara_t = alps::gf::matsubara_positive_mesh;
using index_t     = alps::gf::index_mesh;
using itime_t     = alps::gf::itime_mesh;
using legendre_t  = alps::gf::legendre_mesh;
using chebyshev_t = alps::gf::chebyshev_mesh;

using iw = alps::gf::matsubara_positive_mesh::index_type;
using ii = alps::gf::index_mesh::index_type;
using it = alps::gf::itime_mesh::index_type;
using il = alps::gf::legendre_mesh::index_type;
using ic = alps::gf::chebyshev_mesh::index_type;

// Initialize Green's function on the specific grids
void init_data(alps::gf::greenf<double, matsubara_t, index_t, itime_t, legendre_t, chebyshev_t> &g) {
  // Dimensions for Green's functions
  size_t N1 = 10;
  size_t N2 = 100;
  size_t N3 = 50;
  size_t N4 = 400;
  size_t N5 = 20;

  double beta = 10.0;

  // Define meshes for the Green's function object
  matsubara_t m1(beta, N1);
  index_t     m2(N2);
  itime_t     m3(beta, N3);
  legendre_t  m4(beta, N4);
  chebyshev_t m5(beta, N5);

  // Change the shape of the Green's function
  g.reshape(m1, m2, m3, m4, m5);
  std::cout<<"Green's function object of the total size "<<g.data().size()<<" has been created"<<std::endl;
}

// Perform I/O routines with Green's functions
void greens_function_io(alps::gf::greenf<double, itime_t, legendre_t, chebyshev_t> &g_copy,
                        alps::gf::greenf_view<double, legendre_t, chebyshev_t> &g_view) {
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
  alps::gf::greenf<double, legendre_t, chebyshev_t> g_view_copy = g_view;

  // change view object
  for (int i = 0; i < g_view.mesh1().extent(); ++i) {
    for (int j = 0; j < g_view.mesh2().extent(); ++j) {
      g_view(il(i), ic(j)) = i*10+j;
    }
  }

  // restore view by loading it from hdf5
  ar["GF_VIEW"] >> g_view;

  // check that the data has been restored
  if((g_view - g_view_copy).norm() > 1e-19 ) {
    throw std::logic_error("Something wrong");
  }
}

// Perform basic arithmetics on the Green's function object
void basic_arithmetics(alps::gf::greenf_view<double, legendre_t, chebyshev_t> &g_view) {
  // create copy of the view object
  alps::gf::greenf<double, legendre_t, chebyshev_t> g_view_copy = g_view;

  // Multiply by scalar
  g_view *= 2.0;

  // check the results
  for (int i = 0; i < g_view.mesh1().extent(); ++i) {
    for (int j = 0; j < g_view.mesh2().extent(); ++j) {
      if(std::abs(g_view(il(i), ic(j)) - 2.0*g_view_copy(il(i), ic(j)) ) > 1e-19 ) {
        throw std::logic_error("Something wrong");
      }
    }
  }

  // Addition
  alps::gf::greenf<double, legendre_t, chebyshev_t> g_a_plus_b = g_view + g_view_copy;
  for (int i = 0; i < g_view.mesh1().extent(); ++i) {
    for (int j = 0; j < g_view.mesh2().extent(); ++j) {
      if(std::abs(g_view(il(i), ic(j)) + g_view_copy(il(i), ic(j)) - g_a_plus_b(il(i), ic(j)) ) > 1e-19 ) {
        throw std::logic_error("Something wrong");
      }
    }
  }
  // asignment
  g_view = g_view_copy;
  for (int i = 0; i < g_view.mesh1().extent(); ++i) {
    for (int j = 0; j < g_view.mesh2().extent(); ++j) {
      if(std::abs(g_view(il(i), ic(j)) - g_view_copy(il(i), ic(j)) ) > 1e-19 ) {
        throw std::logic_error("Something wrong");
      }
    }
  }

  // Inplace addition
  g_view += g_a_plus_b;
  for (int i = 0; i < g_view.mesh1().extent(); ++i) {
    for (int j = 0; j < g_view.mesh2().extent(); ++j) {
      if(std::abs(g_view(il(i), ic(j)) - g_a_plus_b(il(i), ic(j)) - g_view_copy(il(i), ic(j)) ) > 1e-19 ) {
        throw std::logic_error("Something wrong");
      }
    }
  }

  // max norm
  if( (g_view - g_a_plus_b - g_view_copy).norm() > 1e-19 )
    throw std::logic_error("Something wrong");

}

int main() {
  // Define empty 5-dimensional Green's function object
  alps::gf::greenf<double,matsubara_t, index_t, itime_t, legendre_t, chebyshev_t> g;
  // Init 5-dimensional Green's function
  init_data(g);

  // create view of the Green's function object for the first three indices equal to (0,0,1)
  alps::gf::greenf_view<double, legendre_t, chebyshev_t> g_view = g(iw(0), ii(0), it(1));
  std::cout<<"Green's function view of the total size "<<g_view.data().size()<<" has been created"<<std::endl;

  matsubara_t m1 = g.mesh1();
  index_t     m2 = g.mesh2();
  itime_t     m3 = g.mesh3();
  legendre_t  m4 = g.mesh4();
  chebyshev_t m5 = g.mesh5();

  // work with the view object
  for (int i = 0; i < m4.extent(); ++i) {
    for (int j = 0; j < m5.extent(); ++j) {
      g_view(il(i), ic(j)) = i*10+j;
    }
  }

  // Check that we changed the original data using max norm
  if((g_view - g(iw(0), ii(0), it(1))).norm() > 1e-10 ) {
    throw std::logic_error("Something wrong");
  }

  // create copy of the part of the Green's function object for the first three indices equal to (0,0)
  alps::gf::greenf<double, itime_t, legendre_t, chebyshev_t> g_copy = g(iw(0), ii(0));
  std::cout<<"Copy of the part of the Green's function object of the total size "<<g_copy.data().size()<<" has been created"<<std::endl;

  for (int i = 0; i < m4.extent(); ++i) {
    for (int j = 0; j < m5.extent(); ++j) {
      // check that it is the same as view
      if(std::abs(g_view(il(i), ic(j)) - g_copy(it(1), il(i), ic(j))) > 1e-10 ) {
        throw std::logic_error("Something wrong");
      }
      // change the value
      g_copy(it(1), il(i), ic(j)) += i*4 + 5 + j;
      // check that it does not affect original
      if(std::abs(g_view(il(i), ic(j)) - g_copy(it(1), il(i), ic(j))) < 1e-10 ) {
        throw std::logic_error("Something wrong");
      }
    }
  }

  // perform basic I/O operations
  greens_function_io(g_copy, g_view);

  // perform basic arithmetic operations
  basic_arithmetics(g_view);

}
