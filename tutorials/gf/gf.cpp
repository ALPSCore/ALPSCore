/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/gf/gf.hpp>


int main() {
  // Dimensions for Green's functions
  size_t N1 = 10;
  size_t N2 = 100;
  size_t N3 = 50;
  size_t N4 = 400;
  size_t N5 = 20;

  double beta = 10.0;

  // Define meshes for the Green's function object

  using matsubara = alps::gf::matsubara_positive_mesh;
  using index     = alps::gf::index_mesh;
  using itime     = alps::gf::itime_mesh;
  using legendre  = alps::gf::legendre_mesh;
  using chebyshev = alps::gf::chebyshev_mesh;

  using iw = alps::gf::matsubara_positive_mesh::index_type;
  using ii = alps::gf::index_mesh::index_type;
  using it = alps::gf::itime_mesh::index_type;
  using il = alps::gf::legendre_mesh::index_type;
  using ic = alps::gf::chebyshev_mesh::index_type;

  matsubara m1(beta, N1);
  index     m2(N2);
  itime     m3(beta, N3);
  legendre  m4(beta, N4);
  chebyshev m5(beta, N5);

  // create simple Green's function object
  alps::gf::greenf<double, matsubara, index, itime, legendre, chebyshev> g(m1, m2, m3, m4, m5);
  
  std::cout<<"Green's function object of the total size "<<g.data().size()<<" has been created"<<std::endl;

  // create view of the Green's function object for the first three indices equal to (0,0,1)
  alps::gf::greenf_view<double, legendre, chebyshev> g_view = g(iw(0), ii(0), it(1));
  
  std::cout<<"Green's function view of the total size "<<g_view.data().size()<<" has been created"<<std::endl;

  // work with the view object
  for (int i = 0; i < m4.extent(); ++i) {
    for (int j = 0; j < m5.extent(); ++j) {
      g_view(il(i), ic(j)) = i*10+j;
    }
  }

  // Check that we changed the original data
  for (int i = 0, cnt = 0; i < m4.extent(); ++i) {
    for (int j = 0; j < m5.extent(); ++j, ++cnt) {
      if(std::abs(g_view(il(i), ic(j)) - g(iw(0), ii(0), it(1), il(i), ic(j))) > 1e-10 ) {
        throw std::logic_error("Something wrong");
      }
    }
  }

  // create copy of the part of the Green's function object for the first three indices equal to (0,0)
  alps::gf::greenf<double, itime, legendre, chebyshev> g_copy = g(iw(0), ii(0));
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

}


