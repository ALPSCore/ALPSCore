/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <gtest/gtest.h>

#include <alps/gf_new/gf_base.h>
#include <alps/gf/mesh.hpp>


using namespace alps::gf;

TEST(GreensFunction, InitializationTest){
  greenf<double, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh, alps::gf::itime_mesh, alps::gf::legendre_mesh> g;

}

TEST(GreensFunction, AsignmentTest){
  alps::gf::matsubara_positive_mesh x(100, 10);
  alps::gf::index_mesh y(10);
  greenf<std::complex<double>, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh> g1(x,y);
  greenf_view<std::complex<double>, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh> g2(g1);
  greenf<std::complex<double>, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh> g3(g2);
  ASSERT_EQ(g1,g2);
}


TEST(GreensFunction, MeshAccess) {
  greenf<double, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh, alps::gf::itime_mesh, alps::gf::legendre_mesh> g;
  ASSERT_EQ(g.mesh1().extent(), 0);
  ASSERT_EQ(g.mesh2().extent(), 0);
  ASSERT_EQ(g.mesh3().extent(), 0);
  ASSERT_EQ(g.mesh4().extent(), 0);
}

TEST(GreensFunction, BasicArithmetics) {
  alps::gf::index_mesh x(10);
  greenf<double, alps::gf::index_mesh> g(x);
  greenf<double, alps::gf::index_mesh> g2(x);
  for(alps::gf::index_mesh::index_type i(0); i<x.extent(); ++i) {
    g(i) = 1.0 * i();
    g2(i) = 2.0 * i();
  }
  greenf<double, alps::gf::index_mesh> g3 = g+g2;
  for(alps::gf::index_mesh::index_type i(0); i<x.extent(); ++i) {
    ASSERT_DOUBLE_EQ(g3(i), 3.0 * i());
  }
  g3 -= g2;
  for(alps::gf::index_mesh::index_type i(0); i<x.extent(); ++i) {
    ASSERT_DOUBLE_EQ(g3(i), 1.0 * i());
  }
}

TEST(GreensFunction, BasicArithmetics2D) {
  alps::gf::matsubara_positive_mesh x(100, 10);
  alps::gf::index_mesh y(10);
  greenf<std::complex<double>, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh> g(x,y);
  greenf<std::complex<double>, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh> g2(x,y);
  for(alps::gf::index_mesh::index_type i(0); i<y.extent(); ++i) {
    for(alps::gf::matsubara_positive_mesh::index_type w(0); w<x.extent(); ++w) {
      g(w,i) = 1.0 * i();
      g2(w,i) = 2.0 * i();
    }
  }
  greenf<std::complex<double>, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh> g3 = g+g2;
  for(alps::gf::index_mesh::index_type i(0); i<y.extent(); ++i) {
    for(alps::gf::matsubara_positive_mesh::index_type w(0); w<x.extent(); ++w) {
      ASSERT_DOUBLE_EQ(g3(w,i).real(), 3.0 * i());
      ASSERT_DOUBLE_EQ(g3(w,i).imag(), 0.0);
    }
  }
  g3 -= g2;
  for(alps::gf::index_mesh::index_type i(0); i<x.extent(); ++i) {
    for(alps::gf::matsubara_positive_mesh::index_type w(0); w<x.extent(); ++w) {
      ASSERT_DOUBLE_EQ(g3(w, i).real(), 1.0 * i());
      ASSERT_DOUBLE_EQ(g3(w, i).imag(), 0.0);
    }
  }
}

TEST(GreensFunction, BasicArithmetics2DScaling) {
  alps::gf::matsubara_positive_mesh x(100, 10);
  alps::gf::index_mesh y(10);
  greenf<std::complex<double>, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh> g(x,y);
  for(alps::gf::matsubara_positive_mesh::index_type w(0); w<x.extent(); ++w) {
    for(alps::gf::index_mesh::index_type i(0); i<y.extent(); ++i) {
      g(w,i) = 3.0 * i() + std::complex<double>(0.0,w());
    }
  }
  double s = 3.0;
  g *= s;
  for(alps::gf::matsubara_positive_mesh::index_type w(0); w<x.extent(); ++w) {
    for(alps::gf::index_mesh::index_type i(0); i<y.extent(); ++i) {
      ASSERT_DOUBLE_EQ(3.0 * i() * s, g(w, i).real());
      ASSERT_DOUBLE_EQ(w() * s, g(w, i).imag());
    }
  }
  g /= s;
  for(alps::gf::matsubara_positive_mesh::index_type w(0); w<x.extent(); ++w) {
    for(alps::gf::index_mesh::index_type i(0); i<y.extent(); ++i) {
      ASSERT_DOUBLE_EQ(3.0 * i(), g(w, i).real());
      ASSERT_DOUBLE_EQ(w(), g(w, i).imag());
    }
  }
  std::complex<double> f(0.0, s);
  g *= f;
  for(alps::gf::matsubara_positive_mesh::index_type w(0); w<x.extent(); ++w) {
    for(alps::gf::index_mesh::index_type i(0); i<y.extent(); ++i) {
      ASSERT_DOUBLE_EQ(-w() * s, g(w, i).real());
      ASSERT_DOUBLE_EQ(3.0 * i() * s, g(w, i).imag());
    }
  }
}

TEST(GreensFunction, BasicArithmetics2DScalingComplex) {
  alps::gf::matsubara_positive_mesh x(100, 10);
  alps::gf::index_mesh y(10);
  greenf<double, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh > g(x, y);
  for (alps::gf::matsubara_positive_mesh::index_type w(0); w < x.extent(); ++w) {
    for (alps::gf::index_mesh::index_type i(0); i < y.extent(); ++i) {
      g(w, i) = 3.0 * i() + w();
    }
  }
  std::complex<double> s = 3.0;
  greenf<std::complex<double>, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh > g2 = g * s;
  for (alps::gf::matsubara_positive_mesh::index_type w(0); w < x.extent(); ++w) {
    for (alps::gf::index_mesh::index_type i(0); i < y.extent(); ++i) {
      ASSERT_DOUBLE_EQ(g(w, i) * s.real(), g2(w, i).real());
      ASSERT_DOUBLE_EQ(g(w, i) * s.imag(), g2(w, i).imag());
    }
  }
}

TEST(GreensFunction, TestSave) {
  alps::gf::matsubara_positive_mesh x(100, 10);
  alps::gf::index_mesh y(10);
  greenf<std::complex<double>, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh> g(x,y);
  for(alps::gf::matsubara_positive_mesh::index_type w(0); w<x.extent(); ++w) {
    for(alps::gf::index_mesh::index_type i(0); i<y.extent(); ++i) {
      g(w,i) = 3.0 * i() + w();
    }
  }
  alps::hdf5::archive ar("test", "w");
  g.save(ar, "");
  greenf<std::complex<double>, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh> g2;
  g2.load(ar, "");
  g2.mesh1();
  ASSERT_TRUE(g == g2);
}

TEST(GreensFunction, TestSlices) {
  alps::gf::matsubara_positive_mesh x(100, 10);
  alps::gf::index_mesh y(10);
  greenf<double, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh> g(x,y);
  for(alps::gf::matsubara_positive_mesh::index_type w(0); w<x.extent(); ++w) {
    for(alps::gf::index_mesh::index_type i(0); i<y.extent(); ++i) {
      g(w,i) = 3.0 * i() + w();
    }
  }

  for(alps::gf::matsubara_positive_mesh::index_type w(0); w<x.extent(); ++w) {
    greenf_view<double, alps::gf::index_mesh> g2 = g(w);
    for(alps::gf::index_mesh::index_type i(0); i<y.extent(); ++i) {
      double old = g(w,i);
      g2(i) += 10;
      ASSERT_EQ(old+10, g(w,i));
    }
    for(alps::gf::index_mesh::index_type i(0); i<y.extent(); ++i) {
      ASSERT_EQ(g(w, i), g2(i));
    }
  }

}

TEST(GreensFunction, TestMultidimensionalSlices) {
  alps::gf::matsubara_positive_mesh x(100, 10);
  alps::gf::index_mesh y(10);
  alps::gf::itime_mesh z(100, 10);
  greenf<double, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh, alps::gf::itime_mesh> g(x,y,z);
  for(alps::gf::matsubara_positive_mesh::index_type w(0); w<x.extent(); ++w) {
    for(alps::gf::index_mesh::index_type i(0); i<y.extent(); ++i) {
      greenf_view < double, alps::gf::itime_mesh > g3 = g(w,i);
      for(alps::gf::itime_mesh::index_type t(0); t<z.extent(); ++t) {
        g3(t) = w()*x.extent() + i() * y.extent() + t();
      }
    }
    greenf_view < double, alps::gf::index_mesh, alps::gf::itime_mesh > g2 = g(w);
  }
  for(alps::gf::matsubara_positive_mesh::index_type w(0); w<x.extent(); ++w) {
    for (alps::gf::index_mesh::index_type i(0); i < y.extent(); ++i) {
      for (alps::gf::itime_mesh::index_type t(0); t < z.extent(); ++t) {
        ASSERT_EQ(g(w,i,t), w()*x.extent() + i() * y.extent() + t());
      }
    }
  }
}


TEST(GreensFunction, NegateGF) {
  alps::gf::matsubara_positive_mesh x(100, 10);
  alps::gf::index_mesh y(10);
  greenf<double, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh > g(x, y);
  for (alps::gf::matsubara_positive_mesh::index_type w(0); w < x.extent(); ++w) {
    for (alps::gf::index_mesh::index_type i(0); i < y.extent(); ++i) {
      g(w, i) = 3.0 * i() + w();
    }
  }
  greenf<double, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh > g2 = -g;
  for (alps::gf::matsubara_positive_mesh::index_type w(0); w < x.extent(); ++w) {
    for (alps::gf::index_mesh::index_type i(0); i < y.extent(); ++i) {
      ASSERT_DOUBLE_EQ(g(w, i), -g2(w, i));
    }
  }
}