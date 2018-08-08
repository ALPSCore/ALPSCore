/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <gtest/gtest.h>

#include <alps/gf/gf_base.hpp>
#include <alps/gf/mesh.hpp>

#include <alps/testing/near.hpp>


using namespace alps::gf;

TEST(GreensFunction, InitializationTest){
  alps::gf::matsubara_positive_mesh x(100, 10);
  alps::gf::index_mesh y(10);
  alps::gf::itime_mesh z(100, 10);
  alps::gf::legendre_mesh w(100, 10);

  greenf<double, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh, alps::gf::itime_mesh, alps::gf::legendre_mesh> g(x, y, z, w);
  for(std::size_t i = 0; i<g.data().size(); ++i) {
    ASSERT_NEAR(g.data().storage().data(i), 0.0, 1E-15);
  }
  for(alps::gf::matsubara_positive_mesh::index_type i(0); i<x.extent(); ++i) {
    for(alps::gf::index_mesh::index_type j(0); j<x.extent(); ++j){
      for(alps::gf::itime_mesh::index_type k(0); k<x.extent(); ++k){
        for(alps::gf::legendre_mesh::index_type l(0); l<x.extent(); ++l){
          ASSERT_NEAR(g(i,j,k,l), 0.0, 1E-15);
        }
      }
    }
  }
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

TEST(GreensFunction, BasicArithmetics2DMixedTypes) {
  alps::gf::matsubara_positive_mesh x(100, 10);
  alps::gf::index_mesh y(10);
  greenf < double, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh > g(x, y);
  greenf < std::complex < double >, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh > g2(x, y);
  for (alps::gf::index_mesh::index_type i(0); i < y.extent(); ++i) {
    for (alps::gf::matsubara_positive_mesh::index_type w(0); w < x.extent(); ++w) {
      g(w, i) = 1.0 * i();
      g2(w, i) = 2.0 * i();
    }
  }
  greenf<std::complex<double>, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh> g3 = g + g2;
  for(alps::gf::index_mesh::index_type i(0); i<y.extent(); ++i) {
    for(alps::gf::matsubara_positive_mesh::index_type w(0); w<x.extent(); ++w) {
      ASSERT_DOUBLE_EQ(g3(w,i).real(), 3.0 * i());
      ASSERT_DOUBLE_EQ(g3(w,i).imag(), 0.0);
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
  auto gg = g / s;
  g /= s;
  for(alps::gf::matsubara_positive_mesh::index_type w(0); w<x.extent(); ++w) {
    for(alps::gf::index_mesh::index_type i(0); i<y.extent(); ++i) {
      ASSERT_DOUBLE_EQ(3.0 * i(), g(w, i).real());
      ASSERT_DOUBLE_EQ(w(), g(w, i).imag());
      ASSERT_DOUBLE_EQ(gg(w,i).real(), g(w, i).real());
      ASSERT_DOUBLE_EQ(gg(w,i).imag(), g(w, i).imag());
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
  greenf<std::complex<double>, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh > g3 = s * g;
  ASSERT_EQ(g2, g3);
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
  alps::hdf5::archive ar("test.h5", "w");
  g.save(ar, "");
  greenf<std::complex<double>, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh> g2;
  g2.load(ar, "");
  g2.mesh1();
  ASSERT_TRUE(g == g2);
}

TEST(GreensFunction, TestSlices) {
  alps::gf::matsubara_positive_mesh x(100, 10);
  alps::gf::index_mesh y(20);
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
  alps::gf::index_mesh y(15);
  alps::gf::itime_mesh z(100, 25);
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

TEST(GreensFunction, StreamOut) {
  alps::gf::matsubara_positive_mesh x(100, 10);
  alps::gf::itime_mesh y(100, 3);
  alps::gf::index_mesh z(3);
  greenf<double, alps::gf::matsubara_positive_mesh, alps::gf::itime_mesh, alps::gf::index_mesh > g(x, y, z);
  int k = 0;
  for(alps::gf::matsubara_positive_mesh::index_type w(0); w<x.extent(); ++w) {
    for (alps::gf::itime_mesh::index_type t(0); t < y.extent(); ++t) {
      for (alps::gf::index_mesh::index_type i(0); i < z.extent(); ++i) {
        g(w, t, i) = ++k;
      }
    }
  }

  std::stringstream gf_stream_by_hand;
  gf_stream_by_hand<<x
                   <<y
                   <<z;
  k = 0;
  for(int i=0;i<x.extent();++i){
    gf_stream_by_hand<<(2*i+1)*M_PI/100.;
    for(int n = 0; n< 9 ;++n){
      gf_stream_by_hand<<" "<<std::to_string(++k);
    }
    gf_stream_by_hand<<" "<<std::endl;
  }
  std::stringstream gf_stream;
  gf_stream<<g;
  EXPECT_EQ(gf_stream_by_hand.str(), gf_stream.str());
}

TEST(GreensFunction, ConstRef) {
  alps::gf::matsubara_positive_mesh x(100, 10);
  alps::gf::index_mesh y(10);
  alps::gf::index_mesh z(10);
  greenf<double, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh, alps::gf::index_mesh > g(x, y, z);
  const greenf<double, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh, alps::gf::index_mesh >& g2 = g;
  for (alps::gf::matsubara_positive_mesh::index_type w(0); w < x.extent(); ++w) {
    for (alps::gf::index_mesh::index_type i(0); i < y.extent(); ++i) {
      for (alps::gf::index_mesh::index_type j(0); j < z.extent(); ++j) {
        g(w, i, j) = 3.0 * j() + w() + i();
      }
    }
  }
  for (alps::gf::matsubara_positive_mesh::index_type w(0); w < x.extent(); ++w) {
    auto g3 = g2(w);
    for (alps::gf::index_mesh::index_type i(0); i < y.extent(); ++i) {
      std::stringstream gf_stream1;
      std::stringstream gf_stream2;
      auto g4 = g2(w)(i);
      auto g5 = g3(i);
      gf_stream1<<g4;
      gf_stream2<<g5;
      EXPECT_EQ(gf_stream1.str(), gf_stream2.str());
    }
  }
}

TEST(GreensFunction, Reshape) {
  alps::gf::matsubara_positive_mesh x(100, 10);
  alps::gf::index_mesh y(10);
  alps::gf::index_mesh y2(1);
  alps::gf::index_mesh z(10);
  alps::gf::index_mesh z2(100);
  greenf<double, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh, alps::gf::index_mesh > g1(x, y, z);
  greenf<double, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh, alps::gf::index_mesh > g2;
  ASSERT_TRUE(g2.data().size() == 0);
  std::array<size_t, 3> shape{{}};
  ASSERT_TRUE(g2.data().shape() == shape);
  g2.reshape(x, y, z);
  ASSERT_TRUE(g2.data().shape() == g1.data().shape());
  ASSERT_NO_THROW(g2(alps::gf::matsubara_positive_mesh::index_type(0)).reshape(y2, z2));
  ASSERT_THROW(g2(alps::gf::matsubara_positive_mesh::index_type(0)).reshape(y, z2), std::invalid_argument);
}

TEST(GreensFunction, MeshAssignment) {
  alps::gf::real_space_index_mesh m1(4, 10);
  for(int i = 0; i< 4; ++i) {
    for(int j = 0; j<10; ++j) {
      m1.points()[i][j] = i*2+j + 1;
    }
  }
  greenf<double, alps::gf::matsubara_positive_mesh, alps::gf::real_space_index_mesh> g(alps::gf::matsubara_positive_mesh(100, 10),m1);
  for(int i = 0; i< 4; ++i) {
    for(int j = 0; j<10; ++j) {
      ASSERT_EQ(m1.points()[i][j], g.mesh2().points()[i][j]);
    }
  }
}

TEST(GreensFunction, ConstructorTests){
  alps::gf::matsubara_positive_mesh x(100, 10);
  alps::gf::index_mesh y(10);
  alps::gf::itime_mesh z(100, 10);
  alps::gf::legendre_mesh w(100, 10);

  //construct without data passed in and individual meshes
  greenf<double, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh, alps::gf::itime_mesh, alps::gf::legendre_mesh> g(x, y, z, w);
  for(std::size_t i = 0; i<g.data().size(); ++i) {
    ASSERT_EQ(g.data().storage().data(i), 0.0);
  }

  //construct with data passed in as pointer
  std::vector<double> data(x.extent()*y.extent()*z.extent()*w.extent(), 7.);
  greenf<double, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh, alps::gf::itime_mesh, alps::gf::legendre_mesh> g2(&(data[0]), std::make_tuple(x, y, z, w));
  for(std::size_t i = 0; i<g.data().size(); ++i) {
    ASSERT_EQ(g2.data().storage().data(i), 7.);
  }
 
  //construct with data passed in as vector 
  {
    alps::numerics::tensor < double, 4 > data_tensor(x.extent(), y.extent(), z.extent(), w.extent());
    double *data_tensor_data_ptr=data_tensor.data(); std::fill(data_tensor_data_ptr, data_tensor_data_ptr+x.extent()*y.extent()*z.extent()*w.extent(), M_PI);
    greenf<double, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh, alps::gf::itime_mesh, alps::gf::legendre_mesh> g3(data_tensor, std::make_tuple(x, y, z, w));
    greenf<double, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh, alps::gf::itime_mesh, alps::gf::legendre_mesh> g4(data_tensor, x, y, z, w);
    for(std::size_t i = 0; i<g.data().size(); ++i) {
      ASSERT_EQ(g3.data().storage().data(i), M_PI);
      ASSERT_EQ(g4.data().storage().data(i), M_PI);
    }
  }
  
  //construct with data passed in by move constructor
  {
    alps::numerics::tensor < double, 4 > data_tensor_1(x.extent(), y.extent(), z.extent(), w.extent());
    alps::numerics::tensor < double, 4 > data_tensor_2(x.extent(), y.extent(), z.extent(), w.extent());
    double *data_tensor_data_ptr_1=data_tensor_1.data(); std::fill(data_tensor_data_ptr_1, data_tensor_data_ptr_1+x.extent()*y.extent()*z.extent()*w.extent(), M_PI*2);
    double *data_tensor_data_ptr_2=data_tensor_2.data(); std::fill(data_tensor_data_ptr_2, data_tensor_data_ptr_2+x.extent()*y.extent()*z.extent()*w.extent(), M_PI*3);
    greenf<double, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh, alps::gf::itime_mesh, alps::gf::legendre_mesh> g3(std::move(data_tensor_1), std::make_tuple(x, y, z, w));
    greenf<double, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh, alps::gf::itime_mesh, alps::gf::legendre_mesh> g4(std::move(data_tensor_2), x, y, z, w);
    for(std::size_t i = 0; i<g.data().size(); ++i) {
      ASSERT_EQ(g3.data().storage().data(i), M_PI*2);
      ASSERT_EQ(g4.data().storage().data(i), M_PI*3);
    }
  }
}
