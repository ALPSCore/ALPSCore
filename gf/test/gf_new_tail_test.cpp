//
// Created by iskakoff on 08/12/17.
//

#include <gtest/gtest.h>

#include <alps/gf_new/gf_tail.h>
#include <alps/gf/mesh.hpp>


using namespace alps::gf;

TEST(GreensFunction, InitializationTest){
  gf_tail<double, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh, alps::gf::itime_mesh, alps::gf::legendre_mesh> g;

}

TEST(GreensFunction, AssignTest){
  alps::gf::matsubara_positive_mesh x(100, 10);
  alps::gf::index_mesh y(10);
  alps::gf::itime_mesh z(100, 10);
  greenf<double, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh, alps::gf::itime_mesh> g(x,y,z);
  gf_tail<double, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh, alps::gf::itime_mesh> g2(g);
  gf_tail_view<double, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh, alps::gf::itime_mesh> g3(g);
  ASSERT_EQ(g.data(), g2.data());
  ASSERT_EQ(g.data(), g3.data());

}

TEST(GreensFunction, SetTail){
  alps::gf::matsubara_positive_mesh x(100, 10);
  alps::gf::index_mesh y(10);
  alps::gf::itime_mesh z(100, 10);
  greenf<double, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh, alps::gf::itime_mesh> g(x,y,z);
  gf_tail<double, alps::gf::matsubara_positive_mesh, alps::gf::index_mesh, alps::gf::itime_mesh> g2 ( g);
  greenf<double, alps::gf::index_mesh, alps::gf::itime_mesh> tail(y,z);
  tail(alps::gf::index_mesh::index_type(0), alps::gf::itime_mesh::index_type(0)) = 10.0;
  g2.set_tail(0, tail);
  auto tail2 = g2.tail(0);
  ASSERT_EQ(tail, tail2);
}

