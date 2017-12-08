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