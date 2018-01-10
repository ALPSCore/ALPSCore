/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* Test printing of 1-index GF with various meshes */

#include <sstream>
#include <boost/array.hpp>

#include <gtest/gtest.h>
#include <alps/gf/gf.hpp>

namespace g=alps::gf;

template <typename M>
class mesh_generator {};

template <>
class mesh_generator<g::real_frequency_mesh> {
  public:
    static g::real_frequency_mesh get() { return g::real_frequency_mesh(); }
};

template <>
class mesh_generator<g::matsubara_positive_mesh> {
  public:
    static g::matsubara_positive_mesh get() { return g::matsubara_positive_mesh(0.1,10); }
};

template <>
class mesh_generator<g::itime_mesh> {
  public:
    static g::itime_mesh get() { return g::itime_mesh(0.1,10); }
};

template <>
class mesh_generator<g::power_mesh> {
  public:
    static g::power_mesh get() { return g::power_mesh(0.1,2,2); }
};

template <>
class mesh_generator<g::real_space_index_mesh> {
  public:
    typedef g::real_space_index_mesh value_type;
    static value_type get() {
        const boost::array<int,2> dims={{3,2}}; // 3 points, each is 2D
        value_type::container_type points(dims);
        points[0][0]=10; points[0][1]=11; 
        points[1][0]=20; points[1][1]=21; 
        points[2][0]=30; points[2][1]=31; 
        return value_type(points);
    }
};

template <>
class mesh_generator<g::momentum_index_mesh> {
  public:
    typedef g::momentum_index_mesh value_type;
    static value_type get() {
        const boost::array<int,2> dims={{3,2}}; // 3 points, each is 2D
        value_type::container_type points(dims);
        points[0][0]=10; points[0][1]=11; 
        points[1][0]=20; points[1][1]=21; 
        points[2][0]=30; points[2][1]=31; 
        return value_type(points);
    }
};

template <>
class mesh_generator<g::index_mesh> {
  public:
    static g::index_mesh get() { return g::index_mesh(3); }
};


// M is mesh type
template <typename M>
class OneIndexGFPrintTest : public ::testing::Test {
    typedef M mesh_type;
    typedef g::one_index_gf<double,mesh_type> gf_type;

    gf_type gf;
    std::ostringstream ostream;

  public:
    OneIndexGFPrintTest() : gf(mesh_generator<mesh_type>::get()) {
        gf.initialize();
    }
    
    void test_print() {
        ostream << gf;
        std::cout << "Output:\n" << ostream.str() << std::endl;
    }
};

typedef ::testing::Types<
    g::real_frequency_mesh,
    g::matsubara_positive_mesh,
    g::itime_mesh,
    g::power_mesh,
    g::momentum_index_mesh,
    g::real_space_index_mesh,
    g::index_mesh
    > mesh_types;

TYPED_TEST_CASE(OneIndexGFPrintTest, mesh_types);

TYPED_TEST(OneIndexGFPrintTest, print) { this->test_print(); }
