/*
 * Copyright (C) 1998-2019 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file params_vec_hdf5.cpp

    @brief Tests saving/loading of vector of parameters
*/

#include "./params_test_support.hpp"

#include <iostream>
#include <alps/hdf5/vector.hpp>
#include <alps/testing/unique_file.hpp>

using alps::params;
namespace ah5=alps::hdf5;

class ParamsVectorTest : public ::testing::Test {
  protected:
    std::vector<params> par_;
  public:
    ParamsVectorTest() : par_(3) {
      par_[0]["p"] = 1.0;
      par_[1]["q"] = 2.0;
      par_[2]["r"] = 3.0;
    }
};

TEST_F(ParamsVectorTest, saveLoad) {
    alps::testing::unique_file ufile("params_vec.h5.", alps::testing::unique_file::REMOVE_NOW);
    std::vector<params> p_other;
    {
        ah5::archive ar(ufile.name(), "w");
        ar["paramsvec"] << par_;
    }

    {
        ah5::archive ar(ufile.name(), "r");
        ar["paramsvec"] >> p_other;
    }

    EXPECT_EQ(par_.size(), p_other.size());
    for (std::size_t i = 0; i < par_.size(); ++i) {
      EXPECT_EQ(par_[i], p_other[i]);
    }
}
