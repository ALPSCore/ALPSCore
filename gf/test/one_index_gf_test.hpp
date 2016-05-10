/*
 * Copyright (C) 1998-2016 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include "gtest/gtest.h"
#include "alps/gf/gf.hpp"
#include "alps/gf/tail.hpp"

class OneIndexGFTest : public ::testing::Test
{
  public:
    const double beta;
    const int nfreq ;
    alps::gf::omega_gf gf;
    alps::gf::omega_gf gf2;
    typedef alps::gf::matsubara_mesh<alps::gf::mesh::POSITIVE_ONLY> matsubara_mesh;

    OneIndexGFTest():beta(10), nfreq(10),
             gf(matsubara_mesh(beta,nfreq)),
             gf2(gf) {}
};
    
