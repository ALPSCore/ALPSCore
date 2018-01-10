/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include "gtest/gtest.h"
#include "alps/gf/mesh.hpp"

TEST(Index, UnaryAndComparisonOperators){
    alps::gf::matsubara_mesh<alps::gf::mesh::POSITIVE_NEGATIVE> mesh(5.0, 20);
    alps::gf::matsubara_index omega; omega=5;
   alps::gf::matsubara_index omega5=omega++;
   alps::gf::matsubara_index omega7=++omega;
   EXPECT_EQ(7, omega);
   EXPECT_EQ(5, omega5);
   EXPECT_EQ(7, omega7);
   
   omega+=1;
   EXPECT_EQ(8, omega);
   omega-=3;
   EXPECT_EQ(5, omega);
   --omega;
   EXPECT_EQ(4, omega);
   EXPECT_LT(omega,5);
   EXPECT_LE(omega,4);
   EXPECT_GT(omega,3);
   EXPECT_GE(omega,4);

   EXPECT_GT(5,omega);
   EXPECT_GE(4,omega);
   EXPECT_LT(3,omega);
   EXPECT_LE(4,omega);
}

TEST(Index, BinaryOperators){
   alps::gf::matsubara_mesh<alps::gf::mesh::POSITIVE_NEGATIVE> mesh(5.0, 20);
   alps::gf::matsubara_index omega; omega=5;
   alps::gf::matsubara_index omegaprime=omega+11;
   alps::gf::matsubara_index omegaprime1=11+omega;
   alps::gf::matsubara_index omegaprime2=omega-11;

   EXPECT_EQ(5, omega);

   EXPECT_EQ(16, omegaprime);
   EXPECT_EQ(16, omegaprime1);

   EXPECT_EQ(-6, omegaprime2);
}

