/*
 * Copyright (C) 1998-2016 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include "alps/params.hpp"
#include "gtest/gtest.h"

TEST(param,CompareConst)
{
  alps::params p;
  p["1"]=1;
  p["1.25"]=1.25;
  p["str"]="1.25";

  EXPECT_ANY_THROW( if (p["str"]!=1.25) {} );
  
  EXPECT_TRUE(p["1"]==1);
  EXPECT_TRUE(p["1"]<=1);
  EXPECT_TRUE(p["1"]>=1);
  EXPECT_TRUE(p["1"]< 1.5);
  EXPECT_TRUE(p["1"]<=1.5);
  EXPECT_TRUE(p["1"]!=1.5);
  EXPECT_TRUE(p["1"]> 0.9);
  EXPECT_TRUE(p["1"]>=0.9);

  EXPECT_TRUE(1  ==p["1"]);
  EXPECT_TRUE(1  <=p["1"]);
  EXPECT_TRUE(1  >=p["1"]);
  EXPECT_TRUE(1.5> p["1"]);
  EXPECT_TRUE(1.5>=p["1"]);
  EXPECT_TRUE(1.5!=p["1"]);
  EXPECT_TRUE(0.9< p["1"]);
  EXPECT_TRUE(0.9<=p["1"]);
  
  EXPECT_TRUE(p["1.25"]==1.25);
  EXPECT_TRUE(p["1.25"]<=1.25);
  EXPECT_TRUE(p["1.25"]>=1.25);
  EXPECT_TRUE(p["1.25"]< 1.5);
  EXPECT_TRUE(p["1.25"]<=1.5);
  EXPECT_TRUE(p["1.25"]!=1.5);
  EXPECT_TRUE(p["1.25"]> 0.9);
  EXPECT_TRUE(p["1.25"]>=0.9);

  EXPECT_TRUE(1.25  ==p["1.25"]);
  EXPECT_TRUE(1.25  <=p["1.25"]);
  EXPECT_TRUE(1.25  >=p["1.25"]);
  EXPECT_TRUE(1.5> p["1.25"]);
  EXPECT_TRUE(1.5>=p["1.25"]);
  EXPECT_TRUE(1.5!=p["1.25"]);
  EXPECT_TRUE(0.9< p["1.25"]);
  EXPECT_TRUE(0.9<=p["1.25"]);
  
  EXPECT_FALSE(p["1"]!=1);
  EXPECT_FALSE(p["1"]>1);
  EXPECT_FALSE(p["1"]<1);
  EXPECT_FALSE(p["1"]>=1.5);
  EXPECT_FALSE(p["1"]> 1.5);
  EXPECT_FALSE(p["1"]==1.5);
  EXPECT_FALSE(p["1"]<=0.9);
  EXPECT_FALSE(p["1"]< 0.9);

  EXPECT_FALSE(1  !=p["1"]);
  EXPECT_FALSE(1  > p["1"]);
  EXPECT_FALSE(1  < p["1"]);
  EXPECT_FALSE(1.5<=p["1"]);
  EXPECT_FALSE(1.5< p["1"]);
  EXPECT_FALSE(1.5==p["1"]);
  EXPECT_FALSE(0.9>=p["1"]);
  EXPECT_FALSE(0.9> p["1"]);
  
  EXPECT_FALSE(p["1.25"]!=1.25);
  EXPECT_FALSE(p["1.25"]> 1.25);
  EXPECT_FALSE(p["1.25"]< 1.25);
  EXPECT_FALSE(p["1.25"]>=1.5);
  EXPECT_FALSE(p["1.25"]> 1.5);
  EXPECT_FALSE(p["1.25"]==1.5);
  EXPECT_FALSE(p["1.25"]<=0.9);
  EXPECT_FALSE(p["1.25"]< 0.9);

  EXPECT_FALSE(1.25  !=p["1.25"]);
  EXPECT_FALSE(1.25  > p["1.25"]);
  EXPECT_FALSE(1.25  < p["1.25"]);
  EXPECT_FALSE(1.5<=p["1.25"]);
  EXPECT_FALSE(1.5< p["1.25"]);
  EXPECT_FALSE(1.5==p["1.25"]);
  EXPECT_FALSE(0.9>=p["1.25"]);
  EXPECT_FALSE(0.9> p["1.25"]);
}

/* FIXME: The following test does not compile. Do we need param<=>param comparison? */
// TEST(param,CompareParams)
// {
//   alps::params p,p1;
//   p["1"]=1;
//   p["2"]=2;
//   p1["2"]=2;
//   p["3"]=3;
  
//   p["2.0"]=2.0;
//   p["2.5"]=2.5;
//   p["1.5"]=1.5;
  
//   p["str"]="1.5";

//   // Incompatible types:
//   EXPECT_ANY_THROW(p["str"]!=p["1.0"]);
  
//   // Same type:
//   EXPECT_TRUE(p1["2"]==p["2"]);
//   EXPECT_TRUE(p["2"]==p["2"]);
//   EXPECT_TRUE(p["2"]<=p["2"]);
//   EXPECT_TRUE(p["2"]>=p["2"]);
//   EXPECT_TRUE(p["2"]> p["1"]);
//   EXPECT_TRUE(p["2"]>=p["1"]);
//   EXPECT_TRUE(p["2"]< p["3"]);
//   EXPECT_TRUE(p["2"]<=p["3"]);
//   EXPECT_TRUE(p["2"]!=p["3"]);
//   // Same type, negation
//   EXPECT_FALSE(p["2"]!=p["2"]);
//   EXPECT_FALSE(p["2"]> p["2"]);
//   EXPECT_FALSE(p["2"]< p["2"]);
//   EXPECT_FALSE(p["2"]<=p["1"]);
//   EXPECT_FALSE(p["2"]< p["1"]);
//   EXPECT_FALSE(p["2"]>=p["3"]);
//   EXPECT_FALSE(p["2"]> p["3"]);
//   EXPECT_FALSE(p["2"]==p["3"]);

//   // Different types, larger-type RHS
//   EXPECT_TRUE(p["2"]==p["2.0"]);
//   EXPECT_TRUE(p["2"]<=p["2.0"]);
//   EXPECT_TRUE(p["2"]>=p["2.0"]);
//   EXPECT_TRUE(p["2"]> p["1.5"]);
//   EXPECT_TRUE(p["2"]>=p["1.5"]);
//   EXPECT_TRUE(p["2"]< p["2.5"]);
//   EXPECT_TRUE(p["2"]<=p["2.5"]);
//   EXPECT_TRUE(p["2"]!=p["2.5"]);
//   // Different types, larger-type RHS, negation
//   EXPECT_FALSE(p["2"]!=p["2.0"]);
//   EXPECT_FALSE(p["2"]> p["2.0"]);
//   EXPECT_FALSE(p["2"]< p["2.0"]);
//   EXPECT_FALSE(p["2"]<=p["1.5"]);
//   EXPECT_FALSE(p["2"]< p["1.5"]);
//   EXPECT_FALSE(p["2"]>=p["2.5"]);
//   EXPECT_FALSE(p["2"]> p["2.5"]);
//   EXPECT_FALSE(p["2"]==p["2.5"]);

//   // Different type, larger-type LHS 
//   EXPECT_TRUE(p["2.0"]==p["2"]);
//   EXPECT_TRUE(p["2.0"]<=p["2"]);
//   EXPECT_TRUE(p["2.0"]>=p["2"]);
//   EXPECT_TRUE(p["2.5"]> p["1"]);
//   EXPECT_TRUE(p["2.5"]>=p["1"]);
//   EXPECT_TRUE(p["2.5"]< p["3"]);
//   EXPECT_TRUE(p["2.5"]<=p["3"]);
//   EXPECT_TRUE(p["2.5"]!=p["3"]);
//   // Different type, larger-type LHS, negation 
//   EXPECT_FALSE(p["2.0"]!=p["2"]);
//   EXPECT_FALSE(p["2.0"]> p["2"]);
//   EXPECT_FALSE(p["2.0"]< p["2"]);
//   EXPECT_FALSE(p["2.5"]<=p["1"]);
//   EXPECT_FALSE(p["2.5"]< p["1"]);
//   EXPECT_FALSE(p["2.5"]>=p["3"]);
//   EXPECT_FALSE(p["2.5"]> p["3"]);
//   EXPECT_FALSE(p["2.5"]==p["3"]);
// }
