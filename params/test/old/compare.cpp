/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/params.hpp>
#include <gtest/gtest.h>

class ParamTestCompare : public ::testing::Test {
  public:
    alps::params p;

    ParamTestCompare() {
        p["1"]=1;
        p["1.25"]=1.25;
        p["1.00"]=1.0;
        p["str"]="1.25";
    }
};

TEST_F(ParamTestCompare,SameTypeLiteralRHS) {
    // 9 possible comparizons with less-than, equal, and more-than values: ints
    EXPECT_TRUE( p["1"]> 0);
    EXPECT_TRUE( p["1"]>=0);
    EXPECT_TRUE( p["1"]!=0);
    EXPECT_TRUE( p["1"]>=1);
    EXPECT_TRUE( p["1"]<=1);
    EXPECT_TRUE( p["1"]==1);
    EXPECT_TRUE( p["1"]< 2);
    EXPECT_TRUE( p["1"]<=2);
    EXPECT_TRUE( p["1"]!=2);

    // Negation of the 9 comparizons: ints
    EXPECT_FALSE(p["1"]<=0);
    EXPECT_FALSE(p["1"]< 0);
    EXPECT_FALSE(p["1"]==0);
    EXPECT_FALSE(p["1"]< 1);
    EXPECT_FALSE(p["1"]> 1);
    EXPECT_FALSE(p["1"]!=1);
    EXPECT_FALSE(p["1"]>=2);
    EXPECT_FALSE(p["1"]> 2);
    EXPECT_FALSE(p["1"]==2);

    // 9 possible comparizons with less-than, equal, and more-than values: doubles
    EXPECT_TRUE( p["1.25"]> 0.5);
    EXPECT_TRUE( p["1.25"]>=0.5);
    EXPECT_TRUE( p["1.25"]!=0.5);
    EXPECT_TRUE( p["1.25"]>=1.25);
    EXPECT_TRUE( p["1.25"]<=1.25);
    EXPECT_TRUE( p["1.25"]==1.25);
    EXPECT_TRUE( p["1.25"]< 1.50);
    EXPECT_TRUE( p["1.25"]<=1.50);
    EXPECT_TRUE( p["1.25"]!=1.50);

    // Negation of the 9 comparizons: doubles
    EXPECT_FALSE(p["1.25"]<=0.5);
    EXPECT_FALSE(p["1.25"]< 0.5);
    EXPECT_FALSE(p["1.25"]==0.5);
    EXPECT_FALSE(p["1.25"]< 1.25);
    EXPECT_FALSE(p["1.25"]> 1.25);
    EXPECT_FALSE(p["1.25"]!=1.25);
    EXPECT_FALSE(p["1.25"]>=1.50);
    EXPECT_FALSE(p["1.25"]> 1.50);
    EXPECT_FALSE(p["1.25"]==1.50);

    // There are no std::string literals, see the corresponding test for comparison with char-string literals
}

TEST_F(ParamTestCompare,SameTypeLiteralLHS) {
    // 9 possible comparizons with less-than, equal, and more-than values: ints
    EXPECT_TRUE( 0< p["1"]);
    EXPECT_TRUE( 0<=p["1"]);
    EXPECT_TRUE( 0!=p["1"]);
    EXPECT_TRUE( 1<=p["1"]);
    EXPECT_TRUE( 1>=p["1"]);
    EXPECT_TRUE( 1==p["1"]);
    EXPECT_TRUE( 2> p["1"]);
    EXPECT_TRUE( 2>=p["1"]);
    EXPECT_TRUE( 2!=p["1"]);

    // Negation of the 9 comparizons: ints
    EXPECT_FALSE(0>=p["1"]);
    EXPECT_FALSE(0> p["1"]);
    EXPECT_FALSE(0==p["1"]);
    EXPECT_FALSE(1> p["1"]);
    EXPECT_FALSE(1< p["1"]);
    EXPECT_FALSE(1!=p["1"]);
    EXPECT_FALSE(2<=p["1"]);
    EXPECT_FALSE(2< p["1"]);
    EXPECT_FALSE(2==p["1"]);

    // 9 possible comparizons with less-than, equal, and more-than values: doubles
    EXPECT_TRUE( 0.50< p["1.25"]);
    EXPECT_TRUE( 0.50<=p["1.25"]);
    EXPECT_TRUE( 0.50!=p["1.25"]);
    EXPECT_TRUE( 1.25<=p["1.25"]);
    EXPECT_TRUE( 1.25>=p["1.25"]);
    EXPECT_TRUE( 1.25==p["1.25"]);
    EXPECT_TRUE( 1.50> p["1.25"]);
    EXPECT_TRUE( 1.50>=p["1.25"]);
    EXPECT_TRUE( 1.50!=p["1.25"]);

    // Negation of the 9 comparizons: doubles
    EXPECT_FALSE(0.50>=p["1.25"]);
    EXPECT_FALSE(0.50> p["1.25"]);
    EXPECT_FALSE(0.50==p["1.25"]);
    EXPECT_FALSE(1.25> p["1.25"]);
    EXPECT_FALSE(1.25< p["1.25"]);
    EXPECT_FALSE(1.25!=p["1.25"]);
    EXPECT_FALSE(1.50<=p["1.25"]);
    EXPECT_FALSE(1.50< p["1.25"]);
    EXPECT_FALSE(1.50==p["1.25"]);

    // There are no std::string literals, see the corresponding test for comparison with char-string literals
}

TEST_F(ParamTestCompare,SameTypeVarRHS) {
    {
        // 9 possible comparizons with less-than, equal, and more-than values: ints
        int less_than=0;
        int more_than=2;
        int equal=1;
        EXPECT_TRUE( p["1"]> less_than);
        EXPECT_TRUE( p["1"]>=less_than);
        EXPECT_TRUE( p["1"]!=less_than);
        EXPECT_TRUE( p["1"]>=equal);
        EXPECT_TRUE( p["1"]<=equal);
        EXPECT_TRUE( p["1"]==equal);
        EXPECT_TRUE( p["1"]< more_than);
        EXPECT_TRUE( p["1"]<=more_than);
        EXPECT_TRUE( p["1"]!=more_than);

        // Negation of the 9 comparizons: ints
        EXPECT_FALSE(p["1"]<=less_than);
        EXPECT_FALSE(p["1"]< less_than);
        EXPECT_FALSE(p["1"]==less_than);
        EXPECT_FALSE(p["1"]< equal);
        EXPECT_FALSE(p["1"]> equal);
        EXPECT_FALSE(p["1"]!=equal);
        EXPECT_FALSE(p["1"]>=more_than);
        EXPECT_FALSE(p["1"]> more_than);
        EXPECT_FALSE(p["1"]==more_than);
    }

    {
        // 9 possible comparizons with less-than, equal, and more-than values: doubles
        double less_than=0.5;
        double more_than=1.5;
        double equal=1.25;
        
        EXPECT_TRUE( p["1.25"]> less_than);
        EXPECT_TRUE( p["1.25"]>=less_than);
        EXPECT_TRUE( p["1.25"]!=less_than);
        EXPECT_TRUE( p["1.25"]>=equal);
        EXPECT_TRUE( p["1.25"]<=equal);
        EXPECT_TRUE( p["1.25"]==equal);
        EXPECT_TRUE( p["1.25"]< more_than);
        EXPECT_TRUE( p["1.25"]<=more_than);
        EXPECT_TRUE( p["1.25"]!=more_than);

        // Negation of the 9 comparizons: doubles
        EXPECT_FALSE(p["1.25"]<=less_than);
        EXPECT_FALSE(p["1.25"]< less_than);
        EXPECT_FALSE(p["1.25"]==less_than);
        EXPECT_FALSE(p["1.25"]< equal);
        EXPECT_FALSE(p["1.25"]> equal);
        EXPECT_FALSE(p["1.25"]!=equal);
        EXPECT_FALSE(p["1.25"]>=more_than);
        EXPECT_FALSE(p["1.25"]> more_than);
        EXPECT_FALSE(p["1.25"]==more_than);
    }

    {
        // 9 possible comparizons with less-than, equal, and more-than values: strings
        std::string less_than="1.2";
        std::string more_than="2xx";
        std::string equal="1.25";
        
        EXPECT_TRUE( p["str"]> less_than);
        EXPECT_TRUE( p["str"]>=less_than);
        EXPECT_TRUE( p["str"]!=less_than);
        EXPECT_TRUE( p["str"]>=equal);
        EXPECT_TRUE( p["str"]<=equal);
        EXPECT_TRUE( p["str"]==equal);
        EXPECT_TRUE( p["str"]< more_than);
        EXPECT_TRUE( p["str"]<=more_than);
        EXPECT_TRUE( p["str"]!=more_than);

        // Negation of the 9 comparizons: strings
        EXPECT_FALSE(p["str"]<=less_than);
        EXPECT_FALSE(p["str"]< less_than);
        EXPECT_FALSE(p["str"]==less_than);
        EXPECT_FALSE(p["str"]< equal);
        EXPECT_FALSE(p["str"]> equal);
        EXPECT_FALSE(p["str"]!=equal);
        EXPECT_FALSE(p["str"]>=more_than);
        EXPECT_FALSE(p["str"]> more_than);
        EXPECT_FALSE(p["str"]==more_than);
    }
}

TEST_F(ParamTestCompare,SameTypeVarLHS) {
    {
        // 9 possible comparizons with less-than, equal, and more-than values: ints
        int less_than=0;
        int more_than=2;
        int equal=1;
        EXPECT_TRUE( less_than< p["1"]);
        EXPECT_TRUE( less_than<=p["1"]);
        EXPECT_TRUE( less_than!=p["1"]);
        EXPECT_TRUE( equal    <=p["1"]);
        EXPECT_TRUE( equal    >=p["1"]);
        EXPECT_TRUE( equal    ==p["1"]);
        EXPECT_TRUE( more_than> p["1"]);
        EXPECT_TRUE( more_than>=p["1"]);
        EXPECT_TRUE( more_than!=p["1"]);
                       
        // Negation of the 9 comparizons: ints
        EXPECT_FALSE(less_than>=p["1"]);
        EXPECT_FALSE(less_than> p["1"]);
        EXPECT_FALSE(less_than==p["1"]);
        EXPECT_FALSE(equal    > p["1"]);
        EXPECT_FALSE(equal    < p["1"]);
        EXPECT_FALSE(equal    !=p["1"]);
        EXPECT_FALSE(more_than<=p["1"]);
        EXPECT_FALSE(more_than< p["1"]);
        EXPECT_FALSE(more_than==p["1"]);
    }

    {
        // 9 possible comparizons with less-than, equal, and more-than values: doubles
        double less_than=0.5;
        double more_than=1.5;
        double equal=1.25;
        
        EXPECT_TRUE( less_than< p["1.25"]);
        EXPECT_TRUE( less_than<=p["1.25"]);
        EXPECT_TRUE( less_than!=p["1.25"]);
        EXPECT_TRUE( equal    <=p["1.25"]);
        EXPECT_TRUE( equal    >=p["1.25"]);
        EXPECT_TRUE( equal    ==p["1.25"]);
        EXPECT_TRUE( more_than> p["1.25"]);
        EXPECT_TRUE( more_than>=p["1.25"]);
        EXPECT_TRUE( more_than!=p["1.25"]);
                       
        // Negation of the 9 comparizons: doubles
        EXPECT_FALSE(less_than>=p["1.25"]);
        EXPECT_FALSE(less_than> p["1.25"]);
        EXPECT_FALSE(less_than==p["1.25"]);
        EXPECT_FALSE(equal    > p["1.25"]);
        EXPECT_FALSE(equal    < p["1.25"]);
        EXPECT_FALSE(equal    !=p["1.25"]);
        EXPECT_FALSE(more_than<=p["1.25"]);
        EXPECT_FALSE(more_than< p["1.25"]);
        EXPECT_FALSE(more_than==p["1.25"]);
    }

    {
        // 9 possible comparizons with less-than, equal, and more-than values: strings
        std::string less_than="1.2";
        std::string more_than="2xx";
        std::string equal="1.25";
        
        EXPECT_TRUE( less_than< p["str"]);
        EXPECT_TRUE( less_than<=p["str"]);
        EXPECT_TRUE( less_than!=p["str"]);
        EXPECT_TRUE( equal    <=p["str"]);
        EXPECT_TRUE( equal    >=p["str"]);
        EXPECT_TRUE( equal    ==p["str"]);
        EXPECT_TRUE( more_than> p["str"]);
        EXPECT_TRUE( more_than>=p["str"]);
        EXPECT_TRUE( more_than!=p["str"]);
                       
        // Negation of the 9 comparizons: strings
        EXPECT_FALSE(less_than>=p["str"]);
        EXPECT_FALSE(less_than> p["str"]);
        EXPECT_FALSE(less_than==p["str"]);
        EXPECT_FALSE(equal    > p["str"]);
        EXPECT_FALSE(equal    < p["str"]);
        EXPECT_FALSE(equal    !=p["str"]);
        EXPECT_FALSE(more_than<=p["str"]);
        EXPECT_FALSE(more_than< p["str"]);
        EXPECT_FALSE(more_than==p["str"]);
    }
}

TEST_F(ParamTestCompare,SimilarTypeLiteralRHS) {
    // 9 possible comparizons with less-than, equal, and more-than values: ints vs doubles
    EXPECT_TRUE( p["1"]> 0.5);
    EXPECT_TRUE( p["1"]>=0.5);
    EXPECT_TRUE( p["1"]!=0.5);
    EXPECT_TRUE( p["1"]>=1.0);
    EXPECT_TRUE( p["1"]<=1.0);
    EXPECT_TRUE( p["1"]==1.0);
    EXPECT_TRUE( p["1"]< 2.5);
    EXPECT_TRUE( p["1"]<=2.5);
    EXPECT_TRUE( p["1"]!=2.5);

    // Negation of the 9 comparizons: ints vs doubles
    EXPECT_FALSE(p["1"]<=0.5);
    EXPECT_FALSE(p["1"]< 0.5);
    EXPECT_FALSE(p["1"]==0.5);
    EXPECT_FALSE(p["1"]< 1.0);
    EXPECT_FALSE(p["1"]> 1.0);
    EXPECT_FALSE(p["1"]!=1.0);
    EXPECT_FALSE(p["1"]>=2.5);
    EXPECT_FALSE(p["1"]> 2.5);
    EXPECT_FALSE(p["1"]==2.5);

    // 9 possible comparizons with less-than, equal, and more-than values: doubles vs ints
    EXPECT_THROW(p["1.25"]> 0, alps::params::type_mismatch);
    EXPECT_THROW(p["1.25"]>=0, alps::params::type_mismatch);
    EXPECT_THROW(p["1.25"]!=0, alps::params::type_mismatch);
    EXPECT_THROW(p["1.00"]>=1, alps::params::type_mismatch);
    EXPECT_THROW(p["1.00"]<=1, alps::params::type_mismatch);
    EXPECT_THROW(p["1.00"]==1, alps::params::type_mismatch);
    EXPECT_THROW(p["1.25"]< 2, alps::params::type_mismatch);
    EXPECT_THROW(p["1.25"]<=2, alps::params::type_mismatch);
    EXPECT_THROW(p["1.25"]!=2, alps::params::type_mismatch);

    // Negation of the 9 comparizons: doubles vs ints
    EXPECT_THROW(p["1.25"]<=0, alps::params::type_mismatch);
    EXPECT_THROW(p["1.25"]< 0, alps::params::type_mismatch);
    EXPECT_THROW(p["1.25"]==0, alps::params::type_mismatch);
    EXPECT_THROW(p["1.00"]< 1, alps::params::type_mismatch);
    EXPECT_THROW(p["1.00"]> 1, alps::params::type_mismatch);
    EXPECT_THROW(p["1.00"]!=1, alps::params::type_mismatch);
    EXPECT_THROW(p["1.25"]>=2, alps::params::type_mismatch);
    EXPECT_THROW(p["1.25"]> 2, alps::params::type_mismatch);
    EXPECT_THROW(p["1.25"]==2, alps::params::type_mismatch);

    // 9 possible comparizons with less-than, equal, and more-than values: std::strings vs "char-strings"
    EXPECT_TRUE( p["str"]> "1.2");
    EXPECT_TRUE( p["str"]>="1.2");
    EXPECT_TRUE( p["str"]!="1.2");
    EXPECT_TRUE( p["str"]>="1.25");
    EXPECT_TRUE( p["str"]<="1.25");
    EXPECT_TRUE( p["str"]=="1.25");
    EXPECT_TRUE( p["str"]< "2xx");
    EXPECT_TRUE( p["str"]<="2xx");
    EXPECT_TRUE( p["str"]!="2xx");

    // Negation of the 9 comparizons: std::strings vs "char-strings"
    EXPECT_FALSE(p["str"]<="1.2");
    EXPECT_FALSE(p["str"]< "1.2");
    EXPECT_FALSE(p["str"]=="1.2");
    EXPECT_FALSE(p["str"]< "1.25");
    EXPECT_FALSE(p["str"]> "1.25");
    EXPECT_FALSE(p["str"]!="1.25");
    EXPECT_FALSE(p["str"]>="2xx");
    EXPECT_FALSE(p["str"]> "2xx");
    EXPECT_FALSE(p["str"]=="2xx");
}

TEST_F(ParamTestCompare,SimilarTypeLiteralLHS) {
    // 9 possible comparizons with less-than, equal, and more-than values: ints vs doubles
    EXPECT_TRUE( 0.5< p["1"]);
    EXPECT_TRUE( 0.5<=p["1"]);
    EXPECT_TRUE( 0.5!=p["1"]);
    EXPECT_TRUE( 1.0<=p["1"]);
    EXPECT_TRUE( 1.0>=p["1"]);
    EXPECT_TRUE( 1.0==p["1"]);
    EXPECT_TRUE( 2.5> p["1"]);
    EXPECT_TRUE( 2.5>=p["1"]);
    EXPECT_TRUE( 2.5!=p["1"]);

    // Negation of the 9 comparizons: ints vs doubles
    EXPECT_FALSE(0.5>=p["1"]);
    EXPECT_FALSE(0.5> p["1"]);
    EXPECT_FALSE(0.5==p["1"]);
    EXPECT_FALSE(1.0> p["1"]);
    EXPECT_FALSE(1.0< p["1"]);
    EXPECT_FALSE(1.0!=p["1"]);
    EXPECT_FALSE(2.5<=p["1"]);
    EXPECT_FALSE(2.5< p["1"]);
    EXPECT_FALSE(2.5==p["1"]);

    // 9 possible comparizons with less-than, equal, and more-than values: doubles vs ints
    EXPECT_THROW( 0< p["1.25"], alps::params::type_mismatch);
    EXPECT_THROW( 0<=p["1.25"], alps::params::type_mismatch);
    EXPECT_THROW( 0!=p["1.25"], alps::params::type_mismatch);
    EXPECT_THROW( 1<=p["1.00"], alps::params::type_mismatch);
    EXPECT_THROW( 1>=p["1.00"], alps::params::type_mismatch);
    EXPECT_THROW( 1==p["1.00"], alps::params::type_mismatch);
    EXPECT_THROW( 2> p["1.25"], alps::params::type_mismatch);
    EXPECT_THROW( 2>=p["1.25"], alps::params::type_mismatch);
    EXPECT_THROW( 2!=p["1.25"], alps::params::type_mismatch);

    // Negation of the 9 comparizons: doubles vs ints
    EXPECT_THROW(0>=p["1.25"], alps::params::type_mismatch);
    EXPECT_THROW(0> p["1.25"], alps::params::type_mismatch);
    EXPECT_THROW(0==p["1.25"], alps::params::type_mismatch);
    EXPECT_THROW(1> p["1.00"], alps::params::type_mismatch);
    EXPECT_THROW(1< p["1.00"], alps::params::type_mismatch);
    EXPECT_THROW(1!=p["1.00"], alps::params::type_mismatch);
    EXPECT_THROW(2<=p["1.25"], alps::params::type_mismatch);
    EXPECT_THROW(2< p["1.25"], alps::params::type_mismatch);
    EXPECT_THROW(2==p["1.25"], alps::params::type_mismatch);

    // 9 possible comparizons with less-than, equal, and more-than values: std::strings vs "char-strings"
    EXPECT_TRUE( "1.2" < p["str"]);
    EXPECT_TRUE( "1.2" <=p["str"]);
    EXPECT_TRUE( "1.2" !=p["str"]);
    EXPECT_TRUE( "1.25"<=p["str"]);
    EXPECT_TRUE( "1.25">=p["str"]);
    EXPECT_TRUE( "1.25"==p["str"]);
    EXPECT_TRUE( "2xx" > p["str"]);
    EXPECT_TRUE( "2xx" >=p["str"]);
    EXPECT_TRUE( "2xx" !=p["str"]);

    // Negation of the 9 comparizons: std::strings vs "char-strings"
    EXPECT_FALSE("1.2" >=p["str"]);
    EXPECT_FALSE("1.2" > p["str"]);
    EXPECT_FALSE("1.2" ==p["str"]);
    EXPECT_FALSE("1.25"> p["str"]);
    EXPECT_FALSE("1.25"< p["str"]);
    EXPECT_FALSE("1.25"!=p["str"]);
    EXPECT_FALSE("2xx" <=p["str"]);
    EXPECT_FALSE("2xx" < p["str"]);
    EXPECT_FALSE("2xx" ==p["str"]);
}

TEST_F(ParamTestCompare,SimilarTypeVarRHS) {
    {
        // 9 possible comparizons with less-than, equal, and more-than values: ints vs doubles
        double less_than=0.5;
        double more_than=2.5;
        double equal=1.0;
        EXPECT_TRUE( p["1"]> less_than);
        EXPECT_TRUE( p["1"]>=less_than);
        EXPECT_TRUE( p["1"]!=less_than);
        EXPECT_TRUE( p["1"]>=equal);
        EXPECT_TRUE( p["1"]<=equal);
        EXPECT_TRUE( p["1"]==equal);
        EXPECT_TRUE( p["1"]< more_than);
        EXPECT_TRUE( p["1"]<=more_than);
        EXPECT_TRUE( p["1"]!=more_than);

        // Negation of the 9 comparizons: ints vs doubles
        EXPECT_FALSE(p["1"]<=less_than);
        EXPECT_FALSE(p["1"]< less_than);
        EXPECT_FALSE(p["1"]==less_than);
        EXPECT_FALSE(p["1"]< equal);
        EXPECT_FALSE(p["1"]> equal);
        EXPECT_FALSE(p["1"]!=equal);
        EXPECT_FALSE(p["1"]>=more_than);
        EXPECT_FALSE(p["1"]> more_than);
        EXPECT_FALSE(p["1"]==more_than);
    }

    {
        // 9 possible comparizons with less-than, equal, and more-than values: doubles vs ints
        int less_than=0;
        int more_than=2;
        int equal=1;
        
        EXPECT_THROW( p["1.25"]> less_than, alps::params::type_mismatch);
        EXPECT_THROW( p["1.25"]>=less_than, alps::params::type_mismatch);
        EXPECT_THROW( p["1.25"]!=less_than, alps::params::type_mismatch);
        EXPECT_THROW( p["1.00"]>=equal    , alps::params::type_mismatch);
        EXPECT_THROW( p["1.00"]<=equal    , alps::params::type_mismatch);
        EXPECT_THROW( p["1.00"]==equal    , alps::params::type_mismatch);
        EXPECT_THROW( p["1.25"]< more_than, alps::params::type_mismatch);
        EXPECT_THROW( p["1.25"]<=more_than, alps::params::type_mismatch);
        EXPECT_THROW( p["1.25"]!=more_than, alps::params::type_mismatch);

        // Negation of the 9 comparizons: doubles
        EXPECT_THROW(p["1.25"]<=less_than, alps::params::type_mismatch);
        EXPECT_THROW(p["1.25"]< less_than, alps::params::type_mismatch);
        EXPECT_THROW(p["1.25"]==less_than, alps::params::type_mismatch);
        EXPECT_THROW(p["1.00"]< equal    , alps::params::type_mismatch);
        EXPECT_THROW(p["1.00"]> equal    , alps::params::type_mismatch);
        EXPECT_THROW(p["1.00"]!=equal    , alps::params::type_mismatch);
        EXPECT_THROW(p["1.25"]>=more_than, alps::params::type_mismatch);
        EXPECT_THROW(p["1.25"]> more_than, alps::params::type_mismatch);
        EXPECT_THROW(p["1.25"]==more_than, alps::params::type_mismatch);
    }

    {
        // 9 possible comparizons with less-than, equal, and more-than values: std::strings vs "char-strings"
        const char* less_than="1.2";
        const char* more_than="2xx";
        const char* equal="1.25";
        
        EXPECT_TRUE( p["str"]> less_than);
        EXPECT_TRUE( p["str"]>=less_than);
        EXPECT_TRUE( p["str"]!=less_than);
        EXPECT_TRUE( p["str"]>=equal);
        EXPECT_TRUE( p["str"]<=equal);
        EXPECT_TRUE( p["str"]==equal);
        EXPECT_TRUE( p["str"]< more_than);
        EXPECT_TRUE( p["str"]<=more_than);
        EXPECT_TRUE( p["str"]!=more_than);

        // Negation of the 9 comparizons: std::strings vs "char-strings"
        EXPECT_FALSE(p["str"]<=less_than);
        EXPECT_FALSE(p["str"]< less_than);
        EXPECT_FALSE(p["str"]==less_than);
        EXPECT_FALSE(p["str"]< equal);
        EXPECT_FALSE(p["str"]> equal);
        EXPECT_FALSE(p["str"]!=equal);
        EXPECT_FALSE(p["str"]>=more_than);
        EXPECT_FALSE(p["str"]> more_than);
        EXPECT_FALSE(p["str"]==more_than);
    }
}

TEST_F(ParamTestCompare,SimilarTypeVarLHS) {
    {
        // 9 possible comparizons with less-than, equal, and more-than values: ints vs doubles
        double less_than=0.5;
        double more_than=2.5;
        double equal=1.0;
        EXPECT_TRUE( less_than< p["1"]);
        EXPECT_TRUE( less_than<=p["1"]);
        EXPECT_TRUE( less_than!=p["1"]);
        EXPECT_TRUE( equal    <=p["1"]);
        EXPECT_TRUE( equal    >=p["1"]);
        EXPECT_TRUE( equal    ==p["1"]);
        EXPECT_TRUE( more_than> p["1"]);
        EXPECT_TRUE( more_than>=p["1"]);
        EXPECT_TRUE( more_than!=p["1"]);
                       
        // Negation of the 9 comparizons: ints vs doubles
        EXPECT_FALSE(less_than>=p["1"]);
        EXPECT_FALSE(less_than> p["1"]);
        EXPECT_FALSE(less_than==p["1"]);
        EXPECT_FALSE(equal    > p["1"]);
        EXPECT_FALSE(equal    < p["1"]);
        EXPECT_FALSE(equal    !=p["1"]);
        EXPECT_FALSE(more_than<=p["1"]);
        EXPECT_FALSE(more_than< p["1"]);
        EXPECT_FALSE(more_than==p["1"]);
    }

    {
        // 9 possible comparizons with less-than, equal, and more-than values: doubles vs ints
        int less_than=0;
        int more_than=2;
        int equal=1;
        
        EXPECT_THROW( less_than< p["1.25"], alps::params::type_mismatch);
        EXPECT_THROW( less_than<=p["1.25"], alps::params::type_mismatch);
        EXPECT_THROW( less_than!=p["1.25"], alps::params::type_mismatch);
        EXPECT_THROW( equal    <=p["1.00"], alps::params::type_mismatch);
        EXPECT_THROW( equal    >=p["1.00"], alps::params::type_mismatch);
        EXPECT_THROW( equal    ==p["1.00"], alps::params::type_mismatch);
        EXPECT_THROW( more_than> p["1.25"], alps::params::type_mismatch);
        EXPECT_THROW( more_than>=p["1.25"], alps::params::type_mismatch);
        EXPECT_THROW( more_than!=p["1.25"], alps::params::type_mismatch);
                       
        // Negation of the 9 comparizons: doubles vs ints
        EXPECT_THROW(less_than>=p["1.25"], alps::params::type_mismatch);
        EXPECT_THROW(less_than> p["1.25"], alps::params::type_mismatch);
        EXPECT_THROW(less_than==p["1.25"], alps::params::type_mismatch);
        EXPECT_THROW(equal    > p["1.00"], alps::params::type_mismatch);
        EXPECT_THROW(equal    < p["1.00"], alps::params::type_mismatch);
        EXPECT_THROW(equal    !=p["1.00"], alps::params::type_mismatch);
        EXPECT_THROW(more_than<=p["1.25"], alps::params::type_mismatch);
        EXPECT_THROW(more_than< p["1.25"], alps::params::type_mismatch);
        EXPECT_THROW(more_than==p["1.25"], alps::params::type_mismatch);
    }

    {
        // 9 possible comparizons with less-than, equal, and more-than values: std::strings vs "char-strings"
        const char* less_than="1.2";
        const char* more_than="2xx";
        const char* equal="1.25";
        
        EXPECT_TRUE( less_than< p["str"]);
        EXPECT_TRUE( less_than<=p["str"]);
        EXPECT_TRUE( less_than!=p["str"]);
        EXPECT_TRUE( equal    <=p["str"]);
        EXPECT_TRUE( equal    >=p["str"]);
        EXPECT_TRUE( equal    ==p["str"]);
        EXPECT_TRUE( more_than> p["str"]);
        EXPECT_TRUE( more_than>=p["str"]);
        EXPECT_TRUE( more_than!=p["str"]);
                       
        // Negation of the 9 comparizons: std::strings vs "char-strings"
        EXPECT_FALSE(less_than>=p["str"]);
        EXPECT_FALSE(less_than> p["str"]);
        EXPECT_FALSE(less_than==p["str"]);
        EXPECT_FALSE(equal    > p["str"]);
        EXPECT_FALSE(equal    < p["str"]);
        EXPECT_FALSE(equal    !=p["str"]);
        EXPECT_FALSE(more_than<=p["str"]);
        EXPECT_FALSE(more_than< p["str"]);
        EXPECT_FALSE(more_than==p["str"]);
    }
}

TEST_F(ParamTestCompare,IncompatTypeRHS) {
    // 6 possible comparizons with a value: ints vs strings
    EXPECT_THROW( p["1"]> "0", alps::params::type_mismatch);
    EXPECT_THROW( p["1"]>="0", alps::params::type_mismatch);
    EXPECT_THROW( p["1"]< "2", alps::params::type_mismatch);
    EXPECT_THROW( p["1"]<="2", alps::params::type_mismatch);
    EXPECT_THROW( p["1"]=="1", alps::params::type_mismatch);
    EXPECT_THROW( p["1"]!="2", alps::params::type_mismatch);

    // 6 possible comparizons with a value: doubles vs strings
    EXPECT_THROW( p["1.25"]> "0",    alps::params::type_mismatch);
    EXPECT_THROW( p["1.25"]>="0",    alps::params::type_mismatch);
    EXPECT_THROW( p["1.25"]< "2",    alps::params::type_mismatch);
    EXPECT_THROW( p["1.25"]<="2",    alps::params::type_mismatch);
    EXPECT_THROW( p["1.25"]=="1.25", alps::params::type_mismatch);
    EXPECT_THROW( p["1.25"]!="2",    alps::params::type_mismatch);

    // 6 possible comparizons with a value: strings vs doubles
    EXPECT_THROW( p["str"]> 0.5,  alps::params::type_mismatch);
    EXPECT_THROW( p["str"]>=0.5,  alps::params::type_mismatch);
    EXPECT_THROW( p["str"]< 1.5,  alps::params::type_mismatch);
    EXPECT_THROW( p["str"]<=1.5,  alps::params::type_mismatch);
    EXPECT_THROW( p["str"]==1.25, alps::params::type_mismatch);
    EXPECT_THROW( p["str"]!=1.5,  alps::params::type_mismatch);
}

TEST_F(ParamTestCompare,IncompatTypeLHS) {
    // 6 possible comparizons with a value: ints vs strings
    EXPECT_THROW("0"< p["1"], alps::params::type_mismatch);
    EXPECT_THROW("0"<=p["1"], alps::params::type_mismatch);
    EXPECT_THROW("2"> p["1"], alps::params::type_mismatch);
    EXPECT_THROW("2">=p["1"], alps::params::type_mismatch);
    EXPECT_THROW("1"==p["1"], alps::params::type_mismatch);
    EXPECT_THROW("2"!=p["1"], alps::params::type_mismatch);

    // 6 possible comparizons with a value: doubles vs strings
    EXPECT_THROW("0"   < p["1.25"], alps::params::type_mismatch);
    EXPECT_THROW("0"   <=p["1.25"], alps::params::type_mismatch);
    EXPECT_THROW("2"   > p["1.25"], alps::params::type_mismatch);
    EXPECT_THROW("2"   >=p["1.25"], alps::params::type_mismatch);
    EXPECT_THROW("1.25"==p["1.25"], alps::params::type_mismatch);
    EXPECT_THROW("2"   !=p["1.25"], alps::params::type_mismatch);

    // 6 possible comparizons with a value: strings vs doubles
    EXPECT_THROW(0.5 < p["str"], alps::params::type_mismatch);
    EXPECT_THROW(0.5 <=p["str"], alps::params::type_mismatch);
    EXPECT_THROW(1.5 > p["str"], alps::params::type_mismatch);
    EXPECT_THROW(1.5 >=p["str"], alps::params::type_mismatch);
    EXPECT_THROW(1.25==p["str"], alps::params::type_mismatch);
    EXPECT_THROW(1.5 !=p["str"], alps::params::type_mismatch);
}

