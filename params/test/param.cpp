/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/params.hpp>
#include "gtest/gtest.h"
#include <alps/utility/temporary_filename.hpp>
#include <fstream>

TEST(param, TextParamRead){
   //create a file name
   std::string pfilename(alps::temporary_filename("pfile"));

   //create a few parameters
   int int_parameter=1;
   double double_parameter =2.;
   bool bool_parameter_true=true;
   bool bool_parameter_false=false;
   std::string string_parameter="Hello";
   std::string string_parameter_semic="Hello;";
   std::string complicated_string_parameter="/path/to/nowhere -parameter -run --this -program";
   
   {
     std::ofstream pfile(pfilename.c_str());
     pfile<<"INT = "<<int_parameter<<std::endl;
     pfile<<"DOUBLE = "<<double_parameter<<std::endl;
     pfile<<"STRING = "<<string_parameter<<std::endl;
     pfile<<"STRING_SEMI = "<<string_parameter_semic<<std::endl;
     pfile<<"BOOL_TRUE = "<<bool_parameter_true<<std::endl;
     pfile<<"BOOL_FALSE = "<<bool_parameter_false<<std::endl;
     pfile<<"" <<std::endl; //empty line
     pfile<<" this is a  line with some gunk in it" <<std::endl; //empty line
     pfile<<"complicated_string = "<<complicated_string_parameter<<std::endl; //empty line
   }
   
   //define the parameters and read them
   alps::params p(pfilename);
   
   int int_parameter_read=p["INT"];  
   double double_parameter_read=p["DOUBLE"];  
   std::string string_parameter_read=p["STRING"];
   std::string string_parameter_semi_read=p["STRING_SEMI"];
   std::string complicated_string_parameter_read=p["complicated_string"];
   bool bool_parameter_true_read=p["BOOL_TRUE"];
   bool bool_parameter_false_read=p["BOOL_FALSE"];

   EXPECT_EQ(int_parameter, int_parameter_read);    
   EXPECT_NEAR(double_parameter, double_parameter_read, 1.e-12);    
   EXPECT_EQ(string_parameter, string_parameter_read);    
   EXPECT_EQ(complicated_string_parameter, complicated_string_parameter_read);    
   EXPECT_EQ(string_parameter, string_parameter_semi_read); //test that the ; gets trimmed
   EXPECT_EQ(bool_parameter_true, bool_parameter_true_read); 
   EXPECT_EQ(bool_parameter_false, bool_parameter_false_read); 

}
int main(int argc, char **argv) 
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

