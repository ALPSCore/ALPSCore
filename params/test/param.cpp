/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/params.hpp>
#include "gtest/gtest.h"
#include <alps/utilities/temporary_filename.hpp>
#include <fstream>

void Test(void) {
// TEST(param, TextParamRead){
   // define a few vector types
   // typedef alps::params::dblvec dblvec;
   // typedef alps::params::some_vec<int> intvec;
    typedef std::vector<double> dblvec;
    typedef std::vector<int> intvec;
  
   //create a file name
   std::string pfilename(alps::temporary_filename("pfile"));

   //create a few scalar parameters
   int int_parameter=1;
   double double_parameter =2.;
   bool bool_parameter_true=true;
   bool bool_parameter_false=false;
   std::string string_parameter="Hello";
   std::string string_parameter_semic="Hello;";
   std::string complicated_string_parameter="/path/to/nowhere -parameter -run --this -program";

   //create some vector parameters
   dblvec dvec_param(3);
   dvec_param[0]=1.25; dvec_param[1]=2.25; dvec_param[2]=4.25; 

   intvec mvec_param(3);
   mvec_param[0]=10; mvec_param[1]=20; mvec_param[2]=30;
   
   {
     std::ofstream pfile(pfilename.c_str());
     pfile<<"INT = "<<int_parameter<<std::endl;
     pfile<<"DOUBLE = "<<double_parameter<<std::endl;
     pfile<<"STRING = "<<string_parameter<<std::endl;
     pfile<<"STRING_SEMI = "<<string_parameter_semic<<std::endl;
     pfile<<"BOOL_TRUE = "<<std::boolalpha<<bool_parameter_true<<std::endl;
     pfile<<"BOOL_FALSE = "<<bool_parameter_false<<std::endl;
     pfile<<"" <<std::endl; //empty line
     pfile<<"# this is a  line with some gunk in it" <<std::endl; //non-parameter line
     pfile<<"complicated_string = "<<complicated_string_parameter<<std::endl;
     pfile<<"NOSPACE_LEFT= "<<double_parameter<<std::endl;
     pfile<<"NOSPACE_RIGHT ="<<double_parameter<<std::endl;
     pfile<<"NOSPACE_BOTH="<<double_parameter<<std::endl;

     pfile << "DVEC=" << dvec_param[0] << ',' << dvec_param[1] << ',' << dvec_param[2] << std::endl;
     pfile << "MVEC=" << mvec_param[0] << ',' << mvec_param[1] << ',' << mvec_param[2] << std::endl;
   }

   // Imitate the command line args
   const int argc=2;
   const char* argv[2]={"THIS_PROGRAM",0};
   argv[1]=pfilename.c_str();

   //define the parameters
   alps::params p(argc,argv);
   p.description("This is a test program").
     define<int>("INT","Int parameter").
     define<double>("DOUBLE","Double parameter").
     define<std::string>("STRING","String parameter").
     define<std::string>("STRING_SEMI","String parameter with semicolon").
     define<bool>("BOOL_TRUE","Bool parameter which is true").
     define<bool>("BOOL_FALSE","Bool parameter which is false").
     define<std::string>("complicated_string","complicated string").
     define<double>("NOSPACE_LEFT","parameter in INI without space left of '='").
     define<double>("NOSPACE_RIGHT","parameter in INI without space right of '='").
     define<double>("NOSPACE_BOTH","parameter in INI without spaces around '='").
     define< alps::params::vector<double> >("DVEC","double vector").
     define< alps::params::vector<int> >("MVEC","int vector");
   

   //  read the parameters (parsing on first access)
   int int_parameter_read=p["INT"].as<int>();  
   double double_parameter_read=p["DOUBLE"].as<double>();  
   std::string string_parameter_read=p["STRING"].as<std::string>();
   std::string string_parameter_semi_read=p["STRING_SEMI"].as<std::string>();
   std::string complicated_string_parameter_read=p["complicated_string"].as<std::string>();
   bool bool_parameter_true_read=p["BOOL_TRUE"].as<bool>();
   bool bool_parameter_false_read=p["BOOL_FALSE"].as<bool>();
   double nospace_left=p["NOSPACE_LEFT"].as<double>();  
   double nospace_right=p["NOSPACE_RIGHT"].as<double>();  
   double nospace_both=p["NOSPACE_BOTH"].as<double>();

   dblvec dvec_param_read=p["DVEC"].as<dblvec>();
   intvec mvec_param_read=p["MVEC"].as<intvec>();

   EXPECT_EQ(int_parameter, int_parameter_read);    
   EXPECT_NEAR(double_parameter, double_parameter_read, 1.e-12);    
   EXPECT_EQ(string_parameter, string_parameter_read);    
   EXPECT_EQ(complicated_string_parameter, complicated_string_parameter_read);    
   EXPECT_EQ(string_parameter, string_parameter_semi_read); //test that the ; gets trimmed
   EXPECT_EQ(bool_parameter_true, bool_parameter_true_read); 
   EXPECT_EQ(bool_parameter_false, bool_parameter_false_read); 
   EXPECT_NEAR(double_parameter, nospace_left, 1.e-12);    
   EXPECT_NEAR(double_parameter, nospace_right, 1.e-12);    
   EXPECT_NEAR(double_parameter, nospace_both, 1.e-12);

   EXPECT_EQ(dvec_param.size(),dvec_param_read.size());
   for (int i=0; i<dvec_param.size(); ++i) {
       EXPECT_NEAR(dvec_param[i], dvec_param_read[i], 1.e-12);
   }
   EXPECT_EQ(mvec_param.size(),mvec_param_read.size());
   for (int i=0; i<mvec_param.size(); ++i) {
       EXPECT_EQ(mvec_param[i], mvec_param_read[i]);
   }
}
int main(int argc, char **argv) 
{
  Test();
  return 0;
   // ::testing::InitGoogleTest(&argc, argv);
   // return RUN_ALL_TESTS();
}

