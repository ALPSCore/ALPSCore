/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/*
  Classes to generate a parameter object in different ways.

  Whichever way it is constructed, it should pass a series of tests. 

 */

#ifndef PARAMS_TEST_PARAM_GENERATORS_V2_HPP_INCLUDED
#define PARAMS_TEST_PARAM_GENERATORS_V2_HPP_INCLUDED

#include <fstream>
#include <cstdio> // for std::remove()
#include "alps/params.hpp"
#include <alps/testing/unique_file.hpp>

namespace alps {
  namespace params_ns {
    namespace testing {

        /// Tag for the special case: trigger parameter
        struct trigger_tag {};
 
 
        /// Helper class for data traits: maps non-void type T to {value=true, type=T} and void to {value=false, type=U}.
        /// General case
        template <typename T, typename U>
        struct has_type_helper {
            static const bool value = true;
            typedef T type;
        };
 
        /// Helper class for data traits: maps non-void type T to {value=true, type=T} and void to {value=false, type=U}.
        /// T:void specialization
        template <typename U>
        struct has_type_helper<void,U> {
            static const bool value = false;
            typedef U type;
        };
 
        /// Base class for data traits.
        /** T is the data type; SMALLER_T is a "smaller" data type or void if none; LARGER_T is a "larger" data type or void if none;
            WRONG_T is an incompatible type.
     
            Defines has_larger_type and has_smaller_type constants; incompatible_type, smaller_type and larger_type types.
 
            If there is no larger/smaller type, defines it as T to keep compiler happy
            (so that a valid code can still be generated, although it should never be executed,
            protected by `if (has_larger_type) {...}` at runtime).
 
            An alternative solution to avoid generating the code for the missing type altogether is too cumbersome.
        */
        template <typename T, typename SMALLER_T, typename LARGER_T, typename WRONG_T>
        struct data_trait_base {
            static T get(bool choice) { return T(choice?('A'+.25):('B'+.75)); } // good for T=int,char,double
            typedef T return_type; // good for anything except trigger_tag
 
            typedef WRONG_T incompatible_type;
 
            static const bool has_smaller_type=has_type_helper<SMALLER_T,T>::value;
            typedef typename has_type_helper<SMALLER_T,T>::type smaller_type;
 
            static const bool has_larger_type=has_type_helper<LARGER_T,T>::value;
            typedef typename has_type_helper<LARGER_T,T>::type larger_type;
        };
 
        /// Trait: properties of a data type for the purposes of alps::params
        template <typename> struct data_trait;
 
        template <>
        struct data_trait<bool> : public data_trait_base<bool, void, int, std::string> {
            static bool get(bool choice) { return choice; } 
        };
 
        template <>
        struct data_trait<int> : public data_trait_base<int, bool, long, std::string> {};
 
        template <>
        struct data_trait<unsigned int> : public data_trait_base<int, bool, long, std::string> {};
 
        template <>
        struct data_trait<long> : public data_trait_base<long, bool, double, std::string> {};
 
        template <>
        struct data_trait<unsigned long> : public data_trait_base<long, bool, double, std::string> {};
 
        template <>
        struct data_trait<double> : public data_trait_base<double, int, void, std::string> {};
 
        template <>
        struct data_trait<trigger_tag> : public data_trait_base<bool, void, void, int> {};
 
        template <>
        struct data_trait<std::string> : public data_trait_base<std::string, void, void, int> {
            static std::string get(bool choice) {return choice?"aaa":"bbb"; }; 
        };
 
        template <typename T>
        struct data_trait< std::vector<T> > {
            static T get(bool choice) { return std::vector<T>(3, data_trait<T>::get()); } 
            typedef std::vector<T> return_type; 
 
            static const bool has_larger_type=false;
            typedef return_type larger_type;
 
            static const bool has_smaller_type=false;
            typedef return_type smaller_type;
 
            typedef int incompatible_type;
        };
 
 
        /// Format a value of type T as an input string
        template <typename T>
        std::string input_string(const T& val)
        {
            return boost::lexical_cast<std::string>(val); // good for numbers and bool
        }
 
        /// Format a string as an input string
        std::string input_string(const std::string& val)
        {
            return "\""+val+"\""; 
        }
 
        /// Format a vector as an input string
        template <typename T>
        std::string input_string(const std::vector<T>& vec)
        {
            typename std::vector<T>::const_iterator it=vec.begin(), end=vec.end();
            std::string out;
            while (it!=end) {
                out += input_string(*it);
                ++it;
                if (it!=end)  out+=",";
            }
            return out;
        }
 
        /// Parameter object generator (from cmdline); accepts parameter type T.
        template <typename T>
        class CmdlineParamGenerator {
            std::string expected_origin_name_;
            public:
            alps::params* param_ptr;
            typedef data_trait<T> data_trait_type;
            typedef T value_type;

            std::string expected_origin_name() {
                return expected_origin_name_;
            }
            
            CmdlineParamGenerator(): param_ptr(0)
            {
                const T val1=data_trait_type::get(true);
                const T val2=data_trait_type::get(false);
                std::string argv_s[]={
                    "--present_def="+input_string<T>(val1),
                    "--present_nodef="+input_string<T>(val1),
                };
                expected_origin_name_="/path/to/progname";
                const char* argv[]={
                    expected_origin_name_.c_str(),
                    argv_s[0].c_str(),
                    argv_s[1].c_str()
                };
                const int argc=sizeof(argv)/sizeof(*argv);
                param_ptr=new alps::params(argc,argv);
                alps::params& param=*param_ptr;
                param.
                    template define<T>("present_def", val2, "Has default").
                    template define<T>("missing_def", val1, "Missing, has default").
                    template define<T>("present_nodef", "No default").
                    template define<T>("missing_nodef", "Missing, no default");

                param["assigned"]=val1;
            }
 
            ~CmdlineParamGenerator() { delete param_ptr; }  // FIXME: use smart_ptr instead!
        };
        
        /// Parameter object generator (from ini-file); accepts parameter type T.
        template <typename T>
        class InifileParamGenerator {
            std::string expected_origin_name_;
            public:
            alps::params* param_ptr;
            typedef data_trait<T> data_trait_type;
            typedef T value_type;

            std::string expected_origin_name() {
                return expected_origin_name_;
            }
            
            InifileParamGenerator(): param_ptr(0)
            {
                expected_origin_name_=alps::testing::temporary_filename("./inifile_input.ini.");
                std::ofstream inifile(expected_origin_name_.c_str());
                if (!inifile) throw std::runtime_error("Failed to open temporary file "+expected_origin_name_);

                const T val1=data_trait_type::get(true);
                const T val2=data_trait_type::get(false);
                inifile << "present_def=" << input_string<T>(val1) << std::endl
                        << "present_nodef=" << input_string<T>(val1) << std::endl;
                
                const char* argv[]={
                    "/path/to/progname",
                    expected_origin_name_.c_str()
                };
                const int argc=sizeof(argv)/sizeof(*argv);
                param_ptr=new alps::params(argc,argv);
                alps::params& param=*param_ptr;
                param.
                    template define<T>("present_def", val2, "Has default").
                    template define<T>("missing_def", val1, "Missing, has default").
                    template define<T>("present_nodef", "No default").
                    template define<T>("missing_nodef", "Missing, no default");

                param["assigned"]=val1;
            }
 
            ~InifileParamGenerator() {
                delete param_ptr; // FIXME: use smart_ptr instead!
                if (!expected_origin_name_.empty()) std::remove(expected_origin_name_.c_str());
            }
        };

        /// Parameter object generator (from h5-file); accepts parameter type T.
        template <typename T>
        class H5ParamGenerator {
            std::string expected_origin_name_;
            public:
            alps::params* param_ptr;
            typedef data_trait<T> data_trait_type;
            typedef T value_type;

            std::string expected_origin_name() {
                return expected_origin_name_;
            }
            
            H5ParamGenerator(): param_ptr(0)
            {
                expected_origin_name_=alps::testing::temporary_filename("./h5file_input.h5.");
                {
                    alps::hdf5::archive ar(expected_origin_name_,"w");
                    CmdlineParamGenerator<T> tmp_gen;
                    tmp_gen.param_ptr->save(ar,"/parameters");
                }
                
                const char* argv[]={
                    "/path/to/progname",
                    expected_origin_name_.c_str()
                };
                const int argc=sizeof(argv)/sizeof(*argv);

                param_ptr=new alps::params(argc,argv);
            }
 
            ~H5ParamGenerator() {
                delete param_ptr; // FIXME: use smart_ptr instead!
                if (!expected_origin_name_.empty()) std::remove(expected_origin_name_.c_str());
            }
        };

        /// Parameter object generator (from INI, then load() from h5, then `define()` after reading); accepts parameter type T.
        template <typename T>
        class InifileH5ParamGenerator {
            std::string expected_origin_name_;
            public:
            alps::params* param_ptr;
            typedef data_trait<T> data_trait_type;
            typedef T value_type;

            std::string expected_origin_name() {
                return expected_origin_name_;
            }

          
            InifileH5ParamGenerator(): param_ptr(0)
            {
                expected_origin_name_=alps::testing::temporary_filename("./h5file_input.ini.");
                
                std::ofstream inifile(expected_origin_name_.c_str());
                if (!inifile) throw std::runtime_error("Failed to open temporary file "+expected_origin_name_);

                const T val1=data_trait_type::get(true);
                const T val2=data_trait_type::get(false);
                inifile << "present_def=" << input_string<T>(val1) << std::endl
                        << "present_nodef=" << input_string<T>(val1) << std::endl;

                const std::string h5_fname=alps::testing::temporary_filename("./h5file_input.h5."); 
                {
                    const char* argv[]={
                        "/path/to/progname",
                        expected_origin_name_.c_str()
                    };
                    const int argc=sizeof(argv)/sizeof(*argv);
                    alps::params param(argc,argv);
                    alps::hdf5::archive ar(h5_fname,"w");
                    param.save(ar,"/parameters");
                    std::remove(expected_origin_name_.c_str());
                }
                
                param_ptr=new alps::params();
                alps::params& param=*param_ptr;
                {
                    alps::hdf5::archive ar(h5_fname,"r");
                    param.load(ar,"/parameters");
                }
                std::remove(h5_fname.c_str());
                
                param.
                    template define<T>("present_def", val2, "Has default").
                    template define<T>("missing_def", val1, "Missing, has default").
                    template define<T>("present_nodef", "No default").
                    template define<T>("missing_nodef", "Missing, no default");

                param["assigned"]=val1;
            }
 
            ~InifileH5ParamGenerator() {
                delete param_ptr; // FIXME: use smart_ptr instead!
                if (!expected_origin_name_.empty()) std::remove(expected_origin_name_.c_str());
            }
        };

        /// Parameter object generator (from cmdline, then load() from h5, then `define()` after reading); accepts parameter type T.
        template <typename T>
        class CmdlineH5ParamGenerator {
            std::string expected_origin_name_;
            public:
            alps::params* param_ptr;
            typedef data_trait<T> data_trait_type;
            typedef T value_type;

            std::string expected_origin_name() {
                return expected_origin_name_;
            }

          
            CmdlineH5ParamGenerator(): param_ptr(0)
            {
                const T val1=data_trait_type::get(true);
                const T val2=data_trait_type::get(false);
                const std::string h5_fname=alps::testing::temporary_filename("./h5file_input.h5."); 
                {
                    std::string argv_s[]={
                        "--present_def="+input_string<T>(val1),
                        "--present_nodef="+input_string<T>(val1),
                    };
                    expected_origin_name_="/path/to/progname";
                    const char* argv[]={
                        expected_origin_name_.c_str(),
                        argv_s[0].c_str(),
                        argv_s[1].c_str()
                    };
                    const int argc=sizeof(argv)/sizeof(*argv);
                    alps::params param(argc,argv);
                    alps::hdf5::archive ar(h5_fname,"w");
                    param.save(ar,"/parameters");
                }
                
                param_ptr=new alps::params();
                alps::params& param=*param_ptr;
                {
                    alps::hdf5::archive ar(h5_fname,"r");
                    param.load(ar,"/parameters");
                }
                std::remove(h5_fname.c_str());
                
                param.
                    template define<T>("present_def", val2, "Has default").
                    template define<T>("missing_def", val1, "Missing, has default").
                    template define<T>("present_nodef", "No default").
                    template define<T>("missing_nodef", "Missing, no default");

                param["assigned"]=val1;
            }
 
            ~CmdlineH5ParamGenerator() {
                delete param_ptr; // FIXME: use smart_ptr instead!
            }
        };


#ifdef ALPS_HAVE_MPI

        /// Parameter object generator (from cmdline by broadcast); accepts parameter type T.
        template <typename T>
        class CmdlineMpiParamGenerator {
            std::string expected_origin_name_;
            public:
            alps::params* param_ptr;
            typedef data_trait<T> data_trait_type;
            typedef T value_type;
 
            std::string expected_origin_name() {
                return expected_origin_name_;
            }
            
            CmdlineMpiParamGenerator(): param_ptr(0)
            {
                const int root=0;
                alps::mpi::communicator comm;
                const bool is_root=(comm.rank()==root);

                const T val1=data_trait_type::get(true);
                const T val2=data_trait_type::get(false);
                std::string argv_s[]={
                    "--present_def="+input_string<T>(val1),
                    "--present_nodef="+input_string<T>(val1),
                };
                // Make sure that the full command line is available
                // only on a root process:
                expected_origin_name_="/path/to/progname";
                const char* argv[]={
                    is_root? expected_origin_name_.c_str(): "",
                    is_root? argv_s[0].c_str() : "",
                    is_root? argv_s[1].c_str() : ""
                };
                const int argc=is_root? sizeof(argv)/sizeof(*argv) : 1;

                // Collective constructor:
                param_ptr=new alps::params(argc,argv,comm,root);

                alps::params& param=*param_ptr;
                param.
                    template define<T>("present_def", val2, "Has default").
                    template define<T>("missing_def", val1, "Missing, has default").
                    template define<T>("present_nodef", "No default").
                    template define<T>("missing_nodef", "Missing, no default");

                param["assigned"]=val1;
            }
 
            ~CmdlineMpiParamGenerator() { delete param_ptr; } // FIXME: use smart_ptr instead!
        };

#endif /* ALPS_HAVE_MPI*/
        
        
    } // namespace testing
  } // namespace params_ns
} // namespace alps

#endif
