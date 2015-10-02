/*
 * Copyright (C) 1998-2015 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/*
  Classes to generate a parameter object in different ways.

  Whichever way it is constructed, it should pass a series of tests. 

 */

#ifndef PARAMS_TEST_PARAM_GENERATORS_V2_HPP_INCLUDED
#define PARAMS_TEST_PARAM_GENERATORS_V2_HPP_INCLUDED

#include "alps/params.hpp"

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
        struct data_trait<char> : public data_trait_base<char, void, int, std::string> {};
     
        template <>
        struct data_trait<bool> : public data_trait_base<bool, void, int, std::string> {
            static bool get(bool choice) { return choice; } 
        };
 
        template <>
        struct data_trait<int> : public data_trait_base<int, char, long, std::string> {};
 
        template <>
        struct data_trait<unsigned int> : public data_trait_base<int, char, long, std::string> {};
 
        template <>
        struct data_trait<long> : public data_trait_base<long, char, double, std::string> {};
 
        template <>
        struct data_trait<unsigned long> : public data_trait_base<long, char, double, std::string> {};
 
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
 
        /// Parameter object generator; accepts parameter type T.
        template <typename T>
        class CmdlineParamGenerator {
            public:
            alps::params* param_ptr;
            typedef data_trait<T> data_trait_type;
            typedef T value_type;
 
            CmdlineParamGenerator(): param_ptr(0)
            {
              const T val1=data_trait_type::get(true);
              const T val2=data_trait_type::get(false);
              std::string argv_s[]={
                "--present_def="+input_string<T>(val1),
                "--present_nodef="+input_string<T>(val1),
              };
              const char* argv[]={
                "progname",
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
                template define<T>("missing_nodef", "MIssing, no default");

              param["assigned"]=val1;
            }
 
            ~CmdlineParamGenerator() { delete param_ptr; } 
        };
 
        
        
    } // namespace testing
  } // namespace params_ns
} // namespace alps

#endif
