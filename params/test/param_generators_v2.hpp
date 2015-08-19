/*
 * Copyright (C) 1998-2015 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/*
  Classes to generate a parameter object in different ways.
  How can we generate a valid value?
  1) Copy of an object with a valid value.
  2) Load of an object with a valid value.
  3) Default construct, or construct from a command line, or file and command line, then explicitly assign.
  That is, the origin can be: default-constructed, command line constructed, file+command line constructed.
  4) Construct, then define with a default value.
  5) Construct from command line and/or file containing the value.

  Whichever way it is constructed, it should pass a series of tests. 

  Generally, a lifecycle of a parameter object could be:
  1) Construct an object (mandatory): default, from a file and/or commandline, by copying, by loading.
  2) Assign a value, Define with or without default, or do nothing.
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
        struct data_trait<long> : public data_trait_base<long, char, double, std::string> {};
 
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
 
 
        namespace value_choice {
            enum value_choice_type { CONSTRUCTION, DEFINITION, UNDEFINED };
        };
 
        /// Parameter object generator; accepts parameter type T, construction policy, definition policy, and which value to take as "correct".
        /** A value that a parameter acquires when alps::params object is constructed depends on the construction policy.
            Likewise, it may (or may not) acquire a value as a result of definition policy.
            Therefore, we need a parameter generator template argument that tells which value to take as the "correct" one.
            Contract:
            
            alps::params* ConstructPolicy<T>::new_param()
            is expected to return a pointer to new-allocated alps::params object, associating a value choice "true", if any.
            
            void DefinePolicy<T>::define_param(alps::params&)
            is expected to somehow define a parameter of type T in the alps::params object, associating a value choice "false", if any.
        */
        template <typename T,
                  template<typename> class ConstructPolicy,
                  template<typename> class DefinePolicy,
                  value_choice::value_choice_type WHICH>
        class ParamGenerator {
            typedef ConstructPolicy<T> construct_policy_type;
            typedef DefinePolicy<T> define_policy_type;
            public:
            alps::params& param;
            typedef data_trait<T> data_trait_type; 
            typedef typename data_trait_type::return_type value_type;
            value_type expected_value;
            static const value_choice::value_choice_type choice=WHICH;
 
            ParamGenerator(): param(*construct_policy_type::new_param())
            {
                define_policy_type::define_param(param);
                switch (WHICH) {
                case value_choice::CONSTRUCTION:
                    expected_value=data_trait_type::get(true);
                    break;
                case value_choice::DEFINITION:
                    expected_value=data_trait_type::get(false);
                    break;
                default:
                    /* do nothing */
                    break;
                }
            }
 
            ~ParamGenerator() { delete &param; } // horrible, but should work most of the time.
        };
 
        /// Construction policy: construct the alps::params object using default constructor
        template <typename T>
        struct DefaultConstruct {
            static alps::params* new_param() { return new alps::params(); }
        };
 
        /// Construction policy: construct the alps::params object from a commandline containing the value of the parameter
        template <typename T>
        struct PresentCmdlineConstruct {
            static alps::params* new_param() {
                std::string arg="--param=" + input_string(data_trait<T>::get(true));
                const char* argv[]={"progname", arg.c_str()};
                const int argc=sizeof(argv)/sizeof(*argv);
                return new alps::params(argc, argv);
            }
        };
 
        /// Construction policy: construct the alps::params object from a commandline with a trigger parameter
        template <>
        struct PresentCmdlineConstruct<trigger_tag> {
            static alps::params* new_param() {
                const char* argv[]={"progname", "--param"};
                const int argc=sizeof(argv)/sizeof(*argv);
                return new alps::params(argc, argv);
            }
        };
 
        /// Construction policy: construct the alps::params object from a commandline NOT containing the value of the parameter
        template <typename T>
        struct MissingCmdlineConstruct {
            static alps::params* new_param() {
                const char* argv[]={"progname", "--otherparam=0"};
                const int argc=sizeof(argv)/sizeof(*argv);
                return new alps::params(argc, argv);
            }
        };
 
        /// Definition policy: define the parameter by direct assignment
        template <typename T>
        struct AssignDefine {
            static void define_param(alps::params& p) {
                p["param"]=data_trait<T>::get(false);
            }
        };
 
        /// Definition policy: define the parameter with a default value
        template <typename T>
        struct DefaultDefine {
            static void define_param(alps::params& p) {
                p.define<T>("param", data_trait<T>::get(false), "A parameter with a default");
            }
        };
 
        /// Definition policy: define the parameter without a default value
        template <typename T>
        struct NoDefaultDefine {
            static void define_param(alps::params& p) {
                p.define<T>("param", "A parameter without a default");
            }
        };
 
        /// Definition policy: define the trigger parameter (without a default value)
        template <>
        struct NoDefaultDefine<trigger_tag> {
            static void define_param(alps::params& p) {
                p.define("param", "A trigger parameter");
            }
        };
 

        /*
          All ways to generate a parameter object.

          #!/usr/bin/bash

          declare -A Table Row States

          States=( [C]=CONSTRUCTION [D]=DEFINITION [U]=UNDEFINED )

          Table=(                                                       \
                   [DefaultConstruct]='([AssignDefine]=D [DefaultDefine]=D [NoDefaultDefine]=U )' \
            [PresentCmdlineConstruct]='([AssignDefine]=D [DefaultDefine]=C [NoDefaultDefine]=C )' \
            [MissingCmdlineConstruct]='([AssignDefine]=D [DefaultDefine]=D [NoDefaultDefine]=U )' \
                )


          declare -a x_par_set nx_par_set all_par_set
          
          for cons in DefaultConstruct PresentCmdlineConstruct MissingCmdlineConstruct; do \
          for def  in AssignDefine DefaultDefine NoDefaultDefine; do \

          eval Row=${Table[$cons]}
          state=${States[${Row[$def]}]}

          type="ParamGenerator<T, $cons, $def, value_choice::$state>"
          all_par_set[${#all_par_set[*]}]="$type"
          if [ $state = UNDEFINED ]; then
            nx_par_set[${#nx_par_set[*]}]="$type"
          else
            x_par_set[${#x_par_set[*]}]="$type"
          fi

          done; done;

          echo '#define ALPS_PARAMS_TEST_AllParamTestTypes(T) \'
          for s in "${all_par_set[@]:0:$[${#all_par_set[*]}-1]}"; do echo "$s, \\"; done
          echo ${all_par_set[-1]}

          echo '#define ALPS_PARAMS_TEST_ExParamTestTypes(T) \'
          for s in "${x_par_set[@]:0:$[${#x_par_set[*]}-1]}"; do echo "$s, \\"; done
          echo ${x_par_set[-1]}

          echo '#define ALPS_PARAMS_TEST_NxParamTestTypes(T) \'
          for s in "${nx_par_set[@]:0:$[${#nx_par_set[*]}-1]}"; do echo "$s, \\"; done
          echo ${nx_par_set[-1]}
        */

// All parameters generated in all possible ways
#define ALPS_PARAMS_TEST_AllParamTestTypes(T)                           \
        ParamGenerator<T, DefaultConstruct, AssignDefine, value_choice::DEFINITION>, \
        ParamGenerator<T, DefaultConstruct, DefaultDefine, value_choice::DEFINITION>, \
        ParamGenerator<T, DefaultConstruct, NoDefaultDefine, value_choice::UNDEFINED>, \
        ParamGenerator<T, PresentCmdlineConstruct, AssignDefine, value_choice::DEFINITION>, \
        ParamGenerator<T, PresentCmdlineConstruct, DefaultDefine, value_choice::CONSTRUCTION>, \
        ParamGenerator<T, PresentCmdlineConstruct, NoDefaultDefine, value_choice::CONSTRUCTION>, \
        ParamGenerator<T, MissingCmdlineConstruct, AssignDefine, value_choice::DEFINITION>, \
        ParamGenerator<T, MissingCmdlineConstruct, DefaultDefine, value_choice::DEFINITION>, \
        ParamGenerator<T, MissingCmdlineConstruct, NoDefaultDefine, value_choice::UNDEFINED>

#define ALPS_PARAMS_TEST_ExParamTestTypes(T)                            \
        ParamGenerator<T, DefaultConstruct, AssignDefine, value_choice::DEFINITION>, \
        ParamGenerator<T, DefaultConstruct, DefaultDefine, value_choice::DEFINITION>, \
        ParamGenerator<T, PresentCmdlineConstruct, AssignDefine, value_choice::DEFINITION>, \
        ParamGenerator<T, PresentCmdlineConstruct, DefaultDefine, value_choice::CONSTRUCTION>, \
        ParamGenerator<T, PresentCmdlineConstruct, NoDefaultDefine, value_choice::CONSTRUCTION>, \
        ParamGenerator<T, MissingCmdlineConstruct, AssignDefine, value_choice::DEFINITION>, \
        ParamGenerator<T, MissingCmdlineConstruct, DefaultDefine, value_choice::DEFINITION>

#define ALPS_PARAMS_TEST_NxParamTestTypes(T)                            \
        ParamGenerator<T, DefaultConstruct, NoDefaultDefine, value_choice::UNDEFINED>, \
        ParamGenerator<T, MissingCmdlineConstruct, NoDefaultDefine, value_choice::UNDEFINED>
        
        
    } // namespace testing
  } // namespace params_ns
} // namespace alps

#endif
