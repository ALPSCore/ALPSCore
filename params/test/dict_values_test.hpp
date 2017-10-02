/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/*
  Classes to generate values of type T for testing the disctionary

 */

#ifndef PARAMS_TEST_DICT_VALUES_TEST_a01595088dd442f7889035afaf1cbf07
#define PARAMS_TEST_DICT_VALUES_TEST_a01595088dd442f7889035afaf1cbf07

namespace alps {
  namespace params_new_ns {
    namespace testing {

        /// Helper class for data traits: maps non-void type T to {value=true, type=T} and void to {value=false, type=U}. (General case)
        template <typename T, typename U>
        struct has_type_helper {
            static const bool value = true;
            typedef T type;
        };
 
        /// Helper class for data traits: maps non-void type T to {value=true, type=T} and void to {value=false, type=U}. (T:void specialization)
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
        */
        template <typename T, typename SMALLER_T, typename LARGER_T, typename WRONG_T>
        struct data_trait_base {
            static T get(bool choice) { return T(choice?('A'+.25):('B'+.75)); } // good for T=int,char,double
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
        struct data_trait<long> : public data_trait_base<long, bool, double, std::string> {};
 
        template <>
        struct data_trait<unsigned long> : public data_trait_base<long, bool, double, std::string> {};
 
        template <>
        struct data_trait<double> : public data_trait_base<double, int, void, std::string> {};
 
        template <>
        struct data_trait<std::string> : public data_trait_base<std::string, void, void, int> {
            static std::string get(bool choice) {return choice?"aaa":"bbb"; }; 
        };
 
        template <typename T>
        struct data_trait< std::vector<T> > {
            static T get(bool choice) { return std::vector<T>(3+choice, data_trait<T>::get(choice)); } 
 
            static const bool has_larger_type=false;
            typedef T larger_type;
 
            static const bool has_smaller_type=false;
            typedef T smaller_type;
 
            typedef int incompatible_type;
        };
        
        template <typename T>
        struct data_trait< std::pair<std::string, T> > {
            static T get(bool choice)
            {
                return std::pair<std::string, T>(data_trait<std::string>::get(choice),
                                                 data_trait<T>::get(choice));
            } 
 
            static const bool has_larger_type=false;
            typedef T larger_type;
 
            static const bool has_smaller_type=false;
            typedef T smaller_type;
 
            typedef int incompatible_type;
        };
        
    } // namespace testing
  } // namespace params_new_ns
} // namespace alps


#endif /* PARAMS_TEST_DICT_VALUES_TEST_a01595088dd442f7889035afaf1cbf07 */
