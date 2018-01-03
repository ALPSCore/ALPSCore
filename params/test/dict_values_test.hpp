/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/*
  Classes to generate values of type T for testing the disctionary

 */

#ifndef PARAMS_TEST_DICT_VALUES_TEST_a01595088dd442f7889035afaf1cbf07
#define PARAMS_TEST_DICT_VALUES_TEST_a01595088dd442f7889035afaf1cbf07

#include <boost/integer_traits.hpp>

namespace alps {
  namespace params_ns {
    namespace testing {
        using boost::integer_traits;

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
        struct data_trait_base0 {
            typedef WRONG_T incompatible_type;
 
            static const bool has_smaller_type=has_type_helper<SMALLER_T,T>::value;
            typedef typename has_type_helper<SMALLER_T,T>::type smaller_type;
 
            static const bool has_larger_type=has_type_helper<LARGER_T,T>::value;
            typedef typename has_type_helper<LARGER_T,T>::type larger_type;
        };
 
        template <typename T, T val1, T val2, typename SMALLER_T, typename LARGER_T, typename WRONG_T>
        struct data_trait_base : public data_trait_base0<T,SMALLER_T,LARGER_T, WRONG_T> {
            static T get(bool choice) { return choice?val1:val2; }
        };
 
        /// Trait: properties of a data type for the purposes of alps::params
        template <typename> struct data_trait;
 
        template <>
        struct data_trait<bool> : public data_trait_base<bool, true, false, void, int, std::string> {
            static bool get(bool choice) { return choice; } 
        };
 
        template <>
        struct data_trait<char> : public data_trait_base<char, 'A', 'B', bool, int, std::string> {};
 
        template <>
        struct data_trait<int> : public data_trait_base<int,
                                                        integer_traits<int>::const_max-15,
                                                        integer_traits<int>::const_min+20,
                                                        bool, long, std::string> {};
 
        template <>
        struct data_trait<unsigned int> : public data_trait_base<unsigned int,
                                                                 integer_traits<unsigned int>::const_max-25,
                                                                 integer_traits<unsigned int>::const_max-30,
                                                                 bool, long, std::string> {};
 
        template <>
        struct data_trait<long> : public data_trait_base<long,
                                                         integer_traits<long>::const_max-35,
                                                         integer_traits<long>::const_min+40,
                                                         int, double, std::string> {};
 
        template <>
        struct data_trait<unsigned long> : public data_trait_base<unsigned long,
                                                                  integer_traits<unsigned long>::const_max-45,
                                                                  integer_traits<unsigned long>::const_max-50,
                                                                  int, double, std::string> {};
 
        template <>
        struct data_trait<double> : public data_trait_base0<double, int, void, std::string> {
            static double get(bool choice) { return choice?-2.75:1.25; }
        };
 
        template <>
        struct data_trait<float> : public data_trait_base0<float, int, void, std::string> {
            static float get(bool choice) { return choice?-1.75f:3.25f; }
        };
 
        template <>
        struct data_trait<std::string> : public data_trait_base0<std::string, void, void, int> {
            static std::string get(bool choice) {return choice?"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa":"bbb"; }; 
        };
 
        template <typename T>
        struct data_trait< std::vector<T> > {
            typedef std::vector<T> value_type;
            static value_type get(bool choice) { return value_type(3+choice, data_trait<T>::get(choice)); } 
 
            static const bool has_larger_type=false;
            typedef value_type larger_type;
 
            static const bool has_smaller_type=false;
            typedef value_type smaller_type;
 
            typedef int incompatible_type;
        };
        
        template <typename T>
        struct data_trait< std::pair<std::string, T> > {
            typedef std::pair<std::string, T> value_type;
            
            static value_type get(bool choice)
            {
                return value_type(data_trait<std::string>::get(choice),
                                  data_trait<T>::get(choice));
            } 
 
            static const bool has_larger_type=false;
            typedef value_type larger_type;
 
            static const bool has_smaller_type=false;
            typedef value_type smaller_type;
 
            typedef int incompatible_type;
        };
        
    } // ::testing
  } // ::params_ns
} // ::alps


#endif /* PARAMS_TEST_DICT_VALUES_TEST_a01595088dd442f7889035afaf1cbf07 */
