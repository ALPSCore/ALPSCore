/*
 * Copyright (C) 1998-2016 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/**
   Utility classes for testing.
   (FIXME: move to some publicly-available place?)
*/

#include <complex>
#include <iostream>

namespace alps {
    namespace testing {
        // Debug printing of a vector
        template <typename T>
        std::ostream& operator<<(std::ostream& strm, const std::vector<T>& vec)
        {
            typedef std::vector<T> vtype;
            typedef typename vtype::const_iterator itype;

            strm << "[";
            itype it=vec.begin();
            const itype end=vec.end();

            if (end!=it) strm << *it;
            
            for (++it; end!=it; ++it) {
                strm << ", " << *it;
            }
            strm << "]";

            return strm;
        }
        
        
        // Data generators: T=int, char, double
        template <typename T>
        class datapoint {
          public:
            /// Returns different values of type T for the different value of the argument.
            static T get(bool choice)
            {
                return T(choice?('A'+.25):('B'+.75)); // good for T=int,char,double
            }
        };

        template <>
        class datapoint<bool> {
          public:
            /// Returns different bool values for the different value of the argument.
            static bool get(bool choice) { return choice; }
        };

        template <>
        class datapoint<std::string> {
          public:
            /// Returns different string values T for the different value of the argument.
            static std::string get(bool choice) { return choice?"one":"another"; }
        };

        template <typename T>
        class datapoint< std::complex<T> > {
          public:
            /// Returns different complex values for the different value of the argument.
            static std::complex<T> get(bool choice) {
                return std::complex<T>(datapoint<T>::get(choice), datapoint<T>::get(!choice));
            }
        };

        template <typename T>
        class datapoint< std::vector<T> > {
          public:
            /// Returns different vector values for the different value of the argument.
            static std::vector<T> get(bool choice) {
                T arr[4]={ datapoint<T>::get(choice),
                           datapoint<T>::get(!choice),
                           datapoint<T>::get(choice),
                           datapoint<T>::get(!choice) };
                std::size_t sz=sizeof(arr)/sizeof(*arr);
                std::size_t len=choice? sz : (sz-1);
                return std::vector<T>(arr, arr+len);
            }
        };
        
    } // testing::
} // alps::
