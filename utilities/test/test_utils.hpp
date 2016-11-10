/*
 * Copyright (C) 1998-2016 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/**
   Utility classes for testing.
   (FIXME: move to some publicly-available place?)
*/

#ifndef ALPS_UTILITIES_TEST_TEST_UTILS_HPP_bc68762a102b40968489f16ee603e253
#define ALPS_UTILITIES_TEST_TEST_UTILS_HPP_bc68762a102b40968489f16ee603e253

#include <complex>
#include <iostream>
#include <boost/foreach.hpp>

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

            if (end!=it) {
                strm << *it;
                for (++it; end!=it; ++it) {
                    strm << ", " << *it;
                }
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

            /// Returns different values of type T for the different value of the argument.
            /** Defined for interface uniformity with the vector variant */
            static T get(bool choice, std::size_t)
            {
                return T(choice?('A'+.25):('B'+.75)); // good for T=int,char,double
            }
        };

        template <>
        class datapoint<bool> {
          public:
            /// Returns different bool values for the different value of the argument.
            static bool get(bool choice) { return choice; }

            /// Returns different bool values for the different value of the argument.
            static bool get(bool choice, std::size_t) { return choice; }
        };

        template <>
        class datapoint<std::string> {
          public:
            /// Returns different string values for the different value of the argument.
            static std::string get(bool choice) { return choice?"one":"another"; }

            /// Returns different string values (of size sz) for the different value of the argument.
            static std::string get(bool choice, std::size_t sz) {
                std::string base=get(choice);
                std::size_t base_sz=base.size();
                std::size_t nrep=(sz+base_sz-1)/base_sz;
                std::string ret;
                while (nrep--) ret.append(base);
                ret.resize(sz);
                return ret;
            }
        };

        template <typename T>
        class datapoint< std::complex<T> > {
          public:
            /// Returns different complex values for the different value of the argument.
            static std::complex<T> get(bool choice) {
                return std::complex<T>(datapoint<T>::get(choice), datapoint<T>::get(!choice));
            }

            /// Returns different complex values for the different value of the argument.
            static std::complex<T> get(bool choice, std::size_t) {
                return std::complex<T>(datapoint<T>::get(choice), datapoint<T>::get(!choice));
            }
        };

        template <typename T>
        class datapoint< std::vector<T> > {
          public:
            /// Returns different vector values of the specified length for the different value of the argument.
            static std::vector<T> get(bool choice, std::size_t sz) {
                std::vector<T> arr(sz);
                BOOST_FOREACH(typename std::vector<T>::reference vref, arr) {
                    vref=datapoint<T>::get(choice,sz);
                    choice=!choice;
                }
                return arr;
            }

            /// Returns different vector values (length, content) for the different value of the argument.
            static std::vector<T> get(bool choice) {
                return get(choice, choice?3:4);
            }
        
        };
        
    } // testing::
} // alps::

#endif /* ALPS_UTILITIES_TEST_TEST_UTILS_HPP_bc68762a102b40968489f16ee603e253 */
