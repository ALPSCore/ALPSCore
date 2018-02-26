/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
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
#include <cmath> // for pow()
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
            static T get(unsigned int choice)
            {
                if (choice==0) return 0;
                return T(choice+std::pow(2.0,-double(choice))); // good for T=int,char,double
            }

            /// Returns different values of type T for the different value of the argument.
            static T get(int choice) { return get((unsigned int)choice); }

            /// Returns different values of type T for the different value of the argument.
            /** Defined for interface uniformity with the vector variant */
            template <typename D>
            static T get(D choice, std::size_t)
            {
                return get(choice);
            }
        };


        template <>
        class datapoint<bool> {
          public:
            /// Returns different bool values for the different value of the argument.
            static bool get(bool choice) { return choice; }

            /// Returns different bool values for the different value of the argument.
            static bool get(unsigned int choice)
            {
                return (choice%2==1);
            }

            /// Returns different bool values for the different value of the argument.
            static bool get(int choice) { return get((unsigned int)choice); }

            /// Returns different bool values for the different value of the argument.
            template <typename D>
            static bool get(D choice, std::size_t) { return get(choice); }
        };

        template <>
        class datapoint<std::string> {
          public:
            /// Returns different string values for the different value of the argument.
            static std::string get(unsigned int choice)
            {
                const char zero='!';
                std::string out;
                if (choice==0) {
                    out=std::string(1,zero);
                } else {
                    const int radix=(127-zero);
                    while (choice>0) {
                        out.insert((std::size_t) 0,(std::size_t) 1,zero+char(choice%radix));
                        choice/=radix;
                    }
                }
                return out;
            }

            /// Returns different string values (of size sz) for the different value of the argument.
            static std::string get(unsigned int choice, std::size_t sz) {
                std::string out=get(choice);
                const std::size_t base_sz=out.size();
                if (sz<base_sz) {
                    throw std::invalid_argument("get<std::string>(): the requested size"
                                                " is too short for the string to be unique");
                }
                const std::size_t pad_sz=sz-base_sz;
                out.insert(0,pad_sz,'!');
                return out;
            }

            /// Returns different string values (of size sz) for the different value of the argument.
            static std::string get(int choice, std::size_t sz) { return get((unsigned int)choice, sz); }

            /// Returns different values of type T for the different value of the argument.
            static std::string get(int choice) { return get((unsigned int)choice); }

            /// Returns different string values for the different value of the argument.
            static std::string get(bool choice) { return choice?"one":"another"; }

            /// Returns different string values (of size sz) for the different value of the argument.
            static std::string get(bool choice, std::size_t sz) {
                const std::string base=get(choice);
                const std::size_t base_sz=base.size();
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

            /// Returns different values of type T for the different value of the argument.
            static std::complex<T> get(unsigned int choice)
            {
                return std::complex<T>(datapoint<T>::get(choice), datapoint<T>::get(choice+1));
            }

            /// Returns different values of type T for the different value of the argument.
            static std::complex<T> get(int choice) { return get((unsigned int)choice); }

            /// Returns different complex values for the different value of the argument.
            template <typename D>
            static std::complex<T> get(D choice, std::size_t) {
                return get(choice);
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

            /// Returns different vector values of the specified length for the different value of the argument.
            static std::vector<T> get(unsigned int choice, std::size_t sz)
            {
                std::vector<T> arr(sz);
                unsigned int i=0;
                BOOST_FOREACH(typename std::vector<T>::reference vref, arr) {
                    vref=datapoint<T>::get(choice+i,sz);
                    ++i;
                }
                return arr;
            }

            /// Returns different values of type T for the different value of the argument.
            static std::vector<T> get(int choice, std::size_t sz) { return get((unsigned int)choice, sz); }

            /// Returns different vector values (length, content) for the different value of the argument.
            static std::vector<T> get(bool choice) {
                return get(choice, choice?3:4);
            }
        
            /// Returns different vector values (length, content) for the different value of the argument.
            static std::vector<T> get(unsigned int choice) {
                return get(choice, 5+choice%10);
            }

            /// Returns different vector values (length, content) for the different value of the argument.
            static std::vector<T> get(int choice) { return get((unsigned int)choice); }
        };

        
    } // testing::
} // alps::

#endif /* ALPS_UTILITIES_TEST_TEST_UTILS_HPP_bc68762a102b40968489f16ee603e253 */
