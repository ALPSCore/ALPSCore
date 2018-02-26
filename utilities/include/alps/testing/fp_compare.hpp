/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/**
   @file fp_compare.hpp
   Utilities for comapring floating point numbers in tests
*/

// FIXME! Not tested.

#ifndef ALPS_TESTING_FP_COMPARE_HPP_161c4f3bdd53475d8c6ea28232c9653c
#define ALPS_TESTING_FP_COMPARE_HPP_161c4f3bdd53475d8c6ea28232c9653c

#include <cmath>
#include <float.h>

namespace alps {
    namespace testing {

        namespace detail {
            // FIXME! There must be somethng usable in Boost
            template <typename> struct fpconst {};

            // FIXME! TODO:C++11 use constexpr

            template <>
            struct fpconst<double> {
                static double epsilon() { return DBL_EPSILON; }
                static double min() {return DBL_MIN; }
                static double max() {return DBL_MAX; }
            };

            template <>
            struct fpconst<float> {
                static float epsilon() { return FLT_EPSILON; }
                static float min() {return FLT_MIN; }
                static float max() {return FLT_MAX; }
            };
        } // ::detail

        /// Compares two floating point values, see http://floating-point-gui.de/errors/comparison/
        template <typename T>
        inline bool is_near(const T a, const T b, const T eps)
        {
            using std::fabs;
            if (a==b) return true;

            static const T FPMIN=detail::fpconst<T>::min();
            static const T FPMAX=detail::fpconst<T>::max();
            
            const double diff=fabs(a-b);
            if (a==0 || b==0 || diff<FPMIN) return diff < (eps*FPMIN);

            const double abs_a=fabs(a);
            const double abs_b=fabs(b);
            const double sum = (abs_b > FPMAX-abs_a)? FPMAX : (abs_a+abs_b);
            return diff/sum < eps;
        }

        /// Compares two floating point values, see http://floating-point-gui.de/errors/comparison/
        template <typename T>
        inline bool is_near(const T a, const T b)
        {
            static const T EPS=detail::fpconst<T>::epsilon();
            return is_near(a, b, EPS);
        }

    } // ::testing
} // ::alps

#endif /* ALPS_TESTING_FP_COMPARE_HPP_161c4f3bdd53475d8c6ea28232c9653c */
