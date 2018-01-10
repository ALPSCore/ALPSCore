/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_UTILITIES_TEST_VECTOR_COMPARISON_PREDICATES_HPP
#define ALPS_UTILITIES_TEST_VECTOR_COMPARISON_PREDICATES_HPP

#include <vector>
#include <algorithm> /* for std::max() */
#include "gtest/gtest.h"

namespace alps {
    namespace testing {

        template <typename T>
        ::testing::AssertionResult is_near(T v1, T v2, double tol=1E-10) {
            if (std::fabs(v1-v2)<tol) {
                return ::testing::AssertionSuccess() << v1 << " almost equals " << v2 << " with tolerance " << tol;
            } else {
                return ::testing::AssertionFailure() << v1 << " not equal " << v2 << " with tolerance " << tol;
            }
        }

        template <typename T>
        ::testing::AssertionResult is_rel_near(T v1, T v2, double tol=1E-10) {
	    T vmax=std::max(std::fabs(v1),std::fabs(v2));
            if (v1==v2 || std::fabs(v1-v2)/vmax < tol) {
                return ::testing::AssertionSuccess() << v1 << " almost equals " << v2 << " with relative tolerance " << tol;
            } else {
                return ::testing::AssertionFailure() << v1 << " not equal " << v2 << " with relative tolerance " << tol;
            }
        }

        template <typename T>
        ::testing::AssertionResult is_near(const std::vector<T>& v1, const std::vector<T>& v2, double tol=1E-10) {
            std::size_t sz1=v1.size(), sz2=v2.size();
            if (sz2!=sz1) {
                return ::testing::AssertionFailure() << "sizes differ: left=" << sz1 << " right=" << sz2;
            }
            for (std::size_t i=0; i<sz1; ++i) {
                ::testing::AssertionResult res=is_near(v1[i], v2[i], tol);
                if (!res) {
                    res << "; content differs at #" << i;
                    return res;
                }
            }
            return ::testing::AssertionSuccess() << "vectors are (almost) equal";
        }

        template <typename T>
        ::testing::AssertionResult is_rel_near(const std::vector<T>& v1, const std::vector<T>& v2, double tol=1E-10) {
            std::size_t sz1=v1.size(), sz2=v2.size();
            if (sz2!=sz1) {
                return ::testing::AssertionFailure() << "sizes differ: left=" << sz1 << " right=" << sz2;
            }
            for (std::size_t i=0; i<sz1; ++i) {
                ::testing::AssertionResult res=is_rel_near(v1[i], v2[i], tol);
                if (!res) {
                    res << "; content differs at #" << i;
                    return res;
                }
            }
            return ::testing::AssertionSuccess() << "vectors are (almost) equal within the tolerance " << tol;
        }
    } // testing::
} // alps::

#endif /* ALPS_UTILITIES_TEST_VECTOR_COMPARISON_PREDICATES_HPP */
