/*
 * Set of auxiliary processing functions useful for implementations
 *
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <alps/alea/core.hpp>

#include <vector>
#include <Eigen/Dense>

/**
 * Get the data pointer from a vector in a portable fashion.
 */
#if __cplusplus >= 201103L
    #define DATA(vec) ((vec).data())
    #define STATIC_ASSERT(X, MSG) static_assert(X, MSG)
#else
    #define DATA(vec) ((vec).size() != 0 ? &vec[0] : NULL)
    #define STATIC_ASSERT(EXPR, MSG) \
                do { \
                    char STATIC_ASSERT_FAILED[(EXPR) ? 1 : -1]; \
                    (void) STATIC_ASSERT_FAILED[0]; \
                } while(0)
#endif /* __cplusplus >= 201103L */

// -------- INTEROPERABILITY WITH EIGEN --------

namespace alps { namespace alea {

template<class T, class U>
struct is_same { static const bool value = false; };

template<class T>
struct is_same<T, T> { static const bool value = true; };

/** Promote real types to complex analogue, leave complex types unchanged */
template <typename T>
struct make_complex { typedef std::complex<T> type; };

/** Promote real types to complex analogue, leave complex types unchanged */
template <typename T>
struct make_complex< std::complex<T> > { typedef std::complex<T> type; };

/** Extract underlying real type from complex, leave real types unchanged */
template <typename T>
struct make_real { typedef T type; };

/** Extract underlying real type from complex, leave real types unchanged */
template <typename T>
struct make_real< std::complex<T> > { typedef T type; };



template <typename T>
struct eigen
{
    typedef typename Eigen::Matrix<T, Eigen::Dynamic, 1> col;
    typedef typename Eigen::Map<col, Eigen::Unaligned> col_map;
    typedef typename Eigen::Map<const col, Eigen::Unaligned> const_col_map;

    typedef typename Eigen::Matrix<T, Eigen::Dynamic, 1> row;
    typedef typename Eigen::Map<row, Eigen::Unaligned> row_map;
    typedef typename Eigen::Map<const row, Eigen::Unaligned> const_row_map;

    typedef typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> matrix;
    typedef typename Eigen::Map<matrix, Eigen::Unaligned> matrix_map;
    typedef typename Eigen::Map<const matrix, Eigen::Unaligned> const_matrix_map;
};

}}
