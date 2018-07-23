/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/**
 * Set of auxiliary processing functions useful for implementations
 */
#pragma once

#include <alps/alea/core.hpp>

#include <vector>
#include <Eigen/Dense>


// -------- INTEROPERABILITY WITH EIGEN --------

namespace alps { namespace alea {

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

    typedef typename Eigen::Matrix<T, 1, Eigen::Dynamic> row;
    typedef typename Eigen::Map<row, Eigen::Unaligned> row_map;
    typedef typename Eigen::Map<const row, Eigen::Unaligned> const_row_map;

    typedef typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> matrix;
    typedef typename Eigen::Map<matrix, Eigen::Unaligned> matrix_map;
    typedef typename Eigen::Map<const matrix, Eigen::Unaligned> const_matrix_map;
};

/** Verbosity for printing */
enum verbosity {
    PRINT_TERSE,
    PRINT_VERBOSE,
    PRINT_DEBUG
};

/** Changes the verbosity level for printing */
std::ostream &operator<<(std::ostream &stream, verbosity verb);

}}
