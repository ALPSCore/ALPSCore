/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2012 by Andreas Hehn <hehn@phys.ethz.ch>                   *
 *                                                                                 *
 * This software is part of the ALPS libraries, published under the ALPS           *
 * Library License; you can use, redistribute it and/or modify it under            *
 * the terms of the license, either version 1 or (at your option) any later        *
 * version.                                                                        *
 *                                                                                 *
 * You should have received a copy of the ALPS Library License along with          *
 * the ALPS Libraries; see the file LICENSE.txt. If not, the license is also       *
 * available from http://alps.comp-phys.org/.                                      *
 *                                                                                 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        *
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT       *
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE       *
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,     *
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER     *
 * DEALINGS IN THE SOFTWARE.                                                       *
 *                                                                                 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef ALPS_MATRIX_INTERFACE_HPP
#define ALPS_MATRIX_INTERFACE_HPP

#include <alps/numeric/matrix/matrix_concept_check.hpp>


// BOOST_CONCEPT_ASSERT((MATRIX_TEMPLATE)); 
// This macro creates free functions that call member functions with the same
// name, e.g. swap_cols(A,i,j) -> A.swap_cols(i,j)
#define ALPS_IMPLEMENT_MATRIX_INTERFACE(MATRIX_TEMPLATE, TEMPLATE_PARAMETERS) \
    template TEMPLATE_PARAMETERS \
    typename MATRIX_TEMPLATE::size_type num_rows(MATRIX_TEMPLATE const& m) { \
        return m.num_rows(); \
    } \
    template TEMPLATE_PARAMETERS \
    typename MATRIX_TEMPLATE::size_type num_cols(MATRIX_TEMPLATE const& m) { \
        return m.num_cols(); \
    } \
    template TEMPLATE_PARAMETERS \
    void swap_rows(MATRIX_TEMPLATE & m, typename MATRIX_TEMPLATE::size_type i1, typename MATRIX_TEMPLATE::size_type i2) { \
        m.swap_rows(i1,i2); \
    } \
    template TEMPLATE_PARAMETERS \
    void swap_cols(MATRIX_TEMPLATE & m, typename MATRIX_TEMPLATE::size_type i1, typename MATRIX_TEMPLATE::size_type i2) { \
        m.swap_cols(i1,i2); \
    } \
    template TEMPLATE_PARAMETERS \
    std::pair<typename MATRIX_TEMPLATE::row_element_iterator, typename MATRIX_TEMPLATE::row_element_iterator> \
    row(MATRIX_TEMPLATE & m, typename MATRIX_TEMPLATE::size_type i) { \
        return m.row(i); \
    } \
    template TEMPLATE_PARAMETERS \
    std::pair<typename MATRIX_TEMPLATE::const_row_element_iterator, typename MATRIX_TEMPLATE::const_row_element_iterator> \
    row(MATRIX_TEMPLATE const& m, typename MATRIX_TEMPLATE::size_type i) { \
        return m.row(i); \
    } \
    template TEMPLATE_PARAMETERS \
    std::pair<typename MATRIX_TEMPLATE::col_element_iterator, typename MATRIX_TEMPLATE::col_element_iterator> \
    col(MATRIX_TEMPLATE & m, typename MATRIX_TEMPLATE::size_type j) { \
        return m.col(j); \
    } \
    template TEMPLATE_PARAMETERS \
    std::pair<typename MATRIX_TEMPLATE::const_col_element_iterator, typename MATRIX_TEMPLATE::const_col_element_iterator> \
    col(MATRIX_TEMPLATE const& m, typename MATRIX_TEMPLATE::size_type j) { \
        return m.col(j); \
    }

#define ALPS_IMPLEMENT_MATRIX_DIAGONAL_ITERATOR_INTERFACE(MATRIX_TEMPLATE, TEMPLATE_PARAMETERS) \
    template TEMPLATE_PARAMETERS \
    std::pair<typename MATRIX_TEMPLATE::diagonal_iterator, typename MATRIX_TEMPLATE::diagonal_iterator> \
    diagonal(MATRIX_TEMPLATE & m) { \
        return m.diagonal(); \
    } \
    template TEMPLATE_PARAMETERS \
    std::pair<typename MATRIX_TEMPLATE::const_diagonal_iterator, typename MATRIX_TEMPLATE::const_diagonal_iterator> \
    diagonal(MATRIX_TEMPLATE const& m)  { \
        return m.diagonal(); \
    }

#define ALPS_IMPLEMENT_MATRIX_ELEMENT_ITERATOR_INTERFACE(MATRIX_TEMPLATE, TEMPLATE_PARAMETERS) \
    template TEMPLATE_PARAMETERS \
    std::pair<typename MATRIX_TEMPLATE::element_iterator, typename MATRIX_TEMPLATE::element_iterator> \
    elements(MATRIX_TEMPLATE & m) { \
        return m.elements(); \
    } \
    template TEMPLATE_PARAMETERS \
    std::pair<typename MATRIX_TEMPLATE::const_element_iterator, typename MATRIX_TEMPLATE::const_element_iterator> \
    elements(MATRIX_TEMPLATE const& m)  { \
        return m.elements(); \
    }

#endif //ALPS_MATRIX_INTERFACE_HPP
