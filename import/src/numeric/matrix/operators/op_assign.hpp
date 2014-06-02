/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#ifndef ALPS_NUMERIC_OPERATORS_OP_ASSIGN_HPP
#define ALPS_NUMERIC_OPERATORS_OP_ASSIGN_HPP

#include <boost/static_assert.hpp>

namespace alps {
namespace numeric {

template <typename T>
struct not_implemented
{
    static bool const value = false;
};

template <typename T1, typename T2, typename Category1, typename Category2>
void plus_assign(T1& t1, T2 const& t2, Category1, Category2)
{
    BOOST_STATIC_ASSERT(not_implemented<T1>::value);
}

template <typename T1, typename T2, typename Category1, typename Category2>
void minus_assign(T1& t1, T2 const& t2, Category1, Category2)
{
    BOOST_STATIC_ASSERT(not_implemented<T1>::value);
}

template <typename T1, typename T2, typename Category1, typename Category2>
void multiplies_assign(T1& t1, T2 const& t2, Category1, Category2)
{
    BOOST_STATIC_ASSERT(not_implemented<T1>::value);
}

} // end namespace numeric
} // end namespace alps

#endif // ALPS_NUMERIC_OPERATORS_OP_ASSIGN_HPP
