/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#ifndef ALPS_NUMERIC_MATRIX_ENTITY_HPP
#define ALPS_NUMERIC_MATRIX_ENTITY_HPP

namespace alps {
namespace numeric {

namespace tag {
    struct scalar {};
    struct vector {};
    struct matrix {};
}


template <typename T>
struct entity
{
    typedef tag::scalar type;
};

template <typename T>
struct get_entity : entity<typename boost::remove_const<T>::type>
{
};

} // end namespace numeric
} // end namespace alps

#endif // ALPS_NUMERIC_MATRIX_ENTITY_HPP
