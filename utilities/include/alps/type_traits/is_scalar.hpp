/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_TYPE_TRAITS_IS_SCALAR_H
#define ALPS_TYPE_TRAITS_IS_SCALAR_H

#include <type_traits>

#include "alps/numeric/scalar.hpp"

namespace alps {
    /// Metafunction-predicate: returns true_type if type T is scalar
    /** @param T: type to test
        @return std::true_type or std::false_type

        @details The type T is scalar iff it's "mathematical scalar"
        (as defined by metafunction alps::numeric::scalar<T>) is the same as T.
    */

    template <typename T>
    struct is_scalar: public std::is_same<T,typename alps::numeric::scalar<T>::type> {};
} // end namespace alps

#endif // ALPS_TYPE_TRAITS_IS_SCALAR_H
