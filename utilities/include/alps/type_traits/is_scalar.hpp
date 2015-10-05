/*
 * Copyright (C) 1998-2015 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_TYPE_TRAITS_IS_SCALAR_H
#define ALPS_TYPE_TRAITS_IS_SCALAR_H

#include <boost/type_traits/is_same.hpp>
#include "alps/numeric/type_traits.hpp"

namespace alps {
    /// Metafunction-predicate: returns true_type if type T is scalar
    /** @param T: type to test
        @return boost::true_type or boost::false_type

        @details The type T is scalar iff it's "mathematical scalar"
        (as defined by metafunction alps::numeric::scalar<T>) is the same as T.
    */
        
    template <typename T>
    struct is_scalar: public boost::is_same<T,typename alps::numeric::scalar<T>::type> {};
} // end namespace alps

#endif // ALPS_TYPE_TRAITS_IS_SCALAR_H
