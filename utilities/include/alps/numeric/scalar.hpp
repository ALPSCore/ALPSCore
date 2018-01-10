/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file scalar.hpp: Defines type traits for ALPSCore numeric types.

    This file is in `alps/numeric/` directory because it defines
    entities in `alps::numeric` namespace.
*/

#ifndef ALPS_NUMERIC_SCALAR_HPP_1444d6233fe94fe1ba881085afd0f012
#define ALPS_NUMERIC_SCALAR_HPP_1444d6233fe94fe1ba881085afd0f012

#include <complex>
#include "alps/type_traits/element_type.hpp"

namespace alps {
    namespace numeric {
        /// Metafunction returning "mathematical scalar" type for type T
        /** The mathematical scalar is defined as type that can be
            used to scale the value of type T.  Note that it is not
            necessarily any underlying storage type for the object of
            type T!
        */
        template <typename T>
        struct scalar : public alps::element_type<T> {};

        /// Metafunction returning "mathematical scalar" type for a type: <complex<T>> specialization
        template <typename T>
        struct scalar< std::complex<T> > {
            typedef std::complex<T> type;
        };
    }
}

#endif /* ALPS_NUMERIC_SCALAR_HPP_1444d6233fe94fe1ba881085afd0f012 */
