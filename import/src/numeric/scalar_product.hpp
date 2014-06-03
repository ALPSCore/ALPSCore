/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_NUMERIC_SCALAR_PRODUCT_HPP
#define ALPS_NUMERIC_SCALAR_PRODUCT_HPP

#include <alps/numeric/matrix/scalar_product.hpp>
#include <alps/functional.h>
#include <alps/type_traits/element_type.hpp>

#include <algorithm>
#include <functional>
#include <numeric>
#include <valarray>

namespace alps { namespace numeric {

// The generic implementation of the scalar_product moved to alps/numeric/matrix/scalar_product.hpp

/// \overload
template <class T>
inline T scalar_product(const std::valarray<T>& c1, const std::valarray<T>& c2) 
{
  return std::inner_product(data(c1),data(c1)+c1.size(),data(c2),T(), std::plus<T>(),conj_mult<T,T>());
}

} } // namespace alps::numeric

#endif // ALPS_NUMERIC_SCALAR_PRODUCT_HPP
