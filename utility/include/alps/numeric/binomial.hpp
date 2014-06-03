/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_NUMERIC_BINOMIAL_HPP
#define ALPS_NUMERIC_BINOMIAL_HPP

namespace alps { namespace numeric {

/// \brief calculate the binomial coefficient
/// \return the binomial coefficient l over n
inline std::size_t binomial(std::size_t l, std::size_t n)
{
  double nominator=1;
  double denominator=1;
  std::size_t n2=std::max BOOST_PREVENT_MACRO_SUBSTITUTION (n,l-n);
  std::size_t n1=std::min BOOST_PREVENT_MACRO_SUBSTITUTION (n,l-n);
  for (std::size_t i=n2+1;i<=l;i++)
    nominator*=i;
  for (std::size_t i=2;i<=n1;i++)
    denominator*=i;
  return std::size_t(nominator/denominator+0.1);
}
} } // end namespace alps::numeric

#endif // ALPS_NUMERIC_BINOMIAL_HPP
