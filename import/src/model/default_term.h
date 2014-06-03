/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_MODEL_DEFAULTTERM_H
#define ALPS_MODEL_DEFAULTTERM_H

#include <alps/model/siteterm.h>
#include <alps/model/bondterm.h>
#include <alps/model/substitute.h>

namespace alps {

template <class TERM>
class DefaultTermDescriptor : public TERM
{
public:
  typedef TERM term_type;
  DefaultTermDescriptor() {}
  DefaultTermDescriptor(const XMLTag& tag, std::istream& in) : term_type(tag,in) {}
  // operator term_type() const { return static_cast<term_type const&>(*this);}
  term_type get(unsigned int type) const;
  Parameters parms(unsigned int type) const { return substitute(TERM::parms(),type); }
};

template <class TERM>
TERM DefaultTermDescriptor<TERM>::get(unsigned int type) const
{
  return term_type(*this,substitute(this->term(),type),Parameters(),type);
}

typedef DefaultTermDescriptor<SiteTermDescriptor> DefaultSiteTermDescriptor;
typedef DefaultTermDescriptor<BondTermDescriptor> DefaultBondTermDescriptor;



} // namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

template <class TERM>
inline alps::oxstream& operator<<(alps::oxstream& out, const alps::DefaultTermDescriptor<TERM>& q)
{
  q.write_xml(out);
  return out;
}

template <class TERM>
inline std::ostream& operator<<(std::ostream& out, const alps::DefaultTermDescriptor<TERM>& q)
{
  alps::oxstream xml(out);
  xml << q;
  return out;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // namespace alps
#endif

#endif
