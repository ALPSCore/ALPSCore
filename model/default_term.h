/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2003-2005 by Matthias Troyer <troyer@comp-phys.org>,
*                            Synge Todo <wistaria@comp-phys.org>
*
* This software is part of the ALPS libraries, published under the ALPS
* Library License; you can use, redistribute it and/or modify it under
* the terms of the license, either version 1 or (at your option) any later
* version.
* 
* You should have received a copy of the ALPS Library License along with
* the ALPS Libraries; see the file LICENSE.txt. If not, the license is also
* available from http://alps.comp-phys.org/.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
* FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT 
* SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE 
* FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, 
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/

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
