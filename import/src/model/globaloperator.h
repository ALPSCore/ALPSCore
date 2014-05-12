/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2003-2009 by Matthias Troyer <troyer@comp-phys.org>,
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

#ifndef ALPS_MODEL_GLOBALOPERATOR_H
#define ALPS_MODEL_GLOBALOPERATOR_H

#include <alps/config.h>
#include <alps/model/default_term.h>
#include <alps/lattice/graph_helper.h>
#include <vector>

namespace alps {

class ModelLibrary;

class ALPS_DECL GlobalOperator
{
public:
  GlobalOperator() {}
  GlobalOperator(const XMLTag&, std::istream&);
  XMLTag read_xml(const XMLTag&, std::istream&);
  void write_xml(oxstream&) const;

  const std::string& name() const { return name_;}
  const std::vector<SiteTermDescriptor>& site_terms() const { return siteterms_;}
  const std::vector<BondTermDescriptor>& bond_terms() const { return bondterms_;}
  SiteOperator site_term(unsigned int type=0) const;
  BondOperator bond_term(unsigned int type=0) const;
  void substitute_operators(const ModelLibrary& m, const Parameters& p);
  boost::optional<Parameters> create_site_term(unsigned int type);
  boost::optional<Parameters> create_bond_term(unsigned int type);
  
  template <class G>
  Parameters create_terms(graph_helper<G> const& l)
  {
    std::set<unsigned int> t;
    for (typename graph_helper<G>::site_iterator it=l.sites().first; it != l.sites().second;++it)
      t.insert(l.site_type(*it));
    Parameters p = create_site_terms(t);
    t.clear();
    for (typename graph_helper<G>::bond_iterator it=l.bonds().first; it != l.bonds().second;++it)
      t.insert(l.bond_type(*it));
    p << create_bond_terms(t);
    return p;
  }
  
  
protected:
  void write_operators_xml(oxstream&) const;
private:
  Parameters create_site_terms(std::set<unsigned int> const&);
  Parameters create_bond_terms(std::set<unsigned int> const&);

  std::string name_;
  std::vector<SiteTermDescriptor> siteterms_;
  std::vector<BondTermDescriptor> bondterms_;
  DefaultSiteTermDescriptor default_siteterm_;
  DefaultBondTermDescriptor default_bondterm_;
};



} // namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

inline alps::oxstream& operator<<(alps::oxstream& out, const alps::GlobalOperator& q)
{
  q.write_xml(out);
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const alps::GlobalOperator& q)
{
  alps::oxstream xml(out);
  xml << q;
  return out;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // namespace alps
#endif

#endif
