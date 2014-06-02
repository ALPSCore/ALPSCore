/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_MODEL_LIBRARY_H
#define ALPS_MODEL_LIBRARY_H

#include <alps/config.h>

#ifndef ALPS_WITHOUT_XML

#include <alps/model/sitebasisstates.h>
#include <alps/model/hamiltonian.h>
#include <alps/model/basisstates.h>
#include <alps/parser/parser.h>
#include <alps/lattice/graph_helper.h>

#include <string>

namespace alps {

class ALPS_DECL ModelLibrary
{
public:
  typedef std::map<std::string,OperatorDescriptor<short> > OperatorDescriptorMap;
  typedef std::map<std::string,SiteOperator> SiteOperatorMap;
  typedef std::map<std::string,BondOperator> BondOperatorMap;
  typedef std::map<std::string,GlobalOperator> GlobalOperatorMap;

  ModelLibrary() {};
  ModelLibrary(std::istream& in) { read_xml(in);}
  ModelLibrary(const XMLTag& tag, std::istream& p) {read_xml(tag,p);}
  ModelLibrary(const Parameters& parms);
  
  void read_xml(std::istream& in) { read_xml(parse_tag(in),in);}
  void read_xml(const XMLTag& tag, std::istream& p);
  void write_xml(alps::oxstream&) const;
  
  bool has_basis(const std::string& name) const;
  bool has_site_basis(const std::string& name) const;
  bool has_hamiltonian(const std::string& name) const;
  bool has_site_operator(const std::string& name) const;
  bool has_bond_operator(const std::string& name) const;
  bool has_global_operator(const std::string& name) const;
  bool has_operator(const std::string& name) const 
  { return has_site_operator(name) || has_bond_operator(name) || has_global_operator(name);}

  const SiteBasisDescriptor<short>& get_site_basis(const std::string& name) const;
  const BasisDescriptor<short>& get_basis(const std::string& name) const;
  const HamiltonianDescriptor<short>& get_hamiltonian(const std::string& name) const;
  HamiltonianDescriptor<short> get_hamiltonian(const std::string& name, Parameters const& parms, bool issymbolic=false) const;
  HamiltonianDescriptor<short> get_hamiltonian(Parameters const& parms, bool issymbolic=false) const 
  { return get_hamiltonian(parms["MODEL"],parms,issymbolic);}
  template <class G>
  HamiltonianDescriptor<short> get_hamiltonian(alps::graph_helper<G> const& g, Parameters const& parms, bool issymbolic=false) const 
  { return get_hamiltonian(g,parms["MODEL"],parms,issymbolic);}
  template <class G>
  HamiltonianDescriptor<short> get_hamiltonian(alps::graph_helper<G> const& g, const std::string& name, Parameters const& parms, bool issymbolic=false) const
  {
    Parameters p(parms);
    alps::HamiltonianDescriptor<short> ham(get_hamiltonian(name));
    ham.create_terms(g);
    if (!issymbolic)
      p.copy_undefined(ham.default_parameters());
    ham.set_parameters(p);
    ham.substitute_operators(*this,issymbolic ? Parameters() : p);
    return ham;
  }

  const SiteOperatorMap& site_operators() const { return site_operators_;}
  const BondOperatorMap& bond_operators() const { return bond_operators_;}
  const GlobalOperatorMap& global_operators() const { return global_operators_;}
  
  SiteOperator get_site_operator(const std::string& name, Parameters const& p = Parameters()) const;
  BondOperator get_bond_operator(const std::string& name, Parameters const& p = Parameters()) const;
  GlobalOperator get_global_operator(const std::string& name, Parameters const& p = Parameters()) const;
  
private:
  typedef std::map<std::string,SiteBasisDescriptor<short> > SiteBasisDescriptorMap;
  typedef std::map<std::string,BasisDescriptor<short> > BasisDescriptorMap;
  typedef std::map<std::string,HamiltonianDescriptor<short> > HamiltonianDescriptorMap;

  SiteBasisDescriptorMap sitebases_;
  BasisDescriptorMap bases_;
  HamiltonianDescriptorMap hamiltonians_;
  SiteOperatorMap site_operators_;
  BondOperatorMap bond_operators_;
  GlobalOperatorMap global_operators_;
};

} // namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

inline alps::oxstream& operator<<(alps::oxstream& os, const alps::ModelLibrary& l)
{
  l.write_xml(os);
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const alps::ModelLibrary& l)
{
  alps::oxstream xml(os);
  xml << l;
  return os;
}

inline std::istream& operator>>(std::istream& is, alps::ModelLibrary& l)
{
  l.read_xml(is);
  return is;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // namespace alps
#endif

#endif

#endif // ALPS_MODEL_LIBRARY_H
