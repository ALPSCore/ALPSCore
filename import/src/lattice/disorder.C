/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2001-2005 by Matthias Troyer <troyer@comp-phys.org>
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

#include <alps/lattice/disorder.h>
#include <alps/expression.h>
#include <boost/assert.hpp>

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
namespace detail {
#endif

alps::oxstream& operator<<(alps::oxstream& out,
                           const alps::detail::BasicVertexReference& d)
{
  if (d.cell_offset().size())
    out << attribute("cell", write_vector(d.cell_offset()));
  if (d.offset().size())
    out << attribute("offset", write_vector(d.offset()));
  out << attribute("vertex", d.vertex());
  return out;
}

alps::oxstream& operator<<(alps::oxstream& out,
                           const alps::detail::VertexReference& d)
{
  out << start_tag("CELL")
      << static_cast<const alps::detail::BasicVertexReference&>(d) 
      << attribute("type", d.new_type())
      << end_tag("CELL");
  return out;
}

alps::oxstream& operator<<(alps::oxstream& out,
                           const alps::detail::EdgeReference& d)
{
  out << start_tag("EDGE") << attribute("type", d.new_type())
      << start_tag("SOURCE") << no_linebreak << d.source() << end_tag("SOURCE")
      << start_tag("TARGET") << no_linebreak << d.target() << end_tag("TARGET")
      << end_tag("EDGE");
  return out;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace detail
} // end namespace alps
#endif


namespace alps {

namespace detail {

BasicVertexReference::BasicVertexReference(XMLTag tag)
{
  vertex_=(tag.attributes["vertex"]=="" ? 0 : boost::lexical_cast<int>(tag.attributes["vertex"]));
  if (tag.attributes["cell"]!="")
    read_vector_resize(tag.attributes["cell"],cell_);
  else
    boost::throw_exception(std::runtime_error("cell attribute missing in <" + tag.name + "> element"));
  if (tag.attributes["offset"]!="")
    read_vector_resize(tag.attributes["offset"],offset_);
}


VertexReference::VertexReference(XMLTag tag, std::istream& in)
{  
  if(tag.attributes["type"]=="")
    boost::throw_exception(std::runtime_error("type attribute missing in changed vertex element"));
  new_type_=boost::lexical_cast<type_type>(tag.attributes["type"]);
  if (tag.type != XMLTag::SINGLE) {
    tag=parse_tag(in);
    if(tag.name!="/VERTEX")
      boost::throw_exception(std::runtime_error("illegal contents in changed vertex element"));
  }
}

EdgeReference::EdgeReference(XMLTag tag, std::istream& in)
{
  if(tag.attributes["type"]=="")
    boost::throw_exception(std::runtime_error("type attribute missing in changed edge"));
  new_type_=boost::lexical_cast<type_type>(tag.attributes["type"]);
  tag=parse_tag(in);
  if (tag.name!="SOURCE")
    boost::throw_exception(std::runtime_error("<SOURCE> element missing in changed edge"));
  source_ = BasicVertexReference(tag);
  if (tag.type!=XMLTag::SINGLE) {
    tag=parse_tag(in);
    if(tag.name!="/SOURCE")
      boost::throw_exception(std::runtime_error("illegal contents in <SOURCE> element in changed edge"));
  }
  tag=parse_tag(in);
  if (tag.name!="TARGET")
    boost::throw_exception(std::runtime_error("<TARGET> element missing in changed edge"));
  source_ = BasicVertexReference(tag);
  if (tag.type!=XMLTag::SINGLE) {
    tag=parse_tag(in);
    if(tag.name!="/TARGET")
      boost::throw_exception(std::runtime_error("illegal contents in <TARGET> element in changed edge"));
  }
  tag=parse_tag(in);
  if(tag.name!="/EDGE")
    boost::throw_exception(std::runtime_error("illegal contents in changed edge element"));
}

} // end namespace detail


InhomogeneityDescriptor::InhomogeneityDescriptor(XMLTag& tag, std::istream& p)
 : disorder_all_vertices_(false), disorder_all_edges_(false)
{
  if (tag.name=="CHANGED") {
    tag=parse_tag(p); 
    while (tag.name!="/CHANGED") {
      if (tag.name=="VERTEX")
        changed_vertices_.push_back(detail::VertexReference(tag,p));
      else if (tag.name=="EDGE")
        changed_edges_.push_back(detail::EdgeReference(tag,p));
      else
        boost::throw_exception(std::runtime_error("Illegal element: " + tag.name + "in <CHANGED>"));
      tag=parse_tag(p); 
    }
    tag=parse_tag(p);  
  }
  if (tag.name=="INHOMOGENEOUS") {
    tag=parse_tag(p); 
    while (tag.name!="/INHOMOGENEOUS") {
      if (tag.name=="VERTEX") {
        if (tag.attributes["type"]=="")
          disorder_all_vertices_=true;
        else
          inhomogeneous_vertices_.push_back(boost::lexical_cast<type_type>(tag.attributes["type"]));
        if (tag.type !=XMLTag::SINGLE) {
          tag=parse_tag(p); 
          if (tag.name!="/VERTEX")
            boost::throw_exception(std::runtime_error("Illegal element: " + tag.name + "in <VERTEX>"));
        }
      }
      else if (tag.name=="EDGE"){
        if (tag.attributes["type"]=="")
          disorder_all_edges_=true;
        else
          inhomogeneous_edges_.push_back(boost::lexical_cast<type_type>(tag.attributes["type"]));
        if (tag.type !=XMLTag::SINGLE) {
          tag=parse_tag(p); 
          if (tag.name!="/EDGE")
            boost::throw_exception(std::runtime_error("Illegal element: " + tag.name + "in <EDGE>"));
        }
      }
      else
        boost::throw_exception(std::runtime_error("Illegal element: " + tag.name + "in <INHOMOGENEOUS>"));
      tag=parse_tag(p); 
    }
    tag=parse_tag(p);  
  }
}

void InhomogeneityDescriptor::write_xml(oxstream& xml) const
{
  if (!changed_vertices_.empty() || !changed_edges_.empty()) {
    xml << start_tag("CHANGED");
    for (unsigned int i=0;i<changed_vertices_.size();++i)
      xml << changed_vertices_[i];
    for (unsigned int i=0;i<changed_edges_.size();++i)
      xml << changed_edges_[i];
    xml << end_tag("CHANGED");
  }
  if (!inhomogeneous_vertices_.empty() || !inhomogeneous_edges_.empty() || 
       disorder_all_vertices_ || disorder_all_edges_) {
    xml << start_tag("INHOMOGENEOUS");
    if (disorder_all_vertices_)
      xml << start_tag("VERTEX") << end_tag("VERTEX");
    else
      for (unsigned int i=0;i<inhomogeneous_vertices_.size();++i)
        xml << start_tag("VERTEX") << attribute("type",inhomogeneous_vertices_[i]) << end_tag("VERTEX");
    if (disorder_all_edges_)
      xml << start_tag("EDGE") << end_tag("EDGE");
    else
      for (unsigned int i=0;i<inhomogeneous_edges_.size();++i)
        xml << start_tag("EDGE") << attribute("type",inhomogeneous_edges_[i]) << end_tag("EDGE");
    xml << end_tag("INHOMOGENEOUS");
  }
}


DepletionDescriptor::DepletionDescriptor(XMLTag& tag, std::istream& is)
{
  if (tag.name=="DEPLETION" && tag.type != XMLTag::SINGLE) {
    tag=parse_tag(is);
    if (tag.name =="VERTEX") {
      if (tag.attributes["seed"] != "")
        seed_name=tag.attributes["seed"];
      else
        seed_name="DEPLETION_SEED";
      if (tag.attributes["probability"]!="")
        prob = Expression(tag.attributes["probability"]);
      if (tag.type !=XMLTag::SINGLE) {
        tag=parse_tag(is); 
        if (tag.name!="/VERTEX")
          boost::throw_exception(std::runtime_error("Illegal element: " + tag.name + "in <DEPLETION>"));
      }
      tag=parse_tag(is); 
    }
    if (tag.name!="/DEPLETION")
      boost::throw_exception(std::runtime_error("Illegal element: " + tag.name + "in <DEPLETION>"));
  }
  tag=parse_tag(is); 
}

void DepletionDescriptor::write_xml(oxstream& xml) const
{
  if (prob)
    xml << start_tag("DEPLETION") 
        << start_tag("VERTEX") 
          << attribute("probability",prob.get()) 
          << attribute("seed",seed_name) 
        << end_tag("VERTEX") << end_tag("DEPLETION");
}

void DepletionDescriptor::set_parameters(const Parameters& p)
{
  if (p.defined(seed_name))
    seed_ = static_cast<int>(evaluate(seed_name,p));
  if (prob)
    prob.get().partial_evaluate(ParameterEvaluator(p));
}



Depletion::Depletion(DepletionDescriptor const& depl, std::size_t num_sites)
 : DepletionDescriptor(depl), num(num_sites)
{
  double p = probability();
  BOOST_ASSERT(p>=0. && p<1.);
  if (p!=0.) {
    boost::mt11213b eng(depl.seed());
    boost::variate_generator<boost::mt11213b&,boost::uniform_real<double> > 
      rng(eng,boost::uniform_real<double>(0.,1.));
    num=0;
    for (std::size_t i=0;i<num_sites;++i) {
      if (rng()<p)
        mapping.push_back(boost::optional<std::size_t>());
      else
        mapping.push_back(boost::optional<std::size_t>(num++));
    }
  }
  else
    for (std::size_t i=0;i<num_sites;++i)
      mapping.push_back(boost::optional<std::size_t>(i));
}

void Depletion::check(unsigned int site) const 
{
  if (site>=mapping.size())
    std::cerr << "Site out of range : " << site << " " << mapping.size() << "\n";
  unsigned int n=0;
  for (unsigned int i=0;i<mapping.size();++i)
    if (mapping[i])
      n++;
  if (n != num)
    std::cerr << "Count mismatch " << n << " " << num << "\n";
}




} // end namespace alps
