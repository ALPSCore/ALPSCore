/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2009 by Matthias Troyer <troyer@comp-phys.org>,
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

#ifndef ALPS_MODEL_MEASUREMENT_OPERATORS_H
#define ALPS_MODEL_MEASUREMENT_OPERATORS_H

#include <alps/hdf5.hpp>
#include <alps/hdf5/vector.hpp>

#include <alps/parameter.h>
#include <alps/xml.h>
#include <alps/config.h>
#include <alps/utility/encode.hpp>
#include <alps/utility/numeric_cast.hpp>
#include <alps/numeric/real.hpp>
#include <boost/foreach.hpp>
#include <vector>
#include <string>
#include <map> 

namespace alps{

class ALPS_DECL MeasurementOperators
{
public:
  MeasurementOperators (Parameters const& p);
  
  bool calc_averages() const 
  { 
    return !(average_expressions.empty() && local_expressions.empty()
              && correlation_expressions.empty() && structurefactor_expressions.empty() );
  }

protected:
  bool calc_labels() const 
  { 
    return !(local_expressions.empty()
              && correlation_expressions.empty() && structurefactor_expressions.empty() );
  }
  
  std::map<std::string,std::string> average_expressions;
  std::map<std::string,std::string> local_expressions;
  std::map<std::string,std::pair<std::string,std::string> > correlation_expressions;
  std::map<std::string,std::pair<std::string,std::string> > structurefactor_expressions;
};

class ALPS_DECL MeasurementLabels : public MeasurementOperators
{
public:

  template <class LatticeModel>
  MeasurementLabels(LatticeModel const&, int);

protected:
  std::vector<std::string> distlabel_;
  std::vector<std::string> momentumlabel_;
  std::vector<std::string> bondlabel_;
  std::vector<std::string> sitelabel_;
  mutable std::map<std::string,bool> bond_operator_; // mutable to allow operator[]
};


template <class ValueType>
class EigenvectorMeasurements 
 : public MeasurementLabels
{
public:
  typedef ValueType value_type;
  
  template <class LatticeModel>
  EigenvectorMeasurements(LatticeModel const&);
  
  void write_xml_one_vector(oxstream& out, const boost::filesystem::path&, std::size_t j) const;
  XMLTag handle_tag(std::istream& infile, const XMLTag& intag);
  
  virtual void save(hdf5::archive &) const;
  virtual void load(hdf5::archive &);

  std::map<std::string,std::vector<value_type> > average_values;
  std::map<std::string,std::vector<std::vector<value_type> > > local_values;
  std::map<std::string,std::vector<std::vector<value_type> > > correlation_values;
  std::map<std::string,std::vector<std::vector<value_type> > > structurefactor_values;
  
  bool empty() const { return average_values.empty() && local_values.empty() && correlation_values.empty() && structurefactor_values.empty();}
};


template<class LatticeModel>
MeasurementLabels::MeasurementLabels(LatticeModel const& lattice_model, int /* unused*/)
 : MeasurementOperators(lattice_model.get_parameters())
{
  if (calc_labels()) 
  {
    distlabel_ = lattice_model.distance_labels();
    momentumlabel_ = lattice_model.momenta_labels();
    bondlabel_ = lattice_model.bond_labels();
    sitelabel_ = lattice_model.site_labels();
    typedef std::pair<std::string,std::string> string_pair;
    BOOST_FOREACH(string_pair const& x, local_expressions)
      bond_operator_[x.first] = lattice_model.has_bond_operator(x.second);
  }
}


template <class ValueType>
template<class LatticeModel>
EigenvectorMeasurements<ValueType>::EigenvectorMeasurements(LatticeModel const& lattice_model)
 : MeasurementLabels(lattice_model,0)
{
}

template <class ValueType>
void EigenvectorMeasurements<ValueType>::save(alps::hdf5::archive& ar) const
{
  using alps::numeric::real;

  for (typename std::map<std::string,std::vector<value_type> >::const_iterator
    it=average_values.begin();it!=average_values.end();++it) 
  {
    std::string path = "results/"+ hdf5_name_encode(it->first);
    ar << make_pvp(path+"/mean/value", real(it->second));
  }    

  for (typename std::map<std::string,std::vector<std::vector<value_type> > >::const_iterator 
          it=local_values.begin();it!=local_values.end();++it) {
    std::string path = "results/"+ hdf5_name_encode(it->first);
    ar << make_pvp(path+"/mean/value", real(it->second));
    if (bond_operator_[it->first])
      ar << make_pvp(path+"/labels", bondlabel_);
    else
      ar << make_pvp(path+"/labels", sitelabel_);
  }

  for (typename std::map<std::string,std::vector<std::vector<value_type> > >::const_iterator 
        it=correlation_values.begin();it!=correlation_values.end();++it) {
    std::string path = "results/"+ hdf5_name_encode(it->first);
    ar << make_pvp(path+"/mean/value", real(it->second));
    ar << make_pvp(path+"/labels", distlabel_);
  }

  for (typename std::map<std::string,std::vector<std::vector<value_type> > >::const_iterator 
        it=structurefactor_values.begin();it!=structurefactor_values.end();++it) {
    std::string path = "results/"+ hdf5_name_encode(it->first);
    ar << make_pvp(path+"/mean/value", real(it->second));
    ar << make_pvp(path+"/labels", momentumlabel_);
  }

}

template <class ValueType>
void EigenvectorMeasurements<ValueType>::load(alps::hdf5::archive & ar)
{
  std::vector<std::string> list = ar.list_children(ar.get_context()+"/results");
  for (std::vector<std::string>::const_iterator it = list.begin(); it != list.end(); ++it) {
    std::string name = hdf5_name_decode(*it);
    std::string path = "results/"+*it;
    if (average_expressions.find(name) != average_expressions.end() || name == "Energy") {
      std::vector<double> vals;
      ar >> make_pvp(path+"/mean/value",vals);
      average_values[name] = numeric_cast<std::vector<ValueType> > (vals);
    }
    else {
      std::vector<std::vector<double> > vals;
      std::vector<std::vector<ValueType> >  converted = numeric_cast<std::vector<std::vector<ValueType> > > (vals);
      ar >> make_pvp(path+"/mean/value", vals);
      std::vector<std::string> labels;
      if (local_expressions.find(name) != local_expressions.end()) {
        if (bond_operator_[local_expressions.find(name)->first]) {
          if (bondlabel_.empty() && ar.is_data(path+"/labels"))
            ar >> make_pvp(path+"/labels", bondlabel_);
        }
        else {
          if (sitelabel_.empty() && ar.is_data(path+"/labels"))
            ar >> make_pvp(path+"/labels", sitelabel_);
        }
        local_values[name]=converted;
      }
      else if (correlation_expressions.find(name) != correlation_expressions.end()) {
        if (distlabel_.empty() && ar.is_data(path+"/labels"))
          ar >> make_pvp(path+"/labels", distlabel_);
        correlation_values[name]=converted;
      }
      else if (structurefactor_expressions.find(name) != structurefactor_expressions.end()) {
        if (momentumlabel_.empty() && ar.is_data(path+"/labels"))
          ar >> make_pvp(path+"/labels", momentumlabel_);
        structurefactor_values[name]=converted;
      }
      else
        boost::throw_exception(std::runtime_error("cannot decide whether " + name + " is local or correlation measurement "));
    }
  }
}


template <class ValueType>
void EigenvectorMeasurements<ValueType>::write_xml_one_vector(
    oxstream& out, const boost::filesystem::path&, std::size_t j) const
{
  using alps::numeric::real;

  for (typename std::map<std::string,std::vector<value_type> >::const_iterator 
    it=average_values.begin();it!=average_values.end();++it)
    if (j<it->second.size())
        out << start_tag("SCALAR_AVERAGE") <<  attribute("name",it->first) << no_linebreak
            << start_tag("MEAN") <<  no_linebreak << real(it->second[j]) << end_tag("MEAN")
            << end_tag("SCALAR_AVERAGE");

  for (typename std::map<std::string,std::vector<std::vector<value_type> > >::const_iterator 
          it=local_values.begin();it!=local_values.end();++it)
    if (j<it->second.size()) {
      out << start_tag("VECTOR_AVERAGE") <<  attribute("name",it->first);
      typename std::vector<value_type> ::const_iterator vit = it->second[j].begin();
      if (bond_operator_[it->first]) {
        for (unsigned nb=0; nb < bondlabel_.size() && vit != it->second[j].end() ; ++vit, ++nb)
          out << start_tag("SCALAR_AVERAGE")
              << attribute("indexvalue",bondlabel_[nb]) << no_linebreak
              << start_tag("MEAN") << no_linebreak <<  real(*vit) << end_tag("MEAN")
              << end_tag("SCALAR_AVERAGE");
      }
      else {
        for (unsigned ns=0; ns < sitelabel_.size() && vit != it->second[j].end() ; ++vit, ++ns)
          out << start_tag("SCALAR_AVERAGE")
              << attribute("indexvalue",sitelabel_[ns]) << no_linebreak
              << start_tag("MEAN") << no_linebreak <<  real(*vit) << end_tag("MEAN")
              << end_tag("SCALAR_AVERAGE");
      }
      out << end_tag("VECTOR_AVERAGE");
    }

  for (typename std::map<std::string,std::vector<std::vector<value_type> > >::const_iterator 
        it=correlation_values.begin();it!=correlation_values.end();++it)
    if (j<it->second.size()) {
      out << start_tag("VECTOR_AVERAGE") <<  attribute("name",it->first);
      typename std::vector<value_type> ::const_iterator vit = it->second[j].begin();
      for (unsigned d=0;d<distlabel_.size() && vit != it->second[j].end();++d,++vit)
        out << start_tag("SCALAR_AVERAGE") 
            << attribute("indexvalue",distlabel_[d]) << no_linebreak
            << start_tag("MEAN") << no_linebreak <<  real(*vit) << end_tag("MEAN")
            << end_tag("SCALAR_AVERAGE");
      out << end_tag("VECTOR_AVERAGE");
    }

  for (typename std::map<std::string,std::vector<std::vector<value_type> > >::const_iterator 
        it=structurefactor_values.begin();it!=structurefactor_values.end();++it)
    if (j<it->second.size()) {
      out << start_tag("VECTOR_AVERAGE") <<  attribute("name",it->first);
      typename std::vector<value_type> ::const_iterator vit = it->second[j].begin();
      for (unsigned d=0;d<momentumlabel_.size() && vit != it->second[j].end();++d,++vit)
        out << start_tag("SCALAR_AVERAGE") 
            << attribute("indexvalue",momentumlabel_[d]) << no_linebreak
            << start_tag("MEAN") << no_linebreak <<  real(*vit) << end_tag("MEAN")
            << end_tag("SCALAR_AVERAGE");
      out << end_tag("VECTOR_AVERAGE");
    }

}

template <class ValueType>
XMLTag EigenvectorMeasurements<ValueType>::handle_tag(std::istream& infile, const XMLTag& intag)
{
  XMLTag tag=intag;
  while (true) {
    if (tag.name=="SCALAR_AVERAGE") {
      std::string name=tag.attributes["name"];
      tag=parse_tag(infile);
      if (tag.name!="MEAN")
        boost::throw_exception(std::runtime_error("<MEAN> element expected inside <SCALAR_AVERAGE>"));
      value_type val;
      infile >> val;
      average_values[name].push_back(val);
      tag=parse_tag(infile);
      if (tag.name!="/MEAN")
        boost::throw_exception(std::runtime_error("</MEAN> element expected inside <SCALAR_AVERAGE>"));
      tag=parse_tag(infile);
      if (tag.name!="/SCALAR_AVERAGE")
        boost::throw_exception(std::runtime_error("</SCALAR_AVERAGE> expected"));
    }
    else if (tag.name=="VECTOR_AVERAGE") {
      std::string name=tag.attributes["name"];
      std::vector<value_type> vals;
      if (tag.type != XMLTag::SINGLE) {
        tag=parse_tag(infile);
        while (tag.name=="SCALAR_AVERAGE") {
          tag=parse_tag(infile);
          if (tag.name!="MEAN")
            boost::throw_exception(std::runtime_error("<MEAN> element expected inside <SCALAR_AVERAGE>"));
          value_type val;
          infile >> val;
          vals.push_back(val);
          tag=parse_tag(infile);
          if (tag.name!="/MEAN")
            boost::throw_exception(std::runtime_error("</MEAN> element expected inside <SCALAR_AVERAGE>"));
          tag=parse_tag(infile);
          if (tag.name!="/SCALAR_AVERAGE")
            boost::throw_exception(std::runtime_error("</SCALAR_AVERAGE> expected"));
          tag=parse_tag(infile);
        }
        if (tag.name!="/VECTOR_AVERAGE")
          boost::throw_exception(std::runtime_error("</VECTOR_AVERAGE> expected"));
      }
      if (local_expressions.find(name) != local_expressions.end())
        local_values[name].push_back(vals);
      else if (correlation_expressions.find(name) != correlation_expressions.end())
        correlation_values[name].push_back(vals);
      else if (structurefactor_expressions.find(name) != structurefactor_expressions.end())
        structurefactor_values[name].push_back(vals);
      else
        boost::throw_exception(std::runtime_error("cannot decide whether " + name + " is local or correlation measurement "));
    }
    else
      return tag;
    tag=parse_tag(infile);
  }
}

}

#endif
