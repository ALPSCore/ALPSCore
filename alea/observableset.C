/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2006 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

#include "observableset.h"
#include "signedobservable.h"
#include "nobinning.h"
#include "detailedbinning.h"
#include "simpleobseval.h"
#include "histogrameval.h"
#include "observableset_p.h"

#include <alps/multi_array.hpp>
#include <boost/lexical_cast.hpp>

namespace alps {

namespace detail {

inline void deleteit(Observable& obs)
{
  delete &obs;
}

} // end namespace detail



ObservableFactory::ObservableFactory()
{
  register_observable<IntObsevaluator>();
  register_observable<RealObsevaluator>();
  register_observable<IntObservable>();
  register_observable<RealObservable>();
  register_observable<AbstractSignedObservable<RealObsevaluator> >();
  //register_observable<AbstractSignedObservable<IntObsevaluator> >();
  //register_observable<SignedObservable<SimpleIntObservable> >();
  //register_observable<SignedObservable<IntObservable> >();
  //register_observable<SignedObservable<IntTimeSeriesObservable> >();
  register_observable<SignedObservable<RealObservable> >();
  register_observable<SignedObservable<SimpleRealObservable> >();
  register_observable<SignedObservable<RealTimeSeriesObservable> >();
  register_observable<IntTimeSeriesObservable>();
  register_observable<RealTimeSeriesObservable>();
  register_observable<SimpleRealObservable>();
  register_observable<SimpleIntObservable>();
#ifdef ALPS_HAVE_VALARRAY
  register_observable<RealVectorObsevaluator>();
  register_observable<RealVectorObservable>();
  register_observable<SignedObservable<RealVectorObservable> >();
  register_observable<IntVectorObservable>();
  register_observable<IntVectorObsevaluator>();
  register_observable<IntVectorTimeSeriesObservable>();
  register_observable<RealVectorTimeSeriesObservable>();
  register_observable<SimpleIntVectorObservable>();
  register_observable<SimpleRealVectorObservable>();
  register_observable<AbstractSignedObservable<RealVectorObsevaluator> >();
  //register_observable<AbstractSignedObservable<IntVectorObsevaluator> >();
  register_observable<SignedObservable<SimpleRealVectorObservable> >();
  register_observable<SignedObservable<RealVectorTimeSeriesObservable> >();
  register_observable<HistogramObservable<double> >();
  //register_observable<SignedObservable<SimpleIntVectorObservable> >();
  //register_observable<SignedObservable<IntVectorObservable> >();
  //register_observable<SignedObservable<IntVectorTimeSeriesObservable> >();
#endif
/*
  register_observable<HistogramObservable<int32_t> >();
  register_observable<HistogramObservable<int32_t,double> >();
*/
}


#ifndef ALPS_WITHOUT_OSIRIS

void ObservableSet::save(ODump& dump) const
{
  dump << uint32_t(size());
  for (base_type::const_iterator it = begin();it != end(); ++it){
    dump << it->second->version_id();
    dump << *(it->second);
  }
}

void ObservableSet::load(IDump& dump) 
{
  uint32_t n(dump);
  for (int i = 0; i < n; ++i) {
    uint32_t v(dump);
    Observable* obs = factory_.create(v);
    dump >> *obs;
    addObservable(obs);
  }
}

#endif

void ObservableSet::update_signs()
{
  signs_.clear();
  for (iterator it = begin(); it != end(); ++it)
    if(it->second->is_signed()) {
      signs_.insert(std::make_pair(it->second->sign_name(),it->second->name()));
      if (has(it->second->sign_name()))
        it->second->set_sign((*this)[it->second->sign_name()]);
    }
}

void ObservableSet::set_sign(const std::string& sign)
{
  for (iterator it = begin(); it != end(); ++it)
    if(it->second->is_signed())
      it->second->set_sign_name(sign);
  update_signs();
}


ObservableSet::ObservableSet(const ObservableSet& m)
  : std::map<std::string,Observable*>()
{
  for (const_iterator it=m.begin();it!=m.end();++it) 
    addObservable(it->second->clone());
}

ObservableSet& ObservableSet::operator=(const ObservableSet& m)
{
  do_for_all(detail::deleteit);
  erase(begin(),end());  
  for (const_iterator it = m.begin(); it != m.end(); ++it)
    addObservable(it->second->clone());
  return *this;
}

ObservableSet::~ObservableSet()
{
  do_for_all(detail::deleteit);
}

Observable& ObservableSet::operator[](const std::string& name) 
{
  base_type::iterator it = base_type::find(name);
  if(it == base_type::end()) 
    boost::throw_exception(std::out_of_range("No Observable found with the name: "+name));
  return *((*it).second);
}

const Observable& ObservableSet::operator[](const std::string& name) const 
{
  base_type::const_iterator it = base_type::find(name);
  if(it == base_type::end()) 
    boost::throw_exception(std::out_of_range("No Observable found with the name: "+name));
  return *((*it).second);
}

bool ObservableSet::has(const std::string& name) const
{
  base_type::const_iterator it = base_type::find(name);
  return it != base_type::end();
}

void ObservableSet::reset(bool why)
{
  do_for_all(boost::bind2nd(boost::mem_fun_ref(&Observable::reset),why));
}

void ObservableSet::addObservable(Observable* obs)
{
  if (obs) {
  // store pointer
  obs->added_to_set(); // can no longer change name
  if(has(obs->name()))
    removeObservable(obs->name());
  base_type::operator[](obs->name())= obs;
  
  // insert into sign list if signed and set sign if possible
  if(obs->is_signed())
  {
    signs_.insert(std::make_pair(obs->sign_name(),obs->name()));
          
    // set sign if possible
    if(has(obs->sign_name()))
      obs->set_sign((*this)[obs->sign_name()]);
  }
  
  // set where this is sign
  for (signmap::iterator it=signs_.lower_bound(obs->name());
       it != signs_.upper_bound(obs->name()); ++it)
    (*this)[it->second].set_sign(*obs);
  }
}

void ObservableSet::addObservable(const Observable& obs)
{
  addObservable(obs.clone());
}

void ObservableSet::removeObservable(const std::string& name) 
{
  base_type::iterator it=base_type::find(name);
  if(it==base_type::end()) 
    boost::throw_exception(std::out_of_range("No Observable found with the name: "+name));

  // delete where this is sign
  for (signmap::iterator is=signs_.lower_bound(name);
    is != signs_.upper_bound(name); ++is)
    (*this)[is->second].clear_sign();
  
  if(it->second->is_signed())
  {
    // remove its sign entry
    for (signmap::iterator is=signs_.lower_bound(it->second->sign_name());
      is != signs_.upper_bound(it->second->sign_name()); ++is)
      if(is->second == name)
        signs_.erase(is);
  }
  
#ifndef ALPS_NO_DELETE
  delete it->second;
#endif
  erase(it);  
}

bool ObservableSet::can_set_thermalization_all() const
{
  bool can=true;
  for (const_iterator it=begin(); it !=end(); ++it)
    can = can && it->second->can_set_thermalization();
  return can;
}

bool ObservableSet::can_set_thermalization_any() const
{
  bool can=false;
  for (const_iterator it=begin(); it !=end(); ++it)
    can =  can || it->second->can_set_thermalization();
  return can;
}

void ObservableSet::set_thermalization(uint32_t n) 
{
  for (iterator it=begin(); it !=end(); ++it)
    if(it->second->can_set_thermalization())
      it->second->set_thermalization(n);
}

uint32_t ObservableSet::get_thermalization() const
{
  uint32_t thermal=0;
  bool got_one=false;
  for (const_iterator it=begin(); it !=end(); ++it)
    if(it->second->get_thermalization())
    {
      if(got_one)
        thermal = std::min(thermal,it->second->get_thermalization());
      else
      {
        thermal = it->second->get_thermalization();
        got_one=true;
      }
    }
  return thermal;
}

uint32_t ObservableSet::number_of_runs() const
{
  uint32_t n=0;
  for (const_iterator it=begin(); it !=end(); ++it)
    n = std::max(n, it->second->number_of_runs());
  return n;
}

ObservableSet ObservableSet::get_run(uint32_t i) const
{
  ObservableSet runset;
  for (const_iterator it=begin(); it !=end(); ++it)
    if(i<it->second->number_of_runs())
      runset.addObservable(it->second->get_run(i));
  return runset;
}

ObservableSet& ObservableSet::operator<<(const ObservableSet& obs)
{
  for (const_iterator it=obs.begin(); it !=obs.end(); ++it)
    (*this) << *(it->second);
  return *this;
}

ObservableSet& ObservableSet::operator<<(const Observable& obs)
{
  if(has(obs.name()))
  {
    if(!(*this)[obs.name()].can_merge())
      addObservable((*this)[obs.name()].convert_mergeable());
    (*this)[obs.name()].merge(obs);
  }
  else
    addObservable(obs);
  return *this;
}

ObservableFactory ObservableSet::factory_;

void ObservableSet::compact()
{
  do_for_all(boost::mem_fun_ref(&Observable::compact));
}

void ObservableSet::write_xml(oxstream& oxs, const boost::filesystem::path& fn_hdf5) const
{
  oxs << start_tag("AVERAGES");
  for(const_iterator i = begin(); i != end(); ++i)
    i->second->write_xml(oxs, fn_hdf5);
  oxs << end_tag("AVERAGES");
}

void ObservableSet::read_xml(std::istream& infile, const XMLTag& intag)
{
  if (intag.type == XMLTag::SINGLE)
    return;
  XMLTag tag = parse_tag(infile);
  while (tag.name != "/" + intag.name) {
    if (tag.name == "SCALAR_AVERAGE")
      operator<<(RealObsevaluator(tag.attributes["name"],infile,tag));
    else if (tag.name == "VECTOR_AVERAGE")
      operator<<(RealVectorObsevaluator(tag.attributes["name"],infile,tag));
    else if (tag.name == "HISTOGRAM") {
     operator<< (HistogramObservableEvaluator<double>(tag.attributes["name"],infile,tag));
    }
    else
      boost::throw_exception(std::runtime_error("Cannot parse tag " + tag.name + " in <" + intag.name + ">"));
    tag = parse_tag(infile);
  }
}

void ObservableSet::clear()
{
  base_type::clear();
  signs_.clear();
}


//
// ObsValueXMLHandler
//

ObsValueXMLHandler::ObsValueXMLHandler(const std::string& basename, double& val,
   const std::string& attr) :
  XMLHandlerBase(basename), value_(val), attr_(attr), started_(false) {}
    
void ObsValueXMLHandler::start_element(const std::string& name, const XMLAttributes& attributes, 
  xml::tag_type type) {
  if (type == xml::element) {
    if (name != basename())
      boost::throw_exception(std::runtime_error(
        "ObsValueXMLHandler::start_element: unknown start tag <" + name + ">"));
    if (started_)
      boost::throw_exception(std::runtime_error(
        "ObsValueXMLHandler::start_element: encountered nested start tags <" + name + ">"));
    if (!attr_.empty()) {
      if (!attributes.defined(attr_))
        boost::throw_exception(std::runtime_error(
          "ObsValueXMLHandler::start_element: attribute \"" + attr_ + 
          "\" not defined in <" + name + "> tag"));
      value_ = text_to_double(attributes[attr_]);
    }
    started_ = true;
  }
}

void ObsValueXMLHandler::end_element(const std::string& name, xml::tag_type type) {
  if (type == xml::element) {
    if (name != "" && name != basename())
      boost::throw_exception(std::runtime_error(
        "ObsValueXMLHandler::end_element: unknown end tag </" + name + ">"));
    if (!started_)
      boost::throw_exception(std::runtime_error(
        "ObsValueXMLHandler::end_element: unbalanced end tag </" + basename() + ">"));
    if (attr_.empty()) {
      value_ = text_to_double(buffer_);
      buffer_.clear();
    }
    started_ = false;
  }
}
    
void ObsValueXMLHandler::text(const std::string& text) {
  if (attr_.empty()) buffer_ += text;
}


//
// RealObsevaluatorValueXMLHandler
//

RealObsevaluatorValueXMLHandler::RealObsevaluatorValueXMLHandler(std::string const& name,
  double& value, std::string& method, int& conv) :
  XMLHandlerBase(name), value_(value), method_(method), conv_(conv) {}

void RealObsevaluatorValueXMLHandler::start_element(std::string const& /* name */,
  XMLAttributes const& attributes, xml::tag_type /* type */) {
  method_ = attributes["method"];
  conv_ = (attributes["converged"] == "no" ? NOT_CONVERGED : 
           attributes["converged"] == "maybe" ? MAYBE_CONVERGED : CONVERGED);
  found_value_ = false;
}

void RealObsevaluatorValueXMLHandler::end_element(std::string const& /* name */,
  xml::tag_type /* type */) {
  if (!found_value_)
    boost::throw_exception(std::runtime_error("value not found"));
}

void RealObsevaluatorValueXMLHandler::text(std::string const& text) {
  value_ = text_to_double(text);
  found_value_ = true;
}


//
// RealObsevaluatorXMLHandler
//

RealObsevaluatorXMLHandler::RealObsevaluatorXMLHandler(RealObsevaluator& obs, std::string& index) :
  CompositeXMLHandler("SCALAR_AVERAGE"), obs_(obs), index_(index),
  count_handler_("COUNT", obs_.all_.count_), mean_handler_("MEAN", obs_.all_.mean_),
  error_handler_("ERROR", obs_.all_.error_, obs_.all_.eval_method_, obs_.all_.converged_errors_),
  variance_handler_("VARIANCE", obs_.all_.variance_), tau_handler_("AUTOCORR", obs.all_.tau_),
  binned_handler_("BINNED") {
  add_handler(count_handler_);
  add_handler(mean_handler_);
  add_handler(error_handler_);
  add_handler(variance_handler_);
  add_handler(tau_handler_);
  add_handler(binned_handler_);
}

void RealObsevaluatorXMLHandler::start_top(const std::string& /* name */,
  const XMLAttributes& attributes, xml::tag_type /* type */) {
  obs_.reset();
  if (attributes.defined("name"))
    obs_.rename(attributes["name"]);
  else
    obs_.rename("unknown");
  if (attributes.defined("indexvalue"))
    index_ = attributes["indexvalue"];
  else
    index_ = "";
  obs_.valid_ = true;
  obs_.automatic_naming_ = false;
}

void RealObsevaluatorXMLHandler::end_child(std::string const& name, xml::tag_type type) {
  if (type == xml::element) {
    if (name == "ERROR")
      obs_.all_.any_converged_errors_ = obs_.all_.converged_errors_;
    else if (name == "VARIANCE")
      obs_.all_.has_variance_ = true;
    else if (name == "AUTOCORR")
      obs_.all_.has_tau_ = true;
  }
}

//
// RealVectorObsevaluatorXMLHandler
//

RealVectorObsevaluatorXMLHandler::RealVectorObsevaluatorXMLHandler(RealVectorObsevaluator& obs) :
  CompositeXMLHandler("VECTOR_AVERAGE"), obs_(obs), robs_(), robs_handler_(robs_, index_) {
  add_handler(robs_handler_);
}

void RealVectorObsevaluatorXMLHandler::start_top(const std::string& /* name */,
  const XMLAttributes& attributes, xml::tag_type /* type */) {
  obs_.reset();
  obs_.rename(attributes["name"]);
  obs_.valid_ = true;
  obs_.automatic_naming_ = false;

  pos_ = 0;
  int s = boost::lexical_cast<int>(attributes["nvalues"]);
  obs_.label_.resize(s);
  obs_.all_.mean_.resize(s);
  obs_.all_.error_.resize(s);
  obs_.all_.variance_.resize(s);
  obs_.all_.tau_.resize(s);
  obs_.all_.converged_errors_.resize(s);
  obs_.all_.any_converged_errors_.resize(s);
}

void RealVectorObsevaluatorXMLHandler::end_child(std::string const& name, xml::tag_type type) {
  if (type == xml::element) {
    if (name == "SCALAR_AVERAGE") {
      obs_.label_[pos_] = index_;
      obs_.all_.count_ = robs_.all_.count_;
      obs_.all_.mean_[pos_] = robs_.all_.mean_;
      obs_.all_.error_[pos_] = robs_.all_.error_;
      obs_.all_.has_variance_ = robs_.all_.has_variance_;
      obs_.all_.variance_[pos_] = robs_.all_.variance_;
      obs_.all_.has_tau_ = robs_.all_.has_tau_;
      obs_.all_.tau_[pos_] = robs_.all_.tau_;
      obs_.all_.converged_errors_[pos_] = robs_.all_.converged_errors_;
      obs_.all_.any_converged_errors_[pos_] = robs_.all_.any_converged_errors_;
      ++pos_;
    }
  }
}

//
// ObservableSetXMLHandler
//

ObservableSetXMLHandler::ObservableSetXMLHandler(ObservableSet& obs) :
  CompositeXMLHandler("AVERAGES"), obs_(obs), robs_(), rhandler_(robs_, dummy_index_),
  vobs_(), vhandler_(vobs_) {
  add_handler(rhandler_);
  add_handler(vhandler_);
}

void ObservableSetXMLHandler::end_child(std::string const& name,
  xml::tag_type type) {
  if (type == xml::element) {
    if (name == "SCALAR_AVERAGE")
      obs_ << robs_;
    else if (name == "VECTOR_AVERAGE")
      obs_ << vobs_;
  }
}

} // namespace alps
