/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2003 by Simon Trebst <trebst@itp.phys.ethz.ch>,
*                       and Matthias Troyer <troyer@comp-phys.org>
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

#include <alps/parser/xmlstream.h>
#include <alps/parser/xslt.h>
#include <boost/tuple/tuple.hpp>

#include <vector>
#include <string>
#include <iostream>

namespace alps {
namespace plot {

enum SetType {xy, xdxy, xydy, xdxydy};

//- Points --------------------------------------------------------------------------------------------------------

template<class C>
class Point {
public:
  Point() {}
  Point(C x, C y) : storage_(2) {
    storage_[0] = x;
    storage_[1] = y;
  }

  int size() const { return storage_.size(); }
  const C& operator[](int i) const { return storage_[i]; }
  void push_back(C data) { storage_.push_back(data); }
  void clear() { storage_.clear(); }
  void output(oxstream&, SetType);
private:
  std::vector<C> storage_;
};   // xmlPlot::Point

template<class C>
void Point<C>::output(oxstream& out, SetType Type) {
  out << start_tag("point") << no_linebreak << start_tag("x") << no_linebreak << storage_[0]<< end_tag("x");
  if((Type==xdxy) || (Type==xdxydy)) 
    out << start_tag("dx") << no_linebreak << storage_[2]<< end_tag("dx");
  out << start_tag("y") << no_linebreak << storage_[1]<< end_tag("y");
  if((Type==xydy) || (Type==xdxydy)) 
    out << start_tag("dy") << no_linebreak << storage_[3]<< end_tag("dy");
  out << end_tag("point");
}   // operator <<

//- Sets ----------------------------------------------------------------------------------------------------------

template<class C>
class Set : public std::vector<Point<C> > {
public:
  Set(SetType st=xy) : type_(st), show_legend_(true) {}

  Set<C>& operator<<(C p);
  Set<C>& operator<<(const boost::tuples::tuple<C, C>);
  Set<C>& operator<<(const boost::tuples::tuple<C,C,C>);
  Set<C>& operator<<(const boost::tuples::tuple<C,C,C,C>);
  Set<C>& operator<<(std::string label) { label_ = label;}

    
  std::string label() const { return label_; }
  bool show_legend() const { return show_legend_; }
  SetType type() const { return type_; }

  void push_back(Point<C> NewPoint) { std::vector<Point<C> >::push_back(NewPoint); }

private:
  SetType type_;
  std::string label_;
  bool show_legend_;
  Point<C> NewPoint;
};   // xmlPlot::Set

template<class C>
inline Set<C>& Set<C>::operator<<(C p) {
  if(type_==xydy && NewPoint.size()==2) 
    NewPoint.push_back(C(0));
  NewPoint.push_back(p);
  switch(type_) {
    case xy: 
      if(NewPoint.size()==2) {
        push_back(NewPoint);
        NewPoint.clear();
      }
      break;
    case xdxy: 
      if(NewPoint.size()==3) {
        push_back(NewPoint);
        NewPoint.clear();
      }
      break;
    case xydy:
    case xdxydy:
      if(NewPoint.size()==4) {
        push_back(NewPoint);
        NewPoint.clear();
      }
      break;
    default:
      boost::throw_exception(std::logic_error("Default reached in Set<C>& Set<C>::operator<<(C p)"));
  }
  return *this;   
}   // operator<<

template<class C>
inline Set<C>& Set<C>::operator<<(boost::tuples::tuple<C,C> t) {
  NewPoint.clear();
  NewPoint.push_back(boost::tuples::get<0>(t));
  NewPoint.push_back(boost::tuples::get<1>(t));
  push_back(NewPoint);
  return *this;   
}   // operator<<

template<class C>
inline Set<C>& Set<C>::operator<<(boost::tuples::tuple<C,C,C> t) {
  NewPoint.clear();
  NewPoint.push_back(boost::tuples::get<0>(t));
  NewPoint.push_back(boost::tuples::get<1>(t));
  switch(type) {
    case xdxy:
    case xdxydy:
      NewPoint.push_back(boost::tuples::get<2>(t));
      NewPoint.push_back(C(0));
      break;
    case xydy:
      NewPoint.push_back(C(0));
      NewPoint.push_back(boost::tuples::get<2>(t));
      break;        
  }
  push_back(NewPoint);
  return *this;   
}   // operator<<

template<class C>
inline Set<C>& Set<C>::operator<<(boost::tuples::tuple<C,C,C,C> t) {
  NewPoint.clear();
  NewPoint.push_back(boost::tuples::get<0>(t));
  NewPoint.push_back(boost::tuples::get<1>(t));
  NewPoint.push_back(boost::tuples::get<2>(t));
  NewPoint.push_back(boost::tuples::get<3>(t));
  push_back(NewPoint);
  return *this;   
}   // operator<<


template<class C>
inline oxstream& operator<<(oxstream& out, const Set<C> S) {
  out << start_tag("set") << attribute("label",S.label()) 
      << attribute("show_legend", S.show_legend() ? "true" : "false");
  for(int i=0; i<S.size(); ++i) 
    S[i].output(out)
  out << end_tag("set");
  return out;
}   

//- Plot ----------------------------------------------------------------------------------------------------------

template<class C>
class Plot : std::vector<Set<C> > {

public:
  Plot(std::string name="No name", bool show_legend=true) : name_(name), show_legend_(show_legend) {};
  
  Plot<C>& operator<<(const Set<C>& s ) { push_back(s); return *this;}
  Plot<C>& operator<<(std::string name) { name_=name; return *this;} 
  
  const std::string& name() const { return name_; }
  const std::string& xaxis() const { return xaxis_; }
  const std::string& yaxis() const { return yaxis_; }
  bool show_legend() const { return show_legend_; }
  
  void set_name(const std::string& name) { name_ = name; }
  void set_labels(const std::string& xaxis, const std::string& yaxis) { xaxis_ = xaxis; yaxis_ = yaxis; }
  void show_legend(const bool& show) { show_legend_ = show; }
  
  int size() const { return std::vector<Set<C> >::size(); }
  Set<C> operator[](int i) const { return std::vector<Set<C> >::operator[](i); }

private:
  std::string name_, xaxis_, yaxis_;
  bool show_legend_;
};   // xmlPlot::Plot


template<class C>
inline oxstream& operator<<(oxstream& out, Plot<C> P) {
  out << header("UTF-8") << stylesheet(xslt_path("plot2html.xsl"))
      << start_tag("plot") << alps::xml_namespace("xsi","http://www.w3.org/2001/XMLSchema-instance")
      << attribute("xsi:noNamespaceSchemaLocation","http://xml.comp-phys.org/2003/4/plot.xsd")
      << attribute("name",P.name());
  out << start_tag("legend") << attribute("show", P.show_legend() ? "true" : "false") << end_tag("legend");
  out << start_tag("xaxis") << attribute("label", P.xaxis()) << end_tag("xaxis");
  out << start_tag("yaxis") << attribute("label", P.yaxis()) << end_tag("yaxis");
  for(int i=0; i<P.size(); ++i) 
    out << P[i];
  out << end_tag("plot");
  return out;
}   // operator <<

}   // namespace plot
}   // namespace alps
