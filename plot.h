/***************************************************************************
* ALPS/plot library
*
* Plot.h main header for xml plot
*
* $Id$
*
* Copyright (C) 2003 by Simon Trebst <trebst@itp.phys.ethz.ch>
*
* This program is free software; you can redistribute it and/or
* modify it under the terms of the GNU General Public License
* as published by the Free Software Foundation; either version 2
* of the License, or (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
**************************************************************************/

#include <vector>
#include <string>
#include <iostream>

#include <boost/tuple/tuple.hpp>

namespace alps {
namespace plot {

enum SetType {xy, xdxy, xydy, xdxydy};

//- Points --------------------------------------------------------------------------------------------------------

template<class C>
class Point : std::vector<C> {
  public:
  Point() {}
  Point(C x, C y) {
    resize(2);
    (*this)[0] = x;
    (*this)[1] = y;
  }

  int size() const { return std::vector<C>::size(); }

  C operator[](int i) const { return std::vector<C>::operator[](i); }
  void push_back(C data) { std::vector<C>::push_back(data); }
  void clear() { std::vector<C>::clear(); }
  void output(std::ostream&, SetType);

};   // xmlPlot::Point

template<class C>
void Point<C>::output(std::ostream& out, SetType Type) {
  out << "<point>";
    out << "<x>"<<(*this)[0]<<"</x> ";
    if((Type==xdxy) || (Type==xdxydy)) out << "<dx>"<<(*this)[2]<<"</dx> ";
    out << "<y>"<<(*this)[1]<<"</y> ";
    if((Type==xydy) || (Type==xdxydy)) out << "<dy>"<<(*this)[3]<<"</dy> ";
  out << "</point>";
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
  Set<C>& operator<<(std::string label);
    
  std::string label() const { return label_; }
  bool show_legend() const { return show_legend_; }
  int size() const { return std::vector<Point<C> >::size(); }
  SetType type() const { return type_; }

  Point<C> operator[](int i) const { return std::vector<Point<C> >::operator[](i); }
  void push_back(Point<C> NewPoint) { std::vector<Point<C> >::push_back(NewPoint); }

  private:
  SetType type_;
  std::string label_;
  bool show_legend_;
  Point<C> NewPoint;
};   // xmlPlot::Set

template<class C>
inline Set<C>& Set<C>::operator<<(C p) {
  if(type_==xydy && NewPoint.size()==2) { NewPoint.push_back(C(0)); }
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
inline Set<C>& Set<C>::operator<<(std::string label) {
  label_ = label;
  return *this;   
}   // operator<<

template<class C>
inline std::ostream& operator<<(std::ostream& out, const Set<C> S) {
  out << "<set"; 
    out << " label = \"" << S.label() << "\"";
    out << " show_legend = \""; if(S.show_legend()) out << "true\">"; else out << "false\">";
    for(int i=0; i<S.size(); ++i) { S[i].output(out, S.type()); out << std::endl; }
  out << "</set>" << std::endl;
  return out;
}   // operator <<

//- Plot ----------------------------------------------------------------------------------------------------------

template<class C>
class Plot : std::vector<Set<C> > {

public:
  Plot(std::string name="No name", bool show_legend=true) : name_(name), show_legend_(show_legend) {};
  
  Plot<C>& operator<<(Set<C>);
  Plot<C>& operator<<(std::string name);
  
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
Plot<C>& Plot<C>::operator<<(Set<C> S) {
  push_back(S);
  return *this;
}   // operator<<  

template<class C>
Plot<C>& Plot<C>::operator<<(std::string name) {
  name_ = name;
  return *this;
}   // operator<<  

template<class C>
inline std::ostream& operator<<(std::ostream& out, Plot<C> P) {
  // xml header
  out << "<?xml version=\"1.0\"?>" << std::endl	
      << "<?xml-stylesheet type=\"text/xsl\" href=\"http://xml.comp-phys.org/2003/4/plot2html.xsl\"?>" << std::endl;
  // plot  
  out << "<plot name = \"" << P.name() << "\">" << std::endl;
    out << "<legend show = \""; if(P.show_legend()) out << "true\""; else out << "false\""; out << "/>" << std::endl;
    out << "<xaxis label = \"" << P.xaxis() << "\"/>" << std::endl;
    out << "<yaxis label = \"" << P.yaxis() << "\"/>" << std::endl;
    for(int i=0; i<P.size(); ++i) { out << P[i] << std::endl; }
  out << "</plot>" << std::endl;
  return out;
}   // operator <<

}   // namespace plot
}   // namespace alps