/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2003-2009 by Simon Trebst <trebst@itp.phys.ethz.ch>,
*                            Matthias Troyer <troyer@comp-phys.org>,
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

#ifndef ALPS_PLOT_H
#define ALPS_PLOT_H

/// \file plot.h
/// \brief classes to create plots in XML format
/// 
/// This header contains classes to create plots in XML format,
/// compatible with the ALPS XML schema for plot files on the
/// http://xml.comp-phys.org/ web page

#include <alps/parser/xmlstream.h>
#include <alps/parser/xslt_path.h>
#include <alps/parameter/parameters.h>
#include <boost/tuple/tuple.hpp>

#include <vector>
#include <string>
#include <iostream>

namespace alps {

/// namespace for plots
namespace plot {

/// \brief An enum to distinguish various plot types
///
/// The types of plot are
/// - \c xy is a plot of pairs (x,y) of data points without any error bars
/// - \c xdxy is a plot of pairs (x,y) of data points with error bars on the x values
/// - \c xydy is a plot of pairs (x,y) of data points with error bars on the y values
/// - \c xdxydy is a plot of pairs (x,y) of data points with error bars on the x and y values
enum SetType {xy, xdxy, xydy, xdxydy};

/// \brief a class to store a single point in the plot
/// \param C the type to store a coordinate
/// the Point can store an arbitrary number of values, but only up to four are used normally. 
/// These will be the x and y values and the respective errors. 
template<class C>
class Point {
public:
/// the type for a coordinate
  typedef C value_type;
/// the type to store the number of coordinates
  typedef typename std::vector<C>::size_type size_type;
/// the default constructor stores no coordinate
  Point() {}
/// a constructor storing x and y coordinates of a point
  Point(C x, C y) : storage_(2) {
    storage_[0] = x;
    storage_[1] = y;
  }

  /// \brief The number of values stored
  ///
  /// Error bars in the x and y direction will be stored as additional coordinates, such that the size
  /// of a Point with error bars in both directions will be 4 and with error bars in only x or y direction
  /// will be 3.
  size_type size() const { return storage_.size(); }
  
  /// \brief returns the \a i -th coordinate.
  ///
  /// The mapping from coordinate number to meaning is specified by the values of the enum \c SetType
  /// but not stored in each point. For eaxmple a \c SetType of \c xdxy means that 
  /// - index 0 is the x-coordinate
  /// - index 1 is the error on the x-coordinate
  /// - index 2 is the y-coordinate
  const C& operator[](int i) const { return storage_[i]; }
  
  /// adds another value to the point.
  void push_back(C data) { storage_.push_back(data); }
  /// clears the contents, erasing all values
  void clear() { storage_.clear(); }
  
  /// outputs the Point in XML format, where the mapping to tags is specified by the \c SetType. E.g. a \c SetType of \c xdxy means that 
  /// - the first value will be printed as contents of an <x> tag
  /// - the second value will be printed as contents of a <dx> tag
  /// - the third value will be printed as contents of a <y> tag
  void output(oxstream& out, SetType type) const
  {
    out << start_tag("point") << no_linebreak << start_tag("x") << no_linebreak
        << storage_[0]<< end_tag("x");
    if ((type==xdxy) || (type==xdxydy)) 
      out << start_tag("dx") << no_linebreak << storage_[2] << end_tag("dx");
    out << start_tag("y") << no_linebreak << storage_[1] << end_tag("y");
    if ((type==xydy) || (type==xdxydy)) 
      out << start_tag("dy") << no_linebreak << storage_[3] << end_tag("dy");
    out << end_tag("point");
  }

private:
  std::vector<C> storage_;
};   // xmlPlot::Point

//- Sets ----------------------------------------------------------------------------------------------------------

/// \brief a dataset is a vector of points
/// \param C the type to store the coordinate of a data point in the plot set
/// stores a data set as a vector of Points
/// \sa alps::plot::Point
template<class C>
class Set : public std::vector<Point<C> > {
public:
  /// the default set type is \c xy, i.e. no error bars
  Set(SetType st=xy) : type_(st) {}

  /// \brief add another value, building points step by step
  ///
  /// depending on the plot type, 2, 3 or four values are collected to build a Point, which
  /// is then added to the set.
  Set<C>& operator<<(C p);
  /// adds a new point with two coordinates, if the plot type is XY
  Set<C>& operator<<(const boost::tuples::tuple<C, C>&);
  /// adds a new point with three coordinates, if the plot type is XDXY or XYDY
  Set<C>& operator<<(const boost::tuples::tuple<C,C,C>&);
  /// adds a new point with four coordinates, if the plot type is XDXYDY
  Set<C>& operator<<(const boost::tuples::tuple<C,C,C,C>&);
  /// set the label (legend) for the set
  Set<C>& operator<<(const std::string& label) { label_ = label; return *this;}

  /// returns the label (legend) for the set
  std::string label() const { return label_; }
  /// returns the type of set, if it is an XY, XDXY, XYDY or XDXDY plot
  SetType type() const { return type_; }

  /// adds a new point
  void push_back(const Point<C>& NewPoint)
  { std::vector<Point<C> >::push_back(NewPoint); }

private:
  SetType type_;
  std::string label_;
  Point<C> NewPoint;
};   // xmlPlot::Set



//- Plot ----------------------------------------------------------------------------------------------------------

/// \brief a class describing a plot, consisting of a number of data sets
/// \param C the data type for the coordinate of a point
/// a plot is a vector of data sets, with additional information about titles and axis labels
template<class C>
class Plot : public std::vector<Set<C> > {

public:
  /// \brief Constructor of a plot
  /// \param name the title of the plot
  /// \param show_legend indicates whether a legend should be shown
  Plot(std::string name="", alps::Parameters const& p = alps::Parameters(), bool show_legend=true)
    : name_(name), parms(p), show_legend_(show_legend) {};
  
  /// add a set to the plot
  Plot<C>& operator<<(const Set<C>& s) { std::vector<Set<C> >::push_back(s); return *this; }
  /// set the title
  Plot<C>& operator<<(const std::string& name) { name_=name; return *this; } 
  
  /// get the title
  const std::string& name() const { return name_; }
  /// get the x-axis label
  const std::string& xaxis() const { return xaxis_; }
  /// get the y-axis label
  const std::string& yaxis() const { return yaxis_; }
  /// will the legend be shown?
  bool show_legend() const { return show_legend_; }
  
  /// set the name
  void set_name(const std::string& name) { name_ = name; }
  /// set the x- and y-axis labels
  void set_labels(const std::string& xaxis, const std::string& yaxis)
  { 
    xaxis_ = xaxis; 
    yaxis_ = yaxis; 
    if (!parms.defined("observable"))
      parms["observable"] = yaxis;
  }
  /// set whether the legend should be shown
  void show_legend(bool show) { show_legend_ = show; }
  
  /// get the number of sets
  int size() const { return std::vector<Set<C> >::size(); }
  /// get the i-th set
  const Set<C>& operator[](int i) const
  { return std::vector<Set<C> >::operator[](i); }

  Parameters const& parameters() const { return parms;}
private:
  std::string name_, xaxis_, yaxis_;
  alps::Parameters parms;
  bool show_legend_;
};   // xmlPlot::Plot

/// write a plot to an XML file following the ALPS XML schema for plots on http://xml.comp-phys.org/
template<class C>
inline oxstream& operator<<(oxstream& out, const Plot<C>& p)
{
  out << header("UTF-8") << stylesheet(xslt_path("ALPS.xsl"))
      << start_tag("plot") << alps::xml_namespace("xsi","http://www.w3.org/2001/XMLSchema-instance")
      << attribute("xsi:noNamespaceSchemaLocation","http://xml.comp-phys.org/2003/4/plot.xsd")
      << attribute("name", p.name());
  out << p.parameters();
  out << start_tag("legend")
      << attribute("show", p.show_legend() ? "true" : "false")
      << end_tag("legend");
  out << start_tag("xaxis") << attribute("label", p.xaxis())
      << end_tag("xaxis");
  out << start_tag("yaxis") << attribute("label", p.yaxis())
      << end_tag("yaxis");
  for(int i=0; i < p.size(); ++i) out << p[i];
  out << end_tag("plot");
  return out;
}   // operator <<


template<class C>
Set<C>& Set<C>::operator<<(C p) {
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
inline Set<C>& Set<C>::operator<<(const boost::tuples::tuple<C,C>& t) {
  NewPoint.clear();
  NewPoint.push_back(boost::tuples::get<0>(t));
  NewPoint.push_back(boost::tuples::get<1>(t));
  std::vector<Point<C> >::push_back(NewPoint);
  return *this;   
}   // operator<<

template<class C>
inline Set<C>& Set<C>::operator<<(const boost::tuples::tuple<C,C,C>& t) {
  NewPoint.clear();
  NewPoint.push_back(boost::tuples::get<0>(t));
  NewPoint.push_back(boost::tuples::get<1>(t));
  switch(type()) {
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
  std::vector<Point<C> >::push_back(NewPoint);
  return *this;   
}   // operator<<

template<class C>
inline Set<C>& Set<C>::operator<<(const boost::tuples::tuple<C,C,C,C>& t) {
  NewPoint.clear();
  NewPoint.push_back(boost::tuples::get<0>(t));
  NewPoint.push_back(boost::tuples::get<1>(t));
  NewPoint.push_back(boost::tuples::get<2>(t));
  NewPoint.push_back(boost::tuples::get<3>(t));
  std::vector<Point<C> >::push_back(NewPoint);
  return *this;   
}   // operator<<


template<class C>
inline oxstream& operator<<(oxstream& o,  const Set<C>& S) {
  o << start_tag("set") << attribute("label",S.label());
  for(unsigned int i=0; i<S.size(); ++i) 
    S[i].output(o,S.type());
  o << end_tag("set");  
  return o;
}   

}   // namespace plot
}   // namespace alps

#endif
