/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2006-2008 by Synge Todo <wistaria@comp-phys.org>
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

#ifndef ALPS_ALEA_OBSERVABLESET_P_H
#define ALPS_ALEA_OBSERVABLESET_P_H

#include "observableset.h"
#include "simpleobseval.h"
#include "simpleobsdata.h"
#include "histogram.h"
#include <alps/parser/xmlhandler.h>

namespace alps {

class ALPS_DECL ObsValueXMLHandler : public XMLHandlerBase {
public:
  ObsValueXMLHandler(const std::string& basename, double& val, const std::string& attr = "");
  virtual ~ObsValueXMLHandler() {}

  virtual void start_element(const std::string& name, const XMLAttributes& attributes,
    xml::tag_type type);
  virtual void end_element(const std::string& name, xml::tag_type type);
  virtual void text(const std::string& text);

private:
  double& value_;
  std::string attr_;
  bool started_;
  std::string buffer_;
};


/// \brief XML parser for the elements for RealObsevaluator class
class ALPS_DECL RealObsevaluatorValueXMLHandler : public XMLHandlerBase {
public:
  RealObsevaluatorValueXMLHandler(std::string const& name, double& value, std::string& method,
    int& conv);
  virtual ~RealObsevaluatorValueXMLHandler() {}

  void start_element(const std::string& name, const XMLAttributes& attributes, xml::tag_type type);
  void end_element(const std::string& name, xml::tag_type type);
  void text(const std::string& text);

private:
  double& value_;
  std::string& method_;
  int& conv_;
  bool found_value_;
};


/// \brief XML parser for the RealObsevaluator class
class ALPS_DECL RealObsevaluatorXMLHandler : public CompositeXMLHandler {
public:
  RealObsevaluatorXMLHandler(RealObsevaluator& obs, std::string& index);
  virtual ~RealObsevaluatorXMLHandler() {}

protected:
  void start_top(const std::string& /* name */, const XMLAttributes& /* attributes */,
    xml::tag_type /* type */);
  void end_child(const std::string& name, xml::tag_type type);

private:
  RealObsevaluator& obs_;
  std::string& index_;
  SimpleXMLHandler<uint64_t> count_handler_;
  ObsValueXMLHandler mean_handler_;
  RealObsevaluatorValueXMLHandler error_handler_;
  ObsValueXMLHandler variance_handler_;
  ObsValueXMLHandler tau_handler_;
  DummyXMLHandler binned_handler_;
  DummyXMLHandler sign_handler_;
};


/// \brief XML parser for the RealVectorObsevaluator class
class ALPS_DECL RealVectorObsevaluatorXMLHandler : public CompositeXMLHandler {
public:
  RealVectorObsevaluatorXMLHandler(RealVectorObsevaluator& obs);
  virtual ~RealVectorObsevaluatorXMLHandler() {}

protected:
  void start_top(const std::string& /* name */, const XMLAttributes& /* attributes */,
    xml::tag_type /* type */);
  void end_child(const std::string& name, xml::tag_type type);

private:
  RealVectorObsevaluator& obs_;
  int pos_;
  RealObsevaluator robs_;
  std::string index_;
  RealObsevaluatorXMLHandler robs_handler_;
};


/// \brief XML parser for the entries for RealHistogramObservable class
class ALPS_DECL RealHistogramEntryXMLHandler : public CompositeXMLHandler {
public:
  RealHistogramEntryXMLHandler(uint64_t& count, uint64_t& value);
  virtual ~RealHistogramEntryXMLHandler() {}

private:
  SimpleXMLHandler<uint64_t> count_handler_;
  SimpleXMLHandler<uint64_t> value_handler_;
};

/// \brief XML parser for the RealHistogramObservable class
class ALPS_DECL RealHistogramObservableXMLHandler : public CompositeXMLHandler {
public:
  RealHistogramObservableXMLHandler(RealHistogramObservable& obs);
  virtual ~RealHistogramObservableXMLHandler() {}

protected:
  void start_top(const std::string& /* name */, const XMLAttributes& /* attributes */,
    xml::tag_type /* type */);
  void end_child(const std::string& name, xml::tag_type type);

private:
  RealHistogramObservable& obs_;
  uint64_t count_;
  uint64_t value_;
  RealHistogramEntryXMLHandler entry_handler_;
};


/// \brief XML parser for the ObservableSet class
class ALPS_DECL ObservableSetXMLHandler : public CompositeXMLHandler {
public:
  ObservableSetXMLHandler(ObservableSet& obs);

protected:
  void end_child(std::string const& name, xml::tag_type type);

private:
  ObservableSet& obs_;
  RealObsevaluator robs_;
  std::string dummy_index_;
  RealObsevaluatorXMLHandler rhandler_;
  RealVectorObsevaluator vobs_;
  RealVectorObsevaluatorXMLHandler vhandler_;
  RealHistogramObservable hobs_;
  RealHistogramObservableXMLHandler hhandler_;
};

} // namespace alps

#endif // ALPS_ALEA_OBSERVABLESET_P_H
