/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2006 by Synge Todo <wistaria@comp-phys.org>
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

#ifndef ALPS_PARAMETER_PARAMETERLIST_P_H
#define ALPS_PARAMETER_PARAMETERLIST_P_H

#include "parameterlist.h"
#include "parameters_p.h"
#include <boost/spirit/actor.hpp>
#include <boost/spirit/core.hpp>
#include <boost/spirit/utility/confix.hpp>

namespace bs = boost::spirit;

namespace alps {

// parser for alps::ParameterList

struct ParameterListParser : public bs::grammar<ParameterListParser> {

  template<typename ScannerT>
  struct definition {

    bs::rule<ScannerT> parameterlist;

    definition(ParameterListParser const& self) {
      self.stop = false;
      parameterlist =
        *bs::eol_p
        >> +( self.global_p
            | ( bs::ch_p('{') >> *bs::eol_p >> bs::ch_p('}') >> *bs::eol_p
              )[bs::push_back_a(self.plist, self.global)]
            | ( bs::ch_p('{')[bs::assign_a(self.local, self.global)] >> *bs::eol_p
                >> self.local_p >> bs::ch_p('}') >> *bs::eol_p
              )[bs::push_back_a(self.plist, self.local)]
            | ( bs::str_p("#clear") >> !bs::ch_p(";") >> *bs::eol_p )[bs::clear_a(self.global)]
            )
        >> !( bs::str_p("#stop") >> !bs::ch_p(";") >> *bs::eol_p )[bs::increment_a(self.stop)];
    }
    
    bs::rule<ScannerT> const& start() const { 
      return parameterlist;
    }
  };

  ParameterListParser(ParameterList& p) :
    plist(p), global_p(global), local_p(local), stop(false) {}

  ParameterList& plist;
  mutable Parameters global, local;
  ParametersParser global_p, local_p;
  mutable bool stop;
};

/// \brief Implementation handler of the ALPS XML parser for the ParameterList class  
class ParameterListXMLHandler : public CompositeXMLHandler
{
public:
  ParameterListXMLHandler(ParameterList& list);

protected:  
  void start_child(const std::string& name,
                   const XMLAttributes& attributes,
                   xml::tag_type type);
  void end_child(const std::string& name, xml::tag_type type);

private:
  ParameterList& list_;
  Parameter parameter_;
  Parameters default_, current_;
  ParameterXMLHandler parameter_handler_;
  ParametersXMLHandler current_handler_;
};

} // end namespace alps

#endif // ALPS_PARAMETER_PARAMETERLIST_P_H
