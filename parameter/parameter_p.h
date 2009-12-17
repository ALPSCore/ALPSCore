/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2006-2009 by Synge Todo <wistaria@comp-phys.org>
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

#ifndef ALPS_PARAMETER_PARAMTER_P_H
#define ALPS_PARAMETER_PARAMTER_P_H

#include "parameter.h"
#include <alps/xml.h>
#include <boost/classic_spirit.hpp>

namespace bs = boost::spirit;

namespace alps {

/// \brief Text-form parser for the Parameter class
struct ALPS_DECL ParameterParser : public bs::grammar<ParameterParser> {

  template<typename ScannerT>
  struct definition {

    bs::rule<ScannerT> parameter;
    bs::rule<ScannerT> key;
    bs::rule<ScannerT> value;

    definition(ParameterParser const& self) {
      parameter = key >> '=' >> value;
      key =
        ( bs::alpha_p
          >> *( bs::alnum_p | '_' | '\'' | '#' | bs::confix_p('[', *bs::print_p, ']') )
        )[bs::assign_a(self.param.key())],
      value =
        bs::lexeme_d
          [ bs::confix_p('"', (*bs::print_p)[bs::assign_a(self.param.value())], '"')
          | bs::confix_p('\'', (*bs::print_p)[bs::assign_a(self.param.value())], '\'')
          /* | bs::confix_p('[', (*bs::print_p)[bs::assign_a(self.param.value())], ']') */
          | ( *( bs::alnum_p | '#' | bs::range_p('\'', '+') | bs::range_p('-', '/')
               | bs::range_p('^', '_')
               )
              % ( bs::ch_p('$') >> bs::confix_p('{', *bs::graph_p, '}') )
            )[bs::assign_a(self.param.value())]
          ];
    }

    bs::rule<ScannerT> const& start() const {
      return parameter;
    }
  };

  ParameterParser(Parameter& p) : param(p) {}

  Parameter& param;
};

/// \brief ALPS XML handler for the Parameter class
class ALPS_DECL ParameterXMLHandler : public XMLHandlerBase {

public:
  ParameterXMLHandler(Parameter& p);

  void start_element(const std::string& name, const XMLAttributes& attributes, xml::tag_type type);
  void end_element(const std::string& name, xml::tag_type type);
  void text(const std::string& text);

private:
  Parameter& parameter_;
};

} // end namespace alps

#endif // ALPS_PARAMETER_PARAMETER_P_H
