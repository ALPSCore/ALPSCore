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

#include <alps/scheduler/measurement_operators.h>

// some file (probably a python header) defines a tolower macro ...
#undef tolower
#undef toupper

#include <boost/regex.hpp> 

alps::MeasurementOperators::MeasurementOperators (Parameters const& parms)
{
  boost::regex expression("^MEASURE_AVERAGE\\[(.*)]$");
  boost::smatch what;
  for (alps::Parameters::const_iterator it=parms.begin();it != parms.end();++it) {
    std::string lhs = it->key();
    if (boost::regex_match(lhs, what, expression))
      average_expressions[what.str(1)]=it->value();
  }

  expression = boost::regex("^MEASURE_LOCAL\\[(.*)]$");
  for (alps::Parameters::const_iterator it=parms.begin();it != parms.end();++it) {
    std::string lhs = it->key();
    if (boost::regex_match(lhs, what, expression))
      local_expressions[what.str(1)]=it->value();
  }

  expression = boost::regex("^MEASURE_CORRELATIONS\\[(.*)]$");
  for (alps::Parameters::const_iterator it=parms.begin();it != parms.end();++it) {
    std::string lhs = it->key();
    if (boost::regex_match(lhs, what, expression)) {
      std::string key = what.str(1);
      std::string value = it->value();
      boost::regex expression2("^(.*):(.*)$");
      if (boost::regex_match(value, what, expression2))
        correlation_expressions[key] = std::make_pair(what.str(1), what.str(2));
      else
        correlation_expressions[key] = std::make_pair(value, value);
    }
  }

  expression = boost::regex("^MEASURE_STRUCTURE_FACTOR\\[(.*)]$");
  for (alps::Parameters::const_iterator it=parms.begin();it != parms.end();++it) {
    std::string lhs = it->key();
    if (boost::regex_match(lhs, what, expression)) {
      std::string key = what.str(1);
      std::string value = it->value();
      boost::regex expression2("^(.*):(.*)$");
      if (boost::regex_match(value, what, expression2))
        structurefactor_expressions[key] = std::make_pair(what.str(1), what.str(2));
      else
        structurefactor_expressions[key]=std::make_pair(value, value);
    }
  }
}
