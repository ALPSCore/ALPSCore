/*****************************************************************************
*
* ALPS Project Applications
*
* Copyright (C) 1994-2006 by Matthias Troyer <troyer@comp-phys.org>
*
* This software is part of the ALPS Applications, published under the ALPS
* Application License; you can use, redistribute it and/or modify it under
* the terms of the license, either version 1 or (at your option) any later
* version.
* 
* You should have received a copy of the ALPS Application License along with
* the ALPS Applications; see the file LICENSE.txt. If not, the license is also
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

alps::MeasurementOperators::MeasurementOperators (Parameters const& parms)
{ 
  boost::regex expression("^MEASURE_AVERAGE\\[(.*)]$");
  boost::cmatch what;
  for (alps::Parameters::const_iterator it=parms.begin();it != parms.end();++it)
    if (boost::regex_match(static_cast<std::string>(it->key()).c_str(), what, expression))
      average_expressions[std::string(what[1].first,what[1].second)]=it->value();

  expression = boost::regex("^MEASURE_LOCAL\\[(.*)]$");
  for (alps::Parameters::const_iterator it=parms.begin();it != parms.end();++it)
    if (boost::regex_match(static_cast<std::string>(it->key()).c_str(), what, expression))
      local_expressions[std::string(what[1].first,what[1].second)]=it->value();
      
  expression = boost::regex("^MEASURE_CORRELATIONS\\[(.*)]$");
  for (alps::Parameters::const_iterator it=parms.begin();it != parms.end();++it)
    if (boost::regex_match(static_cast<std::string>(it->key()).c_str(), what, expression)) {
      std::string key(what[1].first,what[1].second);
      boost::regex expression2("^(.*):(.*)$");
      if (boost::regex_match(static_cast<std::string>(it->value()).c_str(), what, expression2))
        correlation_expressions[key]=std::make_pair(std::string(what[1].first,what[1].second),std::string(what[2].first,what[2].second));
      else
        correlation_expressions[key]=std::make_pair(std::string(it->value()),std::string(it->value()));
    }
  }

