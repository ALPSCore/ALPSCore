/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1997-2012 by Synge Todo <wistaria@comp-phys.org>,
*                            Ryo Igarashi <rigarash@issp.u-tokyo.ac.jp>,
*                            Haruhiko Matsuo <halm@rist.or.jp>,
*                            Tatsuya Sakashita <t-sakashita@issp.u-tokyo.ac.jp>,
*                            Yuichi Motoyama <yomichi@looper.t.u-tokyo.ac.jp>
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

/* $Id: parameters_p.h 5801 2011-10-18 14:51:54Z wistaria $ */

#ifndef NGS_PARAMS_P_H
#define NGS_PARAMS_P_H

#include <alps/parameter/parameter_p.h>
#include <alps/ngs/params.hpp>
#include <boost/classic_spirit.hpp>

namespace bs = boost::spirit;

namespace alps {

//
// XML support
//

/// \brief ALPS XML handler for the alps::params class
class ALPS_DECL ParamsXMLHandler : public CompositeXMLHandler {
public:
  ParamsXMLHandler(alps::params& p) : CompositeXMLHandler("PARAMETERS"),
    params_(p), parameter_(), handler_(parameter_) {
    add_handler(handler_);
  }

protected:
  void start_child(const std::string& name, const XMLAttributes& attributes,
                   xml::tag_type type) {
    if (type == xml::element) parameter_ = Parameter();
  }
  void end_child(const std::string& name, xml::tag_type type) {
    if (type == xml::element)
      params_[parameter_.key()] = parameter_.value();
  }

private:
  alps::params& params_;
  Parameter parameter_;
  ParameterXMLHandler handler_;
};

} // namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

/// \brief XML output of params
///
/// follows the schema on http://xml.comp-phys.org/
inline alps::oxstream& operator<<(alps::oxstream& oxs, const alps::params& p) {
  oxs << alps::start_tag("PARAMETERS");
  for (alps::params::const_iterator it = p.begin(); it != p.end(); ++it) {
    oxs << alps::start_tag("PARAMETER")
        << alps::attribute("name", it->first) << alps::no_linebreak
        << std::string(it->second)
        << alps::end_tag("PARAMETER");
  }
  oxs << alps::end_tag("PARAMETERS");
  return oxs;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif

#endif // NGS_PARAMS_P_H
