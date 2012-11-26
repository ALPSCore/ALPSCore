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

#ifndef NGS_PARAPACK_CLONE_INFO_P_H
#define NGS_PARAPACK_CLONE_INFO_P_H

#include <alps/ngs/parapack/clone_info.h>
#include <alps/parapack/util.h>
#include <alps/parser/xmlhandler.h>

namespace alps {
namespace ngs_parapack {

class clone_phase_xml_handler : public CompositeXMLHandler {
public:
  clone_phase_xml_handler(clone_phase& phase) :
    CompositeXMLHandler("EXECUTED"), phase_(phase),
    from_handler_("FROM", from_str_), to_handler_("TO", to_str_),
    name_handler_("NAME", name_),
    machine_handler_("MACHINE", phase.hosts_, name_, name_handler_),
    user_handler_("USER", phase_.user_) {
    add_handler(from_handler_);
    add_handler(to_handler_);
    add_handler(machine_handler_);
    add_handler(user_handler_);
  }

protected:
  void start_top(std::string const&, XMLAttributes const& attributes, xml::tag_type) {
    phase_.hosts_.clear();
    phase_.user_ = "";
    np_ = 0;
    if (attributes.defined("processes"))
      np_ = boost::lexical_cast<int>(attributes["processes"]);
    if (attributes.defined("phase"))
      phase_.phase_ = attributes["phase"];
  }

  void end_top(std::string const&, xml::tag_type) {
    if (np_ && phase_.hosts_.size() != np_)
      boost::throw_exception(std::runtime_error("inconsistent number of processes in <EXECUTED>"));
  }

  void end_child(std::string const& name , xml::tag_type) {
    if (name == "FROM")
      phase_.startt_ = boost::posix_time::time_from_string(from_str_);
    if (name == "TO")
      phase_.stopt_ = boost::posix_time::time_from_string(to_str_);
  }

private:
  clone_phase& phase_;
  std::size_t np_;
  std::string from_str_, to_str_;
  SimpleXMLHandler<std::string> from_handler_;
  SimpleXMLHandler<std::string> to_handler_;
  std::string name_;
  SimpleXMLHandler<std::string> name_handler_;
  VectorXMLHandler<std::string> machine_handler_;
  SimpleXMLHandler<std::string> user_handler_;
};

class clone_info_xml_handler : public CompositeXMLHandler {
public:

  clone_info_xml_handler(clone_info& info) :
    CompositeXMLHandler("MCRUN"), info_(info), phase_handler_(phase_),
    checkpoint_handler_("CHECKPOINT", checkpoint_, "file"),
    workerseed_handler_("SEED", seed_, "value"),
    disorder_seed_handler_("DISORDER_SEED", info_.disorder_seed_, "value") {
    add_handler(phase_handler_);
    add_handler(checkpoint_handler_);
    add_handler(workerseed_handler_);
    add_handler(disorder_seed_handler_);
  }

protected:
  void start_top(std::string const&, XMLAttributes const& attributes, xml::tag_type) {
    info_.phases_.clear();
    info_.dumpfiles_.clear();
    info_.worker_seed_.clear();
    np_ = attributes.defined("processes") ? boost::lexical_cast<int>(attributes["processes"]) : 0;
    info_.clone_id_ =
      attributes.defined("id") ? (boost::lexical_cast<int>(attributes["id"]) - 1) : 0;
    info_.progress_ =
      attributes.defined("progress") ? parse_percentage(attributes["progress"]) : 0.;
  }

  void end_top(std::string const&, xml::tag_type) {
    if (np_ && info_.dumpfiles_.size() && info_.dumpfiles_.size() != np_)
      boost::throw_exception(std::runtime_error(
        "inconsistent number of checkpoint files in <MCRUN>"));
    if (np_ && info_.worker_seed_.size() && info_.worker_seed_.size() != np_)
      boost::throw_exception(std::runtime_error("inconsistent number of random seed in <MCRUN>"));
  }

  void end_child(std::string const& name, xml::tag_type type) {
    if (type == xml::element) {
      if (name == "EXECUTED") {
        if (np_ && phase_.hosts().size() && phase_.hosts().size() != np_)
          boost::throw_exception(std::runtime_error("inconsistent number of processes in <MCRUN>"));
        info_.phases_.push_back(phase_);
      } else if (name == "CHECKPOINT") {
        info_.dumpfiles_ .push_back(checkpoint_);
      } else if (name == "SEED") {
        info_.worker_seed_ .push_back(seed_);
      }
    }
  }

private:
  clone_info& info_;
  std::size_t np_;
  clone_phase phase_;
  clone_phase_xml_handler phase_handler_;
  std::string checkpoint_;
  SimpleXMLHandler<std::string> checkpoint_handler_;
  seed_t seed_;
  SimpleXMLHandler<seed_t> workerseed_handler_;
  SimpleXMLHandler<seed_t> disorder_seed_handler_;
};

} // end namespace ngs_parapack
} // end namespace alps

#endif // NGS_PARAPACK_CLONE_INFO_P_H
