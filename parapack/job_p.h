/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1997-2009 by Synge Todo <wistaria@comp-phys.org>
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

#ifndef PARAPACK_JOB_P_H
#define PARAPACK_JOB_P_H

#include "job.h"
#include "util.h"

// some file (probably a python header) defines a tolower macro ...
#undef tolower
#undef toupper

#include <boost/filesystem/operations.hpp>
#include <boost/regex.hpp>
#include <vector>

namespace alps {

struct job_xml_writer {
  job_xml_writer(boost::filesystem::path const& file, std::string const& simulation_name,
    std::string const& file_in_str, std::string const& file_out_str,
    std::string const& alps_version_str, std::string const& application_version_str,
    std::vector<task> const& tasks, bool make_backup) {
    boost::filesystem::path file_bak(file.branch_path() / (file.filename().string() + ".bak"));
    if (make_backup && exists(file)) rename(file, file_bak);
    oxstream os(file);
    os << header("UTF-8")
       << stylesheet(xslt_path("ALPS.xsl"))
       << start_tag("JOB")
       << xml_namespace("xsi","http://www.w3.org/2001/XMLSchema-instance")
       << attribute("xsi:noNamespaceSchemaLocation", "http://xml.comp-phys.org/2003/8/job.xsd");
    if (simulation_name != "")
      os << attribute("name", simulation_name);
    if (alps_version_str != "")
      os << start_tag("VERSION") << attribute("type", "alps")
         << attribute("string", alps_version_str) << end_tag("VERSION");
    if (application_version_str != "")
      os << start_tag("VERSION") << attribute("type", "application")
         << attribute("string", application_version_str) << end_tag("VERSION");
    os << start_tag("INPUT") << attribute("file", file_in_str) << end_tag("INPUT")
       << start_tag("OUTPUT") << attribute("file", file_out_str) << end_tag("OUTPUT");
    BOOST_FOREACH(task const& t, tasks) t.write_xml_summary(os);
    os << end_tag("JOB");
    if (make_backup && exists(file_bak)) remove(file_bak);
  }
};


class job_task_xml_handler : public XMLHandlerBase {
public:
  job_task_xml_handler(task& t) : XMLHandlerBase("TASK"), task_(t) {}

  void start_element(std::string const& name, XMLAttributes const& attributes, xml::tag_type type) {
    if (type == xml::element) {
      if (name == "TASK") {
        task_.progress_ = attributes.defined("progress") ?
          parse_percentage(attributes["progress"]) : 0;
        task_.weight_ = attributes.defined("weight") ?
          boost::lexical_cast<double>(attributes["weight"]) : 3;
        task_.dump_weight_ = task_.weight_;
        if (attributes.defined("status"))
          task_.status_ = task_status::status(attributes["status"]);
        else
          boost::throw_exception(std::runtime_error("missing status attribute in <TASK> tag"));
      } else if (name == "INPUT") {
        if (attributes.defined("file"))
          task_.file_in_str_ = attributes["file"];
        else
          boost::throw_exception(std::runtime_error("missing file attribute in <INPUT> tag"));
      } else if (name == "OUTPUT") {
        if (attributes.defined("file"))
          task_.file_out_str_ = attributes["file"];
        else
          boost::throw_exception(std::runtime_error("missing file attribute in <OUTPUT> tag"));
      } else {
        boost::throw_exception(std::runtime_error("unknown tag name : " + name));
      }
    }
  }

  void end_element(const std::string& name, xml::tag_type type) {
    if (type == xml::element && name == "TASK") {
      if (task_.file_in_str_.empty())
        boost::throw_exception(std::runtime_error("missing <INPUT> tag in <TASK> tag"));
      if (task_.file_out_str_.empty())
        task_.file_out_str_ = task_.file_in_str_;
      task_.base_ = regex_replace(task_.file_out_str_, boost::regex("\\.out\\.xml$"), "");
    }
  }

  void text(std::string const& /* text */) {
    boost::throw_exception(std::runtime_error("text contents not allowed here in <TASK> tag"));
  }

private:
  task& task_;
};


class filename_xml_handler : public XMLHandlerBase {
public:
  filename_xml_handler(std::string& file_in_str, std::string& file_out_str, bool& is_master) :
    XMLHandlerBase("dummy"), in_(file_in_str), out_(file_out_str), master_(is_master),
    found_(false), in_task_(false) {
  }
  virtual ~filename_xml_handler() {
    if (!found_) boost::throw_exception(std::runtime_error("no valid tag found"));
  }

  void start_element(std::string const& name, XMLAttributes const& attributes, xml::tag_type type) {
    if (type == xml::element) {
      if (!found_) {
        if (name == "JOB") {
          found_ = true;
          master_ = true;
        } else if (name == "SIMULATION") {
          found_ = true;
          master_ = false;
        } else {
          boost::throw_exception(std::runtime_error("unknown tag " + name));
        }
      } else {
        if (master_ && !in_task_) {
          if (name == "INPUT") {
            if (attributes.defined("file")) {
              in_ = attributes["file"];
            } else {
              boost::throw_exception(std::runtime_error("missing file attribute in <INPUT> tag"));
            }
          } else if (name == "OUTPUT") {
            if (attributes.defined("file")) {
              out_ = attributes["file"];
            } else {
              boost::throw_exception(std::runtime_error("missing file attribute in <OUTPUT> tag"));
            }
          } else if (name == "TASK") {
            in_task_ = true;
          }
        }
      }
    }
  }

  void end_element(std::string const& name, xml::tag_type type) {
    if (type == xml::element && in_task_ && name == "TASK") in_task_ = false;
  }

  void text(std::string const& /* text */) {}

private:
  std::string& in_;
  std::string& out_;
  bool& master_;
  bool found_;
  bool in_task_;
};


class version_xml_handler : public XMLHandlerBase {
public:
  version_xml_handler(std::vector<std::pair<std::string, std::string> >& versions) :
    XMLHandlerBase("dummy"), versions_(versions) {
  }
  virtual ~version_xml_handler() {}

  void start_element(std::string const& name, XMLAttributes const& attributes, xml::tag_type type) {
    if (type == xml::element && name == "VERSION") {
      versions_.push_back(std::make_pair(attributes["type"], attributes["string"]));
    }
  }

  void end_element(std::string const& /* name */, xml::tag_type /* type */) {}

  void text(std::string const& /* text */) {}

private:
  std::vector<std::pair<std::string, std::string> >& versions_;
};


class job_tasks_xml_handler : public CompositeXMLHandler {
public:
  job_tasks_xml_handler(std::string& simname, std::vector<task>& tasks,
    boost::filesystem::path const& basedir) :
    CompositeXMLHandler("JOB"), simname_(simname), tasks_(tasks), basedir_(basedir), tid_(0),
    task_(), task_handler_(task_) {
    add_handler(task_handler_);
  }

protected:
  void start_top(std::string const& /* name */, XMLAttributes const& attributes,
    xml::tag_type /* type */) {
    if (attributes.defined("name"))
      simname_ = attributes["name"];
    else
      simname_ = "";
  }
  void start_child(std::string const& name, XMLAttributes const&, xml::tag_type type) {
    if (type == xml::element && name == "TASK") {
      task_ = task();
      task_.task_id_ = tid_++;
      task_.basedir_ = basedir_;
    }
  }
  void end_child(std::string const& name, xml::tag_type type) {
    if (type == xml::element && name == "TASK") tasks_.push_back(task_);
  }

  bool start_element_impl(std::string const& name, XMLAttributes const&, xml::tag_type type) {
    return (type == xml::element && (name == "INPUT" || name == "OUTPUT" || name == "VERSION"));
  }

  bool end_element_impl(std::string const& name, xml::tag_type type) {
    return (type == xml::element && (name == "INPUT" || name == "OUTPUT" || name == "VERSION"));
  }

private:
  std::string& simname_;
  std::vector<task>& tasks_;
  boost::filesystem::path basedir_;
  tid_t tid_;
  task task_;
  job_task_xml_handler task_handler_;
};

} // end namespace alps

#endif // PARAPACK_JOB_P_H
