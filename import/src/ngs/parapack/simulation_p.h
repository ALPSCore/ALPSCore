/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef NGS_PARAPACK_SIMULATION_P_H
#define NGS_PARAPACK_SIMULATION_P_H

#include <alps/ngs/parapack/clone_info_p.h>
#include <alps/ngs/parapack/params_p.h>
#include <alps/parser/xmlhandler.h>
#include <boost/filesystem/operations.hpp>

namespace alps {
namespace ngs_parapack {

struct simulation_xml_writer {
  simulation_xml_writer(boost::filesystem::path file, bool write_xml, bool make_backup,
    alps::params const& p, std::deque<clone_info> const& info) {
    boost::filesystem::path file_bak(file.branch_path() / (file.filename().string() + ".bak"));
    if (make_backup && exists(file)) rename(file, file_bak);
    oxstream os(file);
    os << header("UTF-8")
       << stylesheet(xslt_path("ALPS.xsl"))
       << start_tag("SIMULATION")
       << xml_namespace("xsi","http://www.w3.org/2001/XMLSchema-instance")
       << attribute("xsi:noNamespaceSchemaLocation", "http://xml.comp-phys.org/2002/10/ALPS.xsd");
    os << p;
    for (std::size_t i = 0; i < info.size(); ++i) os << info[i];
    os << end_tag("SIMULATION");
    if (make_backup && exists(file_bak)) remove(file_bak);
  }
};


class simulation_xml_handler : public CompositeXMLHandler {
public:
  simulation_xml_handler(alps::params& p, std::deque<clone_info>& clones) :
    CompositeXMLHandler("SIMULATION"), params_handler_(p),
    clones_(clones), clone_handler_(clone_) {
    add_handler(params_handler_);
    add_handler(clone_handler_);
  }

protected:
  void start_child(const std::string& name, alps::XMLAttributes const& /* attributes */,
    xml::tag_type type) {
    if (type == alps::xml::element) {
      if (name == "MCRUN")
        clone_ = clone_info();
    }
  }

  void end_child(const std::string& name, xml::tag_type type) {
    if (type == alps::xml::element) {
      if (name == "MCRUN")
        clones_.push_back(clone_);
    }
  }

private:
  ParamsXMLHandler params_handler_;
  std::deque<clone_info>& clones_;
  clone_info clone_;
  clone_info_xml_handler clone_handler_;
};

class simulation_parameters_xml_handler : public CompositeXMLHandler {
public:
  simulation_parameters_xml_handler(alps::params& p) :
    CompositeXMLHandler("SIMULATION"), params_handler_(p),
    clone_handler_("MCRUN") {
    add_handler(params_handler_);
    add_handler(clone_handler_);
  }

private:
  ParamsXMLHandler params_handler_;
  DummyXMLHandler clone_handler_;
};

} // end namespace ngs_parapack
} // end namespace alps

#endif // NGS_PARAPACK_SIMULATION_P_H
