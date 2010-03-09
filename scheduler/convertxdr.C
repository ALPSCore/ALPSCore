/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2002-2009 by Matthias Troyer <troyer@comp-phys.org>,
*                            Simon Trebst <trebst@comp-phys.org>,
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

/* $Id: convert2xml.C 3523 2009-12-12 05:52:24Z troyer $ */

#include <boost/filesystem/operations.hpp>

#include <alps/scheduler/convert.h>
#include <alps/osiris/xdrdump.h>
#include <alps/parser/xslt_path.h>
#include <alps/scheduler/montecarlo.h>
#include <boost/filesystem/path.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/throw_exception.hpp>
#include <alps/hdf5.hpp>
#include <fstream>
#include <stdexcept>

// an ugly hack for now - this needs to be refactord, see ticket #185
#include <boost/numeric/ublas/matrix.hpp>
#include "../../../applications/diag/diag.h"

namespace alps {

void convert_spectrum(const std::string& inname) 
{
  boost::filesystem::path p(inname, boost::filesystem::native);
  alps::ProcessList nowhere;
  DiagMatrix<double,boost::numeric::ublas::matrix<double,boost::numeric::ublas::column_major> > sim(nowhere,p);
  sim.checkpoint(p,true);
}

void convert_mc(const std::string& inname) 
{
  alps::scheduler::SimpleMCFactory<alps::scheduler::DummyMCRun> factory;
  alps::scheduler::init(factory);
  boost::filesystem::path p(inname, boost::filesystem::native);
  alps::ProcessList nowhere;
  alps::scheduler::MCSimulation sim(nowhere,p);
  sim.checkpoint(p,true);
}
 
void convert_xml(const std::string& inname)
{
  bool is_spectrum=false;
  std::string h5name = inname.substr(0, inname.find_last_of('.')) + ".h5";
  if (boost::filesystem::exists(boost::filesystem::path(h5name,boost::filesystem::native))) 
  {
    hdf5::iarchive ar(h5name);
    if (ar.is_group("/spectrum"))
      is_spectrum=true;
  }
  if (is_spectrum)
    convert_spectrum(inname);
  else
    convert_mc(inname);
}



void convert_params(const std::string& inname)
{
  alps::ParameterList list;
  {
    std::ifstream in(inname.c_str());
    in >> list;
  }

  std::string basename = boost::filesystem::path(inname,
    boost::filesystem::native).leaf();
  std::cout << "Converting parameter file " << inname << " to "
            <<  basename+".in.xml" << std::endl;

  alps::oxstream out(boost::filesystem::path((basename+".in.xml").c_str(),boost::filesystem::native));
  out << alps::header("UTF-8")
      << alps::stylesheet(alps::xslt_path("ALPS.xsl"))
      << alps::start_tag("JOB")
      << alps::xml_namespace("xsi","http://www.w3.org/2001/XMLSchema-instance")
      << alps::attribute("xsi:noNamespaceSchemaLocation",
                         "http://xml.comp-phys.org/2003/8/job.xsd")
      << alps::start_tag("OUTPUT")
      << alps::attribute("file", basename+".out.xml")
      << alps::end_tag("OUTPUT");

  for (unsigned int i = 0; i < list.size(); ++i) {
    std::string taskname =
      basename+".task"+boost::lexical_cast<std::string,int>(i+1);
    out << alps::start_tag("TASK") << alps::attribute("status","new")
        << alps::start_tag("INPUT")
        << alps::attribute("file", taskname + ".in.xml")
        << alps::end_tag("INPUT")
        << alps::start_tag("OUTPUT")
        << alps::attribute("file", taskname + ".out.xml")
        << alps::end_tag("OUTPUT")
        << alps::end_tag("TASK");
    //      out << "    <CPUS min=\"1\">\n";
    alps::oxstream task(boost::filesystem::path((taskname+".in.xml").c_str(),boost::filesystem::native));
    task << alps::header("UTF-8")
         << alps::stylesheet(alps::xslt_path("ALPS.xsl"));
    task << alps::start_tag("SIMULATION")
         << alps::xml_namespace("xsi",
                                "http://www.w3.org/2001/XMLSchema-instance")
         << alps::attribute("xsi:noNamespaceSchemaLocation",
                            "http://xml.comp-phys.org/2002/10/QMCXML.xsd");
    task << list[i];
    task << alps::end_tag("SIMULATION");
  }

  out << alps::end_tag("JOB");
}

void convert_run(const std::string& inname)
{
  boost::filesystem::path xdrpath(inname,boost::filesystem::native);
  boost::filesystem::path hdfpath(inname + ".h5",boost::filesystem::native);
  std::cout << "Converting run file " << inname << " to " <<  inname+".xml" <<std::endl;
  alps::scheduler::DummyMCRun run;
  run.load_from_file(xdrpath,hdfpath);
  run.write_xml(inname);
}

void convert_simulation(const std::string& inname)
{
  alps::IXDRFileDump dump(boost::filesystem::path(inname,boost::filesystem::native));
  if (static_cast<int>(dump)!=alps::scheduler::MCDump_task)
    boost::throw_exception(std::runtime_error("did not get a simulation on dump"));
  std::string jobname=inname+".xml";
  std::cout << "Converting simulation file " << inname << " to " <<  jobname << std::endl;
  alps::oxstream out(boost::filesystem::path(jobname,boost::filesystem::native));
  out << alps::header("UTF-8") << alps::stylesheet(alps::xslt_path("ALPS.xsl"))
      << alps::start_tag("SIMULATION") << alps::xml_namespace("xsi","http://www.w3.org/2001/XMLSchema-instance")
      << alps::attribute("xsi:noNamespaceSchemaLocation","http://xml.comp-phys.org/2002/10/QMCXML.xsd");
  int dummy_i;
  int version;
  int num;
  dump >> version; // version
  dump >> dummy_i;  // user version
  alps::Parameters parms;
  dump >> parms;
  out << parms;
  dump >> dummy_i; // nodes
  dump >> dummy_i; // seed
  dump >> num; // info size
  alps::scheduler::TaskInfo info;
  for (int i=0;i<num;++i)
    info.load(dump,version);
  // dump >> dummy_i; // flag if stored split
  num = static_cast<int>(dump);
  std::cout << num << " run(s)" << std::endl;
  for (int i=0;i<num;++i) {
    std::string srcname = inname+ ".run" + boost::lexical_cast<std::string,int>(i+1);
    out << alps::start_tag("MCRUN") << alps::start_tag("CHECKPOINT")
        << alps::attribute("format","osiris") << alps::attribute("file=","dstname")
        << alps::end_tag("CHECKPOINT") << alps::end_tag("MCRUN");
    convert_run(srcname);
  }
  out << alps::end_tag("SIMULATION");
}

void convert_scheduler(const std::string& inname)
{
  std::map<int,std::string> status_text;
  status_text[alps::scheduler::MasterScheduler::TaskNotStarted]="new";
  status_text[alps::scheduler::MasterScheduler::TaskRunning]="running";
  status_text[alps::scheduler::MasterScheduler::TaskHalted]="running";
  status_text[alps::scheduler::MasterScheduler::TaskFromDump]="running";
  status_text[alps::scheduler::MasterScheduler::TaskFinished]="finished";

  alps::IXDRFileDump dump(boost::filesystem::path(inname,boost::filesystem::native));
  if (static_cast<int>(dump)!=alps::scheduler::MCDump_scheduler)
    boost::throw_exception(std::runtime_error("did not get scheduler on dump"));
  std::string jobname=inname+".xml";
  std::cout << "Converting scheduler file " << inname << " to " <<  jobname << std::endl;
  alps::oxstream out(boost::filesystem::path(jobname,boost::filesystem::native));
  out << alps::header("UTF-8") << alps::stylesheet(alps::xslt_path("ALPS.xsl"))
    << alps::start_tag("JOB") << alps::xml_namespace("xsi","http://www.w3.org/2001/XMLSchema-instance")
    << alps::attribute("xsi:noNamespaceSchemaLocation","http://xml.comp-phys.org/2003/8/job.xsd");
  int dummy_i;
  double dummy_d;
  dump >> dummy_i; // version
  dump >> dummy_d;  // steptime
  alps::ParameterList list;
  dump >> list;
  std::vector<int> status;
  dump >> status;
  for (unsigned int i=0;i<list.size();++i)
    if (status[i]) {
      std::string xmlname = inname;
      std::string dumpname = inname;
      xmlname += ".task" + boost::lexical_cast<std::string,int>(i+1);
      if(boost::filesystem::exists(dumpname)) {
        out << alps::start_tag("TASK") << alps::attribute("status",status_text[status[i]])
          << alps::start_tag("INPUT") << alps::attribute("file",xmlname+".xml")
          << alps::end_tag("INPUT") << alps::end_tag("TASK");
        convert_simulation(xmlname);  
      }
    }
   out << alps::end_tag("JOB");
}

std::string convert2xml(std::string const& inname)
{
    alps::IXDRFileDump dump(boost::filesystem::path(inname,boost::filesystem::native));
    int type;
    dump >> type;
    switch (type) {
    case alps::scheduler::MCDump_scheduler:
      convert_scheduler(inname);
      return inname+".xml";
      break;
    case alps::scheduler::MCDump_task:
      convert_simulation(inname);
      return inname+".xml";
    case alps::scheduler::MCDump_run:
      convert_run(inname);
      return inname+".xml";
    default:
      {
        bool isxml=false;
        {
          std::ifstream is(inname.c_str());
          char c1=is.get();
          char c2=is.get();
          isxml = (c1=='<' && c2 =='?');
        }
        if (isxml)
          convert_xml(inname);
        else
          convert_params(inname);
      }
    }
  return inname+".in.xml";
}

} // end namespace
