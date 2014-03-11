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

#include <alps/config.h>

#include <alps/hdf5.hpp>

#include <alps/scheduler/convert.h>
#include <alps/osiris/xdrdump.h>
#include <alps/parser/xslt_path.h>
#include <alps/scheduler/montecarlo.h>
#include <alps/scheduler/diag.hpp>

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/throw_exception.hpp>

#include <fstream>
#include <stdexcept>

namespace alps {

void convert_spectrum(const std::string& inname) 
{
  boost::filesystem::path p(inname);
  ProcessList nowhere;
  scheduler::DiagTask<double> sim(nowhere,p);
  sim.checkpoint(p,true);
}

void convert_mc(const std::string& inname) 
{
  scheduler::SimpleMCFactory<scheduler::DummyMCRun> factory;
  scheduler::init(factory);
  boost::filesystem::path p(inname);
  ProcessList nowhere;
  scheduler::MCSimulation sim(nowhere,p);
  sim.checkpoint(p,true);
}
 
void convert_xml(const std::string& inname)
{
  bool is_spectrum=false;
  std::string h5name = inname.substr(0, inname.find_last_of('.')) + ".h5";
  if (boost::filesystem::exists(boost::filesystem::path(h5name))) 
  {
    hdf5::archive ar(h5name);
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
  ParameterList list;
  {
    std::ifstream in(inname.c_str());
    in >> list;
  }

  std::string basename = boost::filesystem::path(inname).filename().string();
  std::cout << "Converting parameter file " << inname << " to "
            <<  basename+".in.xml" << std::endl;

  oxstream out(boost::filesystem::path((basename+".in.xml").c_str()));
  out << header("UTF-8")
      << stylesheet(xslt_path("ALPS.xsl"))
      << start_tag("JOB")
      << xml_namespace("xsi","http://www.w3.org/2001/XMLSchema-instance")
      << attribute("xsi:noNamespaceSchemaLocation",
                         "http://xml.comp-phys.org/2003/8/job.xsd")
      << start_tag("OUTPUT")
      << attribute("file", basename+".out.xml")
      << end_tag("OUTPUT");

  for (unsigned int i = 0; i < list.size(); ++i) {
    std::string taskname =
      basename+".task"+boost::lexical_cast<std::string,int>(i+1);
    out << start_tag("TASK") << attribute("status","new")
        << start_tag("INPUT")
        << attribute("file", taskname + ".in.xml")
        << end_tag("INPUT")
        << start_tag("OUTPUT")
        << attribute("file", taskname + ".out.xml")
        << end_tag("OUTPUT")
        << end_tag("TASK");
    //      out << "    <CPUS min=\"1\">\n";
    oxstream task(boost::filesystem::path((taskname+".in.xml").c_str()));
    task << header("UTF-8")
         << stylesheet(xslt_path("ALPS.xsl"));
    task << start_tag("SIMULATION")
         << xml_namespace("xsi",
                                "http://www.w3.org/2001/XMLSchema-instance")
         << attribute("xsi:noNamespaceSchemaLocation",
                            "http://xml.comp-phys.org/2002/10/QMCXML.xsd");
    task << list[i];
    task << end_tag("SIMULATION");
  }

  out << end_tag("JOB");
}

void convert_run(const std::string& inname)
{
  boost::filesystem::path xdrpath(inname);
  boost::filesystem::path hdfpath(inname + ".h5");
  std::cout << "Converting run file " << inname << " to " <<  inname+".xml" <<std::endl;
  scheduler::DummyMCRun run;
  run.load_from_file(xdrpath,hdfpath);
  run.write_xml(inname);
}

void convert_simulation(const std::string& inname)
{
  IXDRFileDump dump=IXDRFileDump(boost::filesystem::path(inname));
  if (static_cast<int>(dump)!=scheduler::MCDump_task)
    boost::throw_exception(std::runtime_error("did not get a simulation on dump"));
  std::string jobname=inname+".xml";
  std::cout << "Converting simulation file " << inname << " to " <<  jobname << std::endl;
  boost::filesystem::path pjobname(jobname);
  oxstream out(pjobname);
  out << header("UTF-8") << stylesheet(xslt_path("ALPS.xsl"))
      << start_tag("SIMULATION") << xml_namespace("xsi","http://www.w3.org/2001/XMLSchema-instance")
      << attribute("xsi:noNamespaceSchemaLocation","http://xml.comp-phys.org/2002/10/QMCXML.xsd");
  int dummy_i;
  int version;
  int num;
  dump >> version; // version
  dump >> dummy_i;  // user version
  Parameters parms;
  dump >> parms;
  out << parms;
  dump >> dummy_i; // nodes
  dump >> dummy_i; // seed
  dump >> num; // info size
  scheduler::TaskInfo info;
  for (int i=0;i<num;++i)
    info.load(dump,version);
  // dump >> dummy_i; // flag if stored split
  num = static_cast<int>(dump);
  std::cout << num << " run(s)" << std::endl;
  for (int i=0;i<num;++i) {
    std::string srcname = inname+ ".run" + boost::lexical_cast<std::string,int>(i+1);
    out << start_tag("MCRUN") << start_tag("CHECKPOINT")
        << attribute("format","osiris") << attribute("file=","dstname")
        << end_tag("CHECKPOINT") << end_tag("MCRUN");
    convert_run(srcname);
  }
  out << end_tag("SIMULATION");
}

void convert_scheduler(const std::string& inname)
{
  std::map<int,std::string> status_text;
  status_text[scheduler::MasterScheduler::TaskNotStarted]="new";
  status_text[scheduler::MasterScheduler::TaskRunning]="running";
  status_text[scheduler::MasterScheduler::TaskHalted]="running";
  status_text[scheduler::MasterScheduler::TaskFromDump]="running";
  status_text[scheduler::MasterScheduler::TaskFinished]="finished";

  IXDRFileDump dump=IXDRFileDump(boost::filesystem::path(inname));
  if (static_cast<int>(dump)!=scheduler::MCDump_scheduler)
    boost::throw_exception(std::runtime_error("did not get scheduler on dump"));
  std::string jobname=inname+".xml";
  std::cout << "Converting scheduler file " << inname << " to " <<  jobname << std::endl;
  boost::filesystem::path pjobname(jobname);
  oxstream out(pjobname);
  out << header("UTF-8") << stylesheet(xslt_path("ALPS.xsl"))
    << start_tag("JOB") << xml_namespace("xsi","http://www.w3.org/2001/XMLSchema-instance")
    << attribute("xsi:noNamespaceSchemaLocation","http://xml.comp-phys.org/2003/8/job.xsd");
  int dummy_i;
  double dummy_d;
  dump >> dummy_i; // version
  dump >> dummy_d;  // steptime
  ParameterList list;
  dump >> list;
  std::vector<int> status;
  dump >> status;
  for (unsigned int i=0;i<list.size();++i)
    if (status[i]) {
      std::string xmlname = inname;
      std::string dumpname = inname;
      xmlname += ".task" + boost::lexical_cast<std::string,int>(i+1);
      if(boost::filesystem::exists(dumpname)) {
        out << start_tag("TASK") << attribute("status",status_text[status[i]])
          << start_tag("INPUT") << attribute("file",xmlname+".xml")
          << end_tag("INPUT") << end_tag("TASK");
        convert_simulation(xmlname);  
      }
    }
   out << end_tag("JOB");
}

std::string convert2xml(std::string const& inname)
{
    IXDRFileDump dump=IXDRFileDump(boost::filesystem::path(inname));
    int type;
    dump >> type;
    switch (type) {
    case scheduler::MCDump_scheduler:
      convert_scheduler(inname);
      return inname+".xml";
    case scheduler::MCDump_task:
      convert_simulation(inname);
      return inname+".xml";
    case scheduler::MCDump_run:
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
