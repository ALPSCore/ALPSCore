/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2003 by Matthias Troyer <troyer@comp-phys.org>,
*                       Synge Todo <wistaria@comp-phys.org>
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

#include <alps/scheduler/task.h>
#include <alps/scheduler/types.h>
#include <alps/scheduler/scheduler.h>
#include <alps/expression.h>
#include <alps/parser/parser.h>

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/throw_exception.hpp>
#include <fstream>
#include <stdexcept>

namespace alps {
namespace scheduler {


void Task::print_copyright(std::ostream& out) 
{
  out << "Non-copyrighted program. Please insert your own copyright statement by overwriting the print_copyright static member function of your Task class.\n";
}


Task::Task(const ProcessList& w,const boost::filesystem::path& filename)
  : AbstractTask(w),
    finished_(false),
    infilename(filename),
    started_(false)//,
{
  parse_task_file(true);
}

Task::~Task()
{
}

void Task::parse_task_file(bool read_parms_only)
{
  boost::filesystem::ifstream infile(infilename);
  
  // read outermost tag (e.g. <SIMULATION>)
  XMLTag tag=parse_tag(infile,true);
  std::string closingtag = "/"+tag.name;
  
  // scan for <PARAMETERS> and read them
  tag=parse_tag(infile,true);
  while (tag.name!="PARAMETERS" && tag.name != closingtag) {
    skip_element(infile,tag);
    tag=parse_tag(infile,true);
  }
  parms.read_xml(tag,infile,true);
  if (!parms.defined("SEED"))
    parms["SEED"]=0;
  if (!read_parms_only) {
    // scan for first worker element (e.g. <MCRUN> or <REALIZATION>)
    tag=parse_tag(infile,true);
    while (tag.name != closingtag) {
      handle_tag(infile,tag);
      tag=parse_tag(infile,true); 
    }
  }
  // astreich, 06/20
  if (!parms.defined("ERROR_VARIABLE"))
    use_error_limit = false;
  else if (!parms.defined("ERROR_LIMIT")) {
    std::cerr << "Invalid input file: Error variable given without error limit\n";
    std::cerr << "Running simulation without error limit!\n";
    use_error_limit = false;
  } else
    use_error_limit = true;
}

/* astreich, 06/17 */
Parameters Task::parse_ext_task_file(std::string infilename)
{
  Parameters res;
  boost::filesystem::ifstream infile(infilename);
  
  // read outermost tag (e.g. <SIMULATION>)
  XMLTag tag=parse_tag(infile,true);
  std::string closingtag = "/"+tag.name;
  
  // scan for <PARAMETERS> and read them
  tag=parse_tag(infile,true);
  while (tag.name!="PARAMETERS" && tag.name != closingtag) {
    std::cerr << "skipping tag with name " << tag.name << "\n";
    skip_element(infile,tag);
    tag=parse_tag(infile,true);
  }
  res.read_xml(tag,infile,true);
  if (!res.defined("SEED"))
    res["SEED"]=0;
  return res;
}

void Task::handle_tag(std::istream& infile, const XMLTag& tag) 
{
  skip_element(infile,tag);
}

void Task::construct() // delayed until child class is fully constructed
{
  parse_task_file();
}

// start all runs which are active
void Task::start()
{
  started_=true;
}

void Task::run()
{
  if(started() && !finished_)
    dostep();
}

// start an extra run on a new node
void Task::add_process(const Process& /* p */)
{
  boost::throw_exception(std::runtime_error("Cannot add a process to a single process task"));
}


// remove one run : hope that a checkpoint was created before!!!
void Task::delete_process(const Process& /* p */)
{
  boost::throw_exception(std::runtime_error("Cannot delete a process from a single process task"));
}


// is it finished???
bool Task::finished(double& /* more_time */) const
{
  return finished_;
}

void Task::finish() 
{
  finished_=true;
}

// halt all active runs
void Task::halt()
{
  started_=false;
}


double Task::work() const
{
  return (finished_ ? 0. : (parms.defined("WORK_FACTOR") ? alps::evaluate<double>(parms["WORK_FACTOR"], parms) : 1. ));
}

// astreich, 06/23
ResultType Task::get_summary() const
{
  std::cerr << "should not call get_summary from Task ... \n";
  ResultType res;
  res.count = 0;
  return res;
}

void Task::write_xml_header(oxstream& out) const
{
  out << header("UTF-8") << stylesheet(xslt_path("ALPS.xsl"));
  out << start_tag("SIMULATION") << xml_namespace("xsi","http://www.w3.org/2001/XMLSchema-instance")
      << attribute("xsi:noNamespaceSchemaLocation","http://xml.comp-phys.org/2002/10/ALPS.xsd");
}


void Task::write_xml_trailer(oxstream& out) const
{
  out << end_tag("SIMULATION");
}

// checkpoint: save into a file
void Task::checkpoint(const boost::filesystem::path& fn) const
{
  boost::filesystem::path dir=fn.branch_path();
  bool make_backup = boost::filesystem::exists(fn);
  boost::filesystem::path filename = (make_backup ? dir/(fn.leaf()+".bak") : fn);
  {
  alps::oxstream out (filename);
  write_xml_header(out);
  out << parms;
  write_xml_body(out,fn);
  write_xml_trailer(out);
  } // close file
  if(make_backup) {
    boost::filesystem::remove(fn);
    boost::filesystem::rename(filename,fn);
  }
}


} // namespace scheduler
} // namespace alps
