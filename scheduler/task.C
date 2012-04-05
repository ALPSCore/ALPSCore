/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2003-2010 by Matthias Troyer <troyer@comp-phys.org>,
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
    from_file(true),
    started_(false)
{
  parse_task_file(true);
}

Task::Task(const ProcessList& w,const Parameters& p)
  : AbstractTask(w),
    parms(p),
    finished_(false),
    from_file(false),
    started_(false)
{
}

Task::~Task()
{
}

const Parameters& Task::get_parameters() const 
{ 
  return parms;
}


void Task::parse_task_file(bool read_parms_only)
{
  bool read_xml = true;
#ifdef ALPS_HAVE_HDF5
  std::string h5name;
  
  if (infilename.string().substr(infilename.string().size() - 3) == ".h5") {
    h5name=infilename.string();
    read_xml = false;
  }
  else
    h5name = infilename.string().substr(0, infilename.string().find_last_of('.')) + ".h5";
  
  if (boost::filesystem::exists(boost::filesystem::path(h5name))) {
    hdf5::archive ar(h5name);
    if (read_parms_only)
      ar >> make_pvp("/parameters",parms);
    else 
      ar >> make_pvp("", *this);
  } 
#endif
  if (read_xml) {
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
    if (!read_parms_only) {
      // scan for first worker element (e.g. <MCRUN> or <REALIZATION>)
      tag=parse_tag(infile,true);
      while (tag.name != closingtag) {
        handle_tag(infile,tag);
        tag=parse_tag(infile,true);
      }
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
  if (!parms.defined("SEED"))
    parms["SEED"]=0;
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

void Task::load(hdf5::archive & ar) {
    ar >> make_pvp("/parameters", parms);
}
void Task::save(hdf5::archive & ar) const {
    ar << make_pvp("/parameters", parms);
}

void Task::handle_tag(std::istream& infile, const XMLTag& tag)
{
  skip_element(infile,tag);
}

void Task::construct() // delayed until child class is fully constructed
{
  if(from_file)
    parse_task_file();
  if (!parms.defined("SEED"))
    parms["SEED"]=0;
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



// is it finished???
bool Task::finished(double& /* more_time */, double& /* percentage */ ) const
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
void Task::checkpoint(const boost::filesystem::path& fn, bool writeallxml) const
{
  boost::filesystem::path dir=fn.branch_path();
  bool make_backup = boost::filesystem::exists(fn);

#ifdef ALPS_HAVE_HDF5
  std::string task_path = fn.string().substr(0, fn.string().find_last_of('.')) + ".h5";
  std::string task_backup = fn.string().substr(0, fn.string().find_last_of('.')) + ".h5.bak";
  bool task_exists = boost::filesystem::exists(task_path);
  if (boost::filesystem::exists(task_backup))
      boost::filesystem::remove(task_backup);

  make_backup = make_backup || task_exists; 

  {
    hdf5::archive ar(make_backup ? task_backup : task_path, "a");
    ar << make_pvp("/",*this);
  } // close file
  
#endif

#ifndef ALPS_ONE_CHECKPOINT_FILE_ONLY
  boost::filesystem::path filename = (make_backup ? dir/(fn.filename().string()+".bak") : fn);
  {
    alps::oxstream out (filename);
    write_xml_header(out);
    out << parms;
    write_xml_body(out,fn,writeallxml);
    write_xml_trailer(out);
  } // close file
#endif

  if(make_backup) {
    if (boost::filesystem::exists(fn))
      boost::filesystem::remove(fn);
#ifndef ALPS_ONE_CHECKPOINT_FILE_ONLY
    boost::filesystem::rename(filename,fn);
#endif
#ifdef ALPS_HAVE_HDF5
    if (boost::filesystem::exists(task_path))
      boost::filesystem::remove(task_path);
    boost::filesystem::rename(task_backup, task_path);
#endif
  }
}

void Task::checkpoint_hdf5(const boost::filesystem::path& fn) const
{
#ifdef ALPS_HAVE_HDF5
  boost::filesystem::path dir=fn.branch_path();
  bool make_backup = boost::filesystem::exists(fn);

  std::string task_path = fn.string().substr(0, fn.string().find_last_of('.')) + ".h5";
  std::string task_backup = fn.string().substr(0, fn.string().find_last_of('.')) + ".h5.bak";
  bool task_exists = boost::filesystem::exists(task_path);
  if (boost::filesystem::exists(task_backup))
      boost::filesystem::remove(task_backup);

  make_backup = make_backup || task_exists; 

  {
    hdf5::archive ar(make_backup ? task_backup : task_path, "a");
    ar << make_pvp("/",*this);
  } // close file
  
  if(make_backup) {
    if (boost::filesystem::exists(task_path))
      boost::filesystem::remove(task_path);
    boost::filesystem::rename(task_backup, task_path);
  }
#endif
}

// checkpoint: save into a file
void Task::checkpoint_xml(const boost::filesystem::path& fn, bool writeallxml) const
{
  boost::filesystem::path dir=fn.branch_path();
  bool make_backup = boost::filesystem::exists(fn);

#ifndef ALPS_ONE_CHECKPOINT_FILE_ONLY
  boost::filesystem::path filename = (make_backup ? dir/(fn.filename().string()+".bak") : fn);
  {
    alps::oxstream out (filename);
    write_xml_header(out);
    out << parms;
    write_xml_body(out,fn,writeallxml);
    write_xml_trailer(out);
  } // close file
#endif

  if(make_backup) {
    if (boost::filesystem::exists(fn))
      boost::filesystem::remove(fn);
#ifndef ALPS_ONE_CHECKPOINT_FILE_ONLY
    boost::filesystem::rename(filename,fn);
#endif
  }
}


} // namespace scheduler
} // namespace alps
