/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2009 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

#ifndef ALPS_SCHEDULER_MONTECARLO_H
#define ALPS_SCHEDULER_MONTECARLO_H

#include <alps/scheduler/scheduler.h>
#include <alps/scheduler/task.h>
#include <alps/scheduler/worker.h>
#include <alps/model/model_helper.h>
#include <alps/model/sign.h>
#include <alps/lattice/graph_helper.h>
#include <alps/alea/observableset.h>
#include <boost/smart_ptr.hpp>
#include <alps/config.h>

#include <alps/hdf5.hpp>

namespace alps {
namespace scheduler {

class ALPS_DECL MCRun : public Worker
{
public:
  static void print_copyright(std::ostream&);

  MCRun(const ProcessList&,const alps::Parameters&,int);

  void save_worker(ODump&) const;
  void load_worker(IDump&);
  
  void save(hdf5::archive &) const;
  void load(hdf5::archive &);

  virtual void save(ODump&) const;
  virtual void load(IDump&);

  void write_xml(const boost::filesystem::path& name) const;
  const ObservableSet& get_measurements() const { return measurements;}
  ObservableSet get_compacted_measurements() const;
  ObservableSet get_and_remove_observable(const std::string& obsname, bool compact=false);

  std::string work_phase();
  void run();
  virtual bool is_thermalized() const;
  virtual bool handle_message(const Process& runmaster,int32_t tag);
protected:
  ObservableSet measurements;
};


class ALPS_DECL DummyMCRun : public MCRun
{
public:
  DummyMCRun(const ProcessList& w,const alps::Parameters& p,int n);
  DummyMCRun();
  void dostep();
  double work_done() const;
// astreich, 06.23
  ResultType get_summary() const;
};


class ALPS_DECL MCSimulation : public WorkerTask
{        
public:
  MCSimulation(const ProcessList& w, const boost::filesystem::path& p) 
      : WorkerTask(w,p) { 
    construct();
  }

  MCSimulation(const ProcessList& w, const Parameters& p) 
      : WorkerTask(w,p) 
  {
    construct();
  }

  static void print_copyright(std::ostream&) {}
  
  ObservableSet get_measurements(bool compact=false) const;
  
  ObservableSet get_and_remove_observable(const std::string& obsname, bool compact=false);
  //Collecting all measurements at once (by get_measurements()) requires to much memory for very large observable sets.
  //Returns an empty set if the observable does not exist
  
  MCSimulation& operator<<(const Observable& obs);
  void addObservable(const Observable& obs);

//protected:
  virtual ResultType get_summary() const;
  virtual ResultType get_summary(const std::string) const;

#ifdef ALPS_HAVE_HDF5
  void save(hdf5::archive &) const;
  void load(hdf5::archive &);
#endif

private:
  void accumulate_measurements(std::vector<std::pair<std::size_t, ObservableSet> > & all_measurements, ObservableSet const & measurements) const;
  std::string worker_tag() const;
  void write_xml_body(alps::oxstream&, boost::filesystem::path const& fn, bool) const;
  virtual void handle_tag(std::istream&, const XMLTag&);
  ObservableSet measurements;
};


template <class G=graph_helper<>::graph_type>
class LatticeMCRun : public MCRun, public graph_helper<G>
{
public:
  LatticeMCRun(const ProcessList& w,const alps::Parameters& p,int n)
   : MCRun(w,p,n), graph_helper<G>(parms)
  {}
};


template <class G=graph_helper<>::graph_type, class I=short>
class LatticeModelMCRun : public LatticeMCRun<G>, public model_helper<I>
{
public:  
  LatticeModelMCRun(const ProcessList& w, const alps::Parameters& p, int n,
                    bool issymbolic = false)
    : LatticeMCRun<G>(w, p, n),
      model_helper<I>(*this, LatticeMCRun<G>::parms, issymbolic)
  {}
  
  bool has_sign_problem() const 
  {
    return alps::has_sign_problem(model_helper<I>::model(), *this,
                                  LatticeMCRun<G>::parms);
  }
};


template <class WORKER>
class SimpleMCFactory : public BasicFactory<MCSimulation,WORKER>
{
public:
  SimpleMCFactory() {}
};

} // end namespace scheduler
} // end namespace alps

#endif
