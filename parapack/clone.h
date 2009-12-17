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

#ifndef PARAPACK_CLONE_H
#define PARAPACK_CLONE_H

#include "clone_info.h"
#include "clone_timer.h"
#include "worker_factory.h"
#include "types.h"

#include <alps/alea.h>
#include <alps/osiris.h>
#include <alps/parameter.h>
#include <alps/scheduler/info.h>

namespace alps {

ALPS_DECL bool load_observable(IDump& dp, Parameters& params, clone_info& info,
  std::vector<ObservableSet>& obs);
ALPS_DECL bool load_observable(boost::filesystem::path const& file, Parameters& params,
  clone_info& info,
  std::vector<ObservableSet>& obs);
ALPS_DECL bool load_observable(IDump& dp, std::vector<ObservableSet>& obs);
ALPS_DECL bool load_observable(boost::filesystem::path const& file,
  std::vector<ObservableSet>& obs);

class abstract_clone : public boost::noncopyable {
public:
  virtual ~abstract_clone() {}
  virtual void run() = 0;
  virtual bool halted() const = 0;
  virtual clone_info const& info() const = 0;

  virtual void load(IDump&) = 0;
  virtual void save(ODump&) const = 0;

  virtual void checkpoint() = 0;
  virtual void suspend() = 0;
};

class clone : public abstract_clone {
public:
  clone(tid_t tid, cid_t cid, Parameters const& params, boost::filesystem::path const& basedir,
    std::string const& base, clone_timer::duration_t const& check_interval, bool is_new);
  virtual ~clone();

  tid_t task_id() const { return task_id_; }
  cid_t clone_id() const { return clone_id_; }

  void run();
  bool halted() const;
  clone_info const& info() const;

  void load(IDump& dump);
  void save(ODump& dump) const;

  void checkpoint();
  void suspend();

  void output() const;

protected:
  void do_halt();

private:
  tid_t task_id_;
  cid_t clone_id_;

  Parameters params_;

  boost::filesystem::path basedir_;
  clone_info info_;
  std::vector<ObservableSet> measurements_;

  clone_timer timer_;
  clone_timer::loops_t loops_;

  boost::shared_ptr<parapack::abstract_worker> worker_;
};

} // end namespace alps

#endif // PARAPACK_CLONE_H
