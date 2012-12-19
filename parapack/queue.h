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

#ifndef PARAPACK_QUEUE_H
#define PARAPACK_QUEUE_H

#include "job.h"
#include <boost/date_time/posix_time/posix_time.hpp>
#include <queue>

namespace alps {

//
// task_queue
//

struct task_queue_element_t {
  task_queue_element_t() {}
  task_queue_element_t(tid_t id, double w) : task_id(id), weight(w) {}
  task_queue_element_t(task const& t) : task_id(t.task_id()), weight(t.weight()) {}
  tid_t task_id;
  double weight;
};

bool operator<(task_queue_element_t const& lhs, task_queue_element_t const& rhs);

typedef std::priority_queue<task_queue_element_t> task_queue_t;


//
// check_queue
//

struct check_type {
  enum check_type_t {
    taskinfo,
    checkpoint,
    report,
    vmusage
  };
};
typedef check_type::check_type_t check_type_t;

struct check_queue_element_t {
  check_queue_element_t(check_type_t tp, boost::posix_time::ptime const& tm, tid_t tid, cid_t cid,
    gid_t gid) : type(tp), time(tm), task_id(tid), clone_id(cid), group_id(gid) {}
  check_type_t type;
  boost::posix_time::ptime time;
  tid_t task_id;
  cid_t clone_id;
  gid_t group_id;

  bool due() const;
};

ALPS_DECL bool operator<(check_queue_element_t const& lhs, check_queue_element_t const& rhs);

ALPS_DECL check_queue_element_t next_taskinfo(boost::posix_time::time_duration const& interval);

ALPS_DECL check_queue_element_t next_checkpoint(tid_t tid, cid_t cid, gid_t gid,
  boost::posix_time::time_duration const& interval);

ALPS_DECL check_queue_element_t next_report(tid_t tid, cid_t cid, gid_t gid,
  boost::posix_time::time_duration const& interval);

ALPS_DECL check_queue_element_t next_vmusage(boost::posix_time::time_duration const& interval);

typedef std::priority_queue<check_queue_element_t> check_queue_t;

} // end namespace alps

#endif // PARAPACK_QUEUE_H
