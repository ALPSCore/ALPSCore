/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

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
