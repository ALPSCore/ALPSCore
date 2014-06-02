/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef PARAPACK_CLONE_TIMER_H
#define PARAPACK_CLONE_TIMER_H

#include <alps/config.h>
#include <boost/date_time/posix_time/posix_time.hpp>

namespace alps {

class clone_timer {
public:
  typedef uint64_t loops_t;
  typedef boost::posix_time::ptime time_t;
  typedef boost::posix_time::time_duration duration_t;

  clone_timer(duration_t const& check_interval, double progress = 0) {
    interval_ = check_interval;
    reset(progress);
  }

  void reset(double progress = 0) {
    start_time_ = current_time();
    start_progress_ = progress;
    next_check_ = start_time_ + interval_;
  }

  static time_t current_time() { return boost::posix_time::microsec_clock::local_time(); }

  loops_t next_loops(loops_t loops) { return next_loops(loops, current_time()); }
  loops_t next_loops(loops_t loops, time_t const& current) {
    if (current > next_check_) {
      loops /= 2;
      if (loops == 0) loops = 1;
    } else if (current + interval_/2 < next_check_) {
      loops *= 2;
    }
    next_check_ = current + interval_;
    return loops;
  }

private:
  duration_t interval_;
  time_t start_time_;
  double start_progress_;
  time_t next_check_;
};

} // end namespace alps

#endif // PARAPACK_CLONE_TIMER_H
