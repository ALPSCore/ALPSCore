/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef PARAPACK_TYPES_H
#define PARAPACK_TYPES_H

#include <alps/scheduler/signal.hpp>
#include <alps/scheduler/types.h>
#include "integer_range.h"

namespace alps {

typedef alps::scheduler::SignalHandler signal_handler;
typedef alps::scheduler::SignalHandler::SignalInfo signal_info_t;
typedef alps::scheduler::SignalHandler signal_info;

namespace parallel_type {

struct single {};
struct mpi {};

}

struct parapack_dump {
  enum parapack_dump_t {
    run_master = scheduler::MCDump_run_master,
    run_slave = scheduler::MCDump_run_slave,

    initial_version = 303, // oldest supported version of dump
    current_version = 305
    // from 304: vector of ObservableSet is stored instead of ObservableSet itself
    // from 305: added user and version information
  };
};
typedef parapack_dump::parapack_dump_t parapack_dump_t;

struct dump_format {
  enum dump_format_t {
    hdf5, // default
    xdr
  };
  static std::string to_string(dump_format_t format);
};
typedef dump_format::dump_format_t dump_format_t;

struct dump_policy {
  enum dump_policy_t {
    Never,
    RunningOnly, // default
    All
  };
  static std::string to_string(dump_policy_t policy);
};
typedef dump_policy::dump_policy_t dump_policy_t;

struct mcmp_tag {
  enum mcmp_tag_t {
    scheduler_halt,

    clone_create,
    clone_checkpoint,
    clone_info,
    clone_suspend,
    clone_halt,

    process_vmusage,

    do_step,
    do_nothing
  };
};
typedef mcmp_tag::mcmp_tag_t mcmp_tag_t;

struct task_status {
  enum task_status_t {
    Undefined,
    Ready,       // on memory  but not started
    Running,     //            0 < progress < 1
    Continuing,  //            1 <= progress < max_work
    Idling,      //            progress >= max_work
    NotStarted,  // on dump    progress = 0
    Suspended,   //            0 < progress < 1
    Finished,    //            1 <= progress < max_work
    Completed    //            progress >= max_work
  };
  static task_status_t status(std::string const& str);
  static std::string to_string(task_status_t status);
};
typedef task_status::task_status_t task_status_t;

struct clone_status {
  enum clone_status_t {
    Undefined,
    NotStarted,
    Running,     // on memory  progress < 1
    Idling,      //            progress >= 1
    Suspended,   // on dump    progress < 1
    Finished,    //            progress >= 1
    Stopping
  };
  static clone_status_t status(double progress);
  static clone_status_t status(double progress, bool on_memory);
  static clone_status_t status(std::string const& str);
  static std::string to_string(clone_status_t status);
};
typedef clone_status::clone_status_t clone_status_t;

typedef uint32_t tid_t;
typedef uint32_t cid_t;
typedef uint32_t gid_t;

typedef uint32_t seed_t;

typedef integer_range<uint32_t> task_range_t;

} // end namespace alps

#endif // PARAPACK_TYPES_H
