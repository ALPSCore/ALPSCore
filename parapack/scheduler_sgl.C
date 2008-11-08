/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1997-2008 by Synge Todo <wistaria@comp-phys.org>
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

#include "scheduler.h"
#include "clone.h"
#include "clone_proxy.h"
#include "filelock.h"
#include "queue.h"

#include <boost/filesystem/operations.hpp>
#include <boost/foreach.hpp>
#include <boost/timer.hpp>
#include <iostream>
#include <string>
#include <vector>

namespace alps {
namespace parapack {

int run_sequential(int argc, char **argv) {

#ifndef BOOST_NO_EXCEPTIONS
  try {
#endif

  alps::ParameterList parameterlist;
  switch (argc) {
  case 1:
    std::cin >> parameterlist;
    break;
  case 2: {
    boost::filesystem::ifstream is(argv[1]);
    is >> parameterlist;
    break;
  }
  default:
    boost::throw_exception(std::invalid_argument(
      "Usage: " + std::string(argv[0]) + " [paramfile]"));
  }

  for (int i = 0; i < parameterlist.size(); ++i) {
    alps::Parameters p = parameterlist[i];
    boost::timer tm;
    if (!p.defined("DIR_NAME")) p["DIR_NAME"] = ".";
    if (!p.defined("BASE_NAME")) p["BASE_NAME"] = "task" + boost::lexical_cast<std::string>(i+1);
    if (!p.defined("SEED")) p["SEED"] = static_cast<unsigned int>(time(0));
    if (!p.defined("WORKER_SEED")) p["WORKER_SEED"] = p["SEED"];
    if (!p.defined("DISORDER_SEED")) p["DISORDER_SEED"] = p["WORKER_SEED"];
    std::cout << "[input parameters]\n" << p;
    std::vector<alps::ObservableSet> obs;
    boost::shared_ptr<alps::parapack::abstract_worker>
      worker = worker_factory::make_worker(p);
    worker->init_observables(p, obs);
    bool thermalized = worker->is_thermalized();
    if (thermalized) BOOST_FOREACH(alps::ObservableSet& o, obs) { o.reset(true); }
    while (worker->progress() < 1.0) {
      worker->run(obs);
      if (!thermalized && worker->is_thermalized()) {
        BOOST_FOREACH(alps::ObservableSet& o, obs) { o.reset(true); }
        thermalized = true;
      }
    }
    boost::shared_ptr<alps::parapack::abstract_evaluator>
      evaluator = evaluator_factory::make_evaluator(p);
    std::vector<alps::ObservableSet> obs_out;
    evaluator->load(obs, obs_out);
    evaluator->evaluate(obs_out);
    std::cerr << "[speed]\nelapsed time = " << tm.elapsed() << " sec\n";
    std::cout << "[results]\n";
    if (obs_out.size() == 1) {
      std::cout << obs_out[0];
    } else {
      for (int i = 0; i < obs_out.size(); ++i)
        std::cout << "[[replica " << i << "]]\n" << obs_out[i];
    }
  }

#ifndef BOOST_NO_EXCEPTIONS
  }
  catch (const std::exception& excp) {
    std::cerr << excp.what() << std::endl;
    alps::comm_exit(true);
    return -1; }
  catch (...) {
    std::cerr << "Unknown exception occurred!" << std::endl;
    alps::comm_exit(true);
    return -1; }
#endif
  return 0;
}

namespace scheduler {

int start(int argc, char** argv) {
  option opt(argc, argv);
  if (!opt.valid) return -1;

  print_copyright(std::clog);

  boost::posix_time::ptime end_time =
    boost::posix_time::second_clock::local_time() + opt.time_limit;

  BOOST_FOREACH(std::string const& file_str, opt.jobfiles) {

    process_helper process;

    boost::filesystem::path file = complete(boost::filesystem::path(file_str)).normalize();
    boost::filesystem::path basedir = file.branch_path();

    // only for master scheduler
    signal_handler signals;
    std::string file_in_str;
    std::string file_out_str;
    boost::filesystem::path file_in;
    boost::filesystem::path file_out;
    filelock master_lock;
    std::string simname;
    std::vector<task> tasks;
    task_queue_t task_queue;
    check_queue_t check_queue;
    int num_finished_tasks = 0;

    //
    // evaluation only
    //

    if (opt.evaluate_only) {
      std::clog << log_header() << "starting evaluation on " << alps::hostname() << std::endl;
      int t = scheduler::load_filename(file, file_in_str, file_out_str);
      if (t == 1) {
        file_in = complete(boost::filesystem::path(file_in_str), basedir);
        file_out = complete(boost::filesystem::path(file_out_str), basedir);
        std::string simname;
        scheduler::load_tasks(file_in, file_out, basedir, /* check_parameter = */ false,
                              simname, tasks);
        std::clog << "  master input file  = " << file_in.native_file_string() << std::endl
                  << "  master output file = " << file_out.native_file_string() << std::endl;
        scheduler::print_taskinfo(std::clog, tasks);
        BOOST_FOREACH(task& t, tasks) t.evaluate();
      } else {
        // process one task
        task t(file);
        t.evaluate();
      }
      std::clog << scheduler::log_header() << "all tasks evaluated\n";
      return 0;
    }

    //
    // rum simulations
    //

    std::clog << log_header() << "starting scheduler on " << alps::hostname() << std::endl;

    int t = scheduler::load_filename(file, file_in_str, file_out_str);
    if (t != 1) {
      std::cerr << "invalid master file: " << file.native_file_string() << std::endl;
      process.halt();
    }
    file_in = complete(boost::filesystem::path(file_in_str), basedir);
    file_out = complete(boost::filesystem::path(file_out_str), basedir);

    master_lock.set_file(file_out);
    master_lock.lock(0);
    if (!master_lock.locked()) {
      std::cerr << "Error: master file (" << file_out.native_file_string()
                << ") is being used by other scheduler.  Skip this file.\n";
      continue;
    }

    std::clog << "  master input file  = " << file_in.native_file_string() << std::endl
              << "  master output file = " << file_out.native_file_string() << std::endl
              << "  number of process(es) = " << all_processes().size() << std::endl
              << "  process(es) per clone = " << opt.procs_per_clone << std::endl
              << "  check parameter = " << (opt.check_parameter ? "yes" : "no") << std::endl
              << "  auto evaluation = " << (opt.auto_evaluate ? "yes" : "no") << std::endl;
    if (opt.time_limit != boost::posix_time::time_duration())
      std::clog << "  time limit = " << opt.time_limit.total_seconds() << " seconds\n";
    else
      std::clog << "  time limit = none (runs forever)\n";
    std::clog << "  interval between checkpointing  = "
              << opt.checkpoint_interval.total_seconds() << " seconds\n"
              << "  minimum interval between checks = "
              << opt.min_check_interval.total_seconds() << " seconds\n"
              << "  maximum interval between checks = "
              << opt.max_check_interval.total_seconds() << " seconds\n";

    load_tasks(file_in, file_out, basedir, opt.check_parameter, simname, tasks);
    if (simname != "")
      std::clog << "  simulation name = " << simname << std::endl;
    BOOST_FOREACH(task const& t, tasks) {
      if (t.status() == task_status::NotStarted || t.status() == task_status::Suspended ||
          t.status() == task_status::Finished)
        task_queue.push(task_queue_t::value_type(t));
      if (t.status() == task_status::Finished || t.status() == task_status::Completed)
        ++num_finished_tasks;
    }
    print_taskinfo(std::clog, tasks);
    if (tasks.size() == 0) std::clog << "Warning: no tasks found\n";
    check_queue.push(next_taskinfo(opt.checkpoint_interval));

    clone* clone_ptr = 0;
    while (true) {

      // server process
      clone_proxy proxy(clone_ptr);

      if (!process.is_halting()) {
        bool to_halt = false;
        if (num_finished_tasks == tasks.size()) {
          std::clog << log_header() << "all tasks have been finished\n";
          to_halt = true;
        }
        if (opt.time_limit != boost::posix_time::time_duration() &&
            boost::posix_time::second_clock::local_time() > end_time) {
          std::clog << log_header() << "time limit reached\n";
          to_halt = true;
        }
        signal_info_t s = signals();
        if (s == signal_info::STOP || s == signal_info::TERMINATE) {
          std::clog << log_header() << "signal received\n";
          to_halt = true;
        }
        if (to_halt) {
          if (clone_ptr) {
            tid_t tid = clone_ptr->task_id();
            cid_t cid = clone_ptr->clone_id();
            std::clog << log_header() << clone_name(clone_ptr->task_id(), clone_ptr->clone_id())
                      << " suspended (" << precision(clone_ptr->info().progress() * 100, 3)
                      << "% done)" << std::endl;
            BOOST_FOREACH(task& t, tasks) t.suspend_clones(proxy);
            tasks[tid].clone_suspended(cid, clone_ptr->info());
            tasks[tid].save();
            tasks[tid].halt();
            save_tasks(file_out, simname, file_in_str, file_out_str, tasks);
            delete clone_ptr;
            clone_ptr = 0;
          }
          process.halt();
        }
      }

      while (true) {
        if (clone_ptr && clone_ptr->halted()) {
          tid_t tid = clone_ptr->task_id();
          cid_t cid = clone_ptr->clone_id();
          std::clog << log_header() << clone_name(tid, cid) << " finished" << std::endl;
          task_status_t old_status = tasks[tid].status();
          tasks[tid].info_updated(cid, clone_ptr->info());
          tasks[tid].halt_clone(proxy, cid);
          tasks[tid].clone_halted(cid);
          tasks[tid].save();
          if (old_status != task_status::Continuing && old_status != task_status::Idling &&
              (tasks[tid].status() == task_status::Continuing ||
               tasks[tid].status() == task_status::Idling))
            ++num_finished_tasks;
          tasks[tid].halt();
          save_tasks(file_out, simname, file_in_str, file_out_str, tasks);
          delete clone_ptr;
          clone_ptr = 0;
        } else if (!process.is_halting() && !clone_ptr && task_queue.size()) {
          // dispatch a new/suspended clone
          tid_t tid = task_queue.top().task_id;
          task_queue.pop();
          bool is_new;
          cid_t cid;
          boost::tie(cid, is_new) = tasks[tid].dispatch_clone(proxy);
          check_queue.push(next_checkpoint(tid, cid, 0, opt.checkpoint_interval));
          check_queue.push(next_check(tid, cid, 0,
            (is_new ? opt.min_check_interval : boost::posix_time::seconds(0))));
          if (tasks[tid].can_dispatch())
            task_queue.push(task_queue_t::value_type(tasks[tid]));
          if (is_new)
            std::clog << log_header() << "dispatching a new " << clone_name(tid, cid)
                      << std::endl;
          else
            std::clog << log_header() << "resuming a suspended " << clone_name(tid, cid)
                      << std::endl;
        } else if (!process.is_halting() && check_queue.size() && check_queue.top().due()) {
          check_queue_t::value_type q = check_queue.top();
          check_queue.pop();
          if (q.type == check_type::taskinfo) {
            print_taskinfo(std::clog, tasks);
            check_queue.push(next_taskinfo(opt.checkpoint_interval));
          } else if (q.type == check_type::checkpoint) {
            // regular checkpointing
            if (clone_ptr && tasks[q.task_id].is_running(q.clone_id)) {
              clone_info const& info = clone_ptr->info();
              std::clog << log_header() << "regular checkpoint: "
                        << clone_name(q.task_id, q.clone_id) << " is " << info.phase()
                        << " (" << precision(info.progress() * 100, 3) << "% done)\n";
              tasks[q.task_id].checkpoint(proxy, q.clone_id);
              tasks[q.task_id].info_updated(q.clone_id, info);
              tasks[q.task_id].save();
              check_queue.push(next_checkpoint(q.task_id, q.clone_id, 0,
                opt.checkpoint_interval));
            }
          } else {
            // regular progress check
            if (clone_ptr && tasks[q.task_id].is_running(q.clone_id)) {
              clone_info const& info = clone_ptr->info();
              std::clog << log_header() << "progress report: "
                        << clone_name(q.task_id, q.clone_id) << " is " << info.phase()
                        << " (" << precision(info.progress() * 100, 3) << "% done)\n";
              tasks[q.task_id].info_updated(q.clone_id, info);
              tasks[q.task_id].save();
              check_queue.push(next_check(q.task_id, q.clone_id, 0, opt.min_check_interval,
                opt.max_check_interval, info.elapsed(), info.progress()));
            }
          }
        } else {
          break;
        }
      }

      if (clone_ptr) {

        // work some
        clone_ptr->run();

      }

      // check if all processes are halted
      if (process.check_halted()) {
        print_taskinfo(std::clog, tasks);
        std::clog << log_header() << "all processes halted\n";
        if (opt.auto_evaluate) {
          std::clog << log_header() << "starting evaluation on " << alps::hostname() << std::endl;
          BOOST_FOREACH(task& t, tasks) t.evaluate();
          std::clog << scheduler::log_header() << "all tasks evaluated\n";
        }
        break;
      }
    }

    master_lock.release();
  }
  return 0;
}

} // end namespace scheduler
} // end namespace parapack
} // end namespace alps
