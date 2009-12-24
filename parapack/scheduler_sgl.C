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

#include "scheduler.h"
#include "clone.h"
#include "clone_proxy.h"
#include "filelock.h"
#include "queue.h"

#include <boost/filesystem/operations.hpp>
#include <boost/foreach.hpp>
#include <boost/regex.hpp>
#include <boost/timer.hpp>
#include <iostream>
#include <string>
#include <vector>

#if defined(ALPS_HAVE_UNISTD_H)
# include <unistd.h>
#elif defined(ALPS_HAVE_WINDOWS_H)
# include <windows.h>
#endif

#ifdef _OPENMP
# include <omp.h>
#endif

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
    p["WORKER_SEED"] = p["SEED"];
    p["DISORDER_SEED"] = p["SEED"];
    std::cout << "[input parameters]\n" << p;
    std::vector<alps::ObservableSet> obs;
    boost::shared_ptr<alps::parapack::abstract_worker>
      worker = worker_factory::make_worker(p);
    worker->init_observables(p, obs);
    bool thermalized = worker->is_thermalized();
    if (thermalized) for (int i = 0; i < obs.size(); ++i) obs[i].reset(true);
    while (worker->progress() < 1.0) {
      worker->run(obs);
      if (!thermalized && worker->is_thermalized()) {
        for (int i = 0; i < obs.size(); ++i) obs[i].reset(true);
        thermalized = true;
      }
    }
    std::vector<alps::ObservableSet> obs_out;
    boost::shared_ptr<alps::parapack::abstract_evaluator>
      evaluator = evaluator_factory::make_evaluator(p);
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
    boost::filesystem::path file_in;   // xxx.in.xml
    boost::filesystem::path file_out;  // xxx.out.xml
    boost::filesystem::path file_term; // xxx.term
    filelock master_lock;
    std::string simname;
    std::vector<task> tasks;
    task_queue_t task_queue;
    int num_finished_tasks = 0;
    int num_groups = opt.num_total_threads / opt.threads_per_clone;
#ifdef _OPENMP
    omp_set_nested(true);
#endif

    //
    // evaluation only
    //

    if (opt.evaluate_only) {
      if (is_master()) {
        std::clog << logger::header() << "starting evaluation on " << alps::hostname() << std::endl;
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
          #pragma omp parallel for
          for (int t = 0; t < tasks.size(); ++t) {
            tasks[t].evaluate();
          }
        } else {
          // process one task
          task t(file);
          t.evaluate();
        }
        std::clog << logger::header() << "all tasks evaluated\n";
      }
      return 0;
    }

    //
    // rum simulations
    //

    if (is_master()) {
      std::clog << logger::header() << "starting scheduler on " << alps::hostname() << std::endl;

      if (load_filename(file, file_in_str, file_out_str) != 1) {
        std::cerr << "invalid master file: " << file.native_file_string() << std::endl;
        process.halt();
      }
      file_in = complete(boost::filesystem::path(file_in_str), basedir);
      file_out = complete(boost::filesystem::path(file_out_str), basedir);
      file_term = complete(boost::filesystem::path(regex_replace(file_out_str,
                    boost::regex("\\.out\\.xml$"), ".term")), basedir);

      master_lock.set_file(file_out);
      master_lock.lock(0);
      if (!master_lock.locked()) {
        std::cerr << "Error: master file (" << file_out.native_file_string()
                  << ") is being used by other scheduler.  Skip this file.\n";
        continue;
      }

      std::clog << "  master input file  = " << file_in.native_file_string() << std::endl
                << "  master output file = " << file_out.native_file_string() << std::endl
                << "  termination file   = " << file_term.native_file_string() << std::endl
                << "  total number of thread(s) = " << opt.num_total_threads << std::endl
                << "  thread(s) per clone       = " << opt.threads_per_clone << std::endl
                << "  number of thread group(s) = " << num_groups << std::endl
                << "  check parameter = " << (opt.check_parameter ? "yes" : "no") << std::endl
                << "  auto evaluation = " << (opt.auto_evaluate ? "yes" : "no") << std::endl;
      if (opt.time_limit != boost::posix_time::time_duration())
        std::clog << "  time limit = " << opt.time_limit.total_seconds() << " seconds\n";
      else
        std::clog << "  time limit = unlimited\n";
      std::clog << "  interval between checkpointing  = "
                << opt.checkpoint_interval.total_seconds() << " seconds\n"
                << "  interval between progress report = "
                << opt.report_interval.total_seconds() << " seconds\n";

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
    }

    #pragma omp parallel num_threads(num_groups)
    {
      thread_group group(thread_id());
#ifdef _OPENMP
      omp_set_num_threads(opt.threads_per_clone);
#endif
      #pragma omp master
      {
        std::clog << logger::header() << "starting " << num_groups << " threadgroup(s)\n";
      }

      check_queue_t check_queue; // check_queue is theread private
      #pragma omp master
      {
        check_queue.push(next_taskinfo(opt.checkpoint_interval / 10));
      } // end omp master

      clone* clone_ptr = 0;
      clone_proxy proxy(clone_ptr, opt.check_interval);

      while (true) {

        while (true) {
          if (clone_ptr && clone_ptr->halted()) {
            tid_t tid = clone_ptr->task_id();
            cid_t cid = clone_ptr->clone_id();
            #pragma omp critical
            {
              double progress = tasks[tid].progress();
              tasks[tid].info_updated(cid, clone_ptr->info());
              tasks[tid].halt_clone(proxy, cid, group);
              if (progress < 1 && tasks[tid].progress() >= 1) ++num_finished_tasks;
              save_tasks(file_out, simname, file_in_str, file_out_str, tasks);
            } // end omp critical
          } else if (clone_ptr && process.is_halting()) {
            tid_t tid = clone_ptr->task_id();
            cid_t cid = clone_ptr->clone_id();
            #pragma omp critical
            {
              tasks[tid].suspend_clone(proxy, cid, group);
            } // end omp critical
          } else if (!process.is_halting() && !clone_ptr) {
            tid_t tid = 0;
            boost::optional<cid_t> cid;
            #pragma omp critical
            {
              if (task_queue.size()) {
                tid = task_queue.top().task_id;
                task_queue.pop();
                cid = tasks[tid].dispatch_clone(proxy, group);
                if (cid && tasks[tid].can_dispatch())
                  task_queue.push(task_queue_t::value_type(tasks[tid]));
              }
            } // end omp critical
            if (cid) {
              check_queue.push(next_checkpoint(tid, *cid, 0, opt.checkpoint_interval));
              check_queue.push(next_report(tid, *cid, 0, opt.report_interval));
            } else {
              break;
            }
          } else if (!process.is_halting() && check_queue.size() && check_queue.top().due()) {
            check_queue_t::value_type q = check_queue.top();
            check_queue.pop();
            if (q.type == check_type::taskinfo) {
              #pragma omp critical
              {
                std::clog << logger::header() << "checkpointing task files\n";
                BOOST_FOREACH(task const& t, tasks) { if (t.on_memory()) t.save(); }
                save_tasks(file_out, simname, file_in_str, file_out_str, tasks);
                print_taskinfo(std::clog, tasks);
              } // end omp critical
              check_queue.push(next_taskinfo(opt.checkpoint_interval));
            } else if (q.type == check_type::checkpoint) {
              if (tasks[q.task_id].on_memory() && tasks[q.task_id].is_running(q.clone_id)) {
                #pragma omp critical
                {
                  tasks[q.task_id].checkpoint(proxy, q.clone_id);
                } // end omp critical
                check_queue.push(next_checkpoint(q.task_id, q.clone_id, q.group_id,
                                                 opt.checkpoint_interval));
              }
            } else {
              if (tasks[q.task_id].on_memory() && tasks[q.task_id].is_running(q.clone_id)) {
                #pragma omp critical
                {
                  tasks[q.task_id].report(proxy, q.clone_id);
                } // end omp critical
                check_queue.push(next_report(q.task_id, q.clone_id, q.group_id,
                                             opt.report_interval));
              }
            }
          } else {
            break;
          }
        }

        #pragma omp master
        {
          if (!process.is_halting()) {
            bool to_halt = false;
            if (num_finished_tasks == tasks.size()) {
              std::clog << logger::header() << "all tasks have been finished\n";
              to_halt = true;
            }
            if (opt.time_limit != boost::posix_time::time_duration() &&
                boost::posix_time::second_clock::local_time() > end_time) {
              std::clog << logger::header() << "time limit reached\n";
              to_halt = true;
            }
            signal_info_t s = signals();
            if (s == signal_info::STOP || s == signal_info::TERMINATE) {
              std::clog << logger::header() << "signal received\n";
              to_halt = true;
            }
            if (exists(file_term)) {
              std::clog << logger::header() << "termination file detected\n";
              remove(file_term);
              to_halt = true;
            }
            if (to_halt) {
              #pragma omp critical
              {
                for (int t = 0; t < tasks.size(); ++t) tasks[t].suspend_remote_clones(proxy);
              }
              process.halt();
            }
          }
        } // end omp master

        if (clone_ptr) {

          // work some
          clone_ptr->run();

        } else {

          // nothing to do
          if (group.group_id == 0) {
            if (process.check_halted()) {
              break;
            } else {
              #if defined(ALPS_HAVE_UNISTD_H)
                sleep(1);    // sleep 1 Sec
              #elif defined(ALPS_HAVE_WINDOWS_H)
                Sleep(100); // sleep 100 mSec
              #endif
            }
          } else {
            break;
          }

        }
      }

      std::clog << logger::header() << logger::group(group) << " halted\n";
    } // end omp parallel

    print_taskinfo(std::clog, tasks);
    if (opt.auto_evaluate) {
      std::clog << logger::header() << "starting evaluation on "
                << alps::hostname() << std::endl;
      #pragma omp parallel for
      for (int t = 0; t < tasks.size(); ++t) {
        tasks[t].evaluate();
      } // end omp parallel for
      std::clog << logger::header() << "all tasks evaluated\n";
    }

    master_lock.release();
  }
  return 0;
}

} // end namespace scheduler
} // end namespace parapack
} // end namespace alps
