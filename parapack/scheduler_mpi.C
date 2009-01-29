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
#include "clone_mpi.h"
#include "clone_proxy_mpi.h"
#include "filelock.h"
#include "parallel_factory.h"
#include "queue.h"

#include <boost/filesystem/operations.hpp>
#include <boost/foreach.hpp>
#include <boost/timer.hpp>
#include <iostream>
#include <string>
#include <vector>

namespace mpi = boost::mpi;

namespace alps {
namespace parapack {

int run_sequential(int argc, char **argv) {

#ifndef BOOST_NO_EXCEPTIONS
  try {
#endif

  boost::throw_exception(std::runtime_error("run_sequaential: not implemented"));

//   mpi::environment env(argc, argv);
//   mpi::communicator comm;

//   alps::ParameterList parameterlist;
//   if (comm.rank() == 0) {
//     switch (argc) {
//     case 1:
//       std::cin >> parameterlist;
//       break;
//     case 2: {
//       boost::filesystem::ifstream is(argv[1]);
//       is >> parameterlist;
//       break;
//     }
//     default:
//       boost::throw_exception(std::invalid_argument(
//         "Usage: " + std::string(argv[0]) + " [paramfile]"));
//     }
//   }
//   broadcast(comm, parameterlist, 0);

//   for (int i = 0; i < parameterlist.size(); ++i) {
//     alps::Parameters p = parameterlist[i];
//     comm.barrier();
//     boost::timer tm;
//     if (!p.defined("DIR_NAME")) p["DIR_NAME"] = ".";
//     if (!p.defined("BASE_NAME")) p["BASE_NAME"] = "task" + boost::lexical_cast<std::string>(i+1);
//     if (!p.defined("SEED")) p["SEED"] = static_cast<unsigned int>(time(0));
//     p["WORKER_SEED"] = static_cast<unsigned int>(p["SEED"]) ^ (comm.rank() << 11);
//     p["DISORDER_SEED"] = p["SEED"];
//     if (comm.rank() == 0) std::cout << "[input parameters]\n" << p;
//     std::vector<alps::ObservableSet> obs;
//     boost::shared_ptr<alps::parapack::abstract_worker>
//       worker = parallel_worker_factory::make_worker(comm, p);
//     worker->init_observables(p, obs);
//     bool thermalized = worker->is_thermalized();
//     if (thermalized) BOOST_FOREACH(alps::ObservableSet& o, obs) { o.reset(true); }
//     while (worker->progress() < 1.0) {
//       worker->run(obs);
//       if (!thermalized && worker->is_thermalized()) {
//         BOOST_FOREACH(alps::ObservableSet& o, obs) { o.reset(true); }
//         thermalized = true;
//       }
//     }
//     if (comm.rank() == 0) {
//       std::vector<alps::ObservableSet> obs_slave;
//       for (int p = 1; p < comm.size(); ++p) {
//         comm.recv(p, 0, obs_slave);
//         for (int j = 0; j < obs.size(); ++j) obs[j] << obs_slave[j];
//       }
//     } else {
//       comm.send(0, 0, obs);
//     }
//     if (comm.rank() == 0) {
//       std::vector<alps::ObservableSet> obs_out;
//       boost::shared_ptr<alps::parapack::abstract_evaluator>
//         evaluator = evaluator_factory::make_evaluator(p);
//       evaluator->load(obs, obs_out);
//       evaluator->evaluate(obs_out);
//       std::cerr << "[speed]\nelapsed time = " << tm.elapsed() << " sec\n";
//       std::cout << "[results]\n";
//       if (obs_out.size() == 1) {
//         std::cout << obs_out[0];
//       } else {
//         for (int i = 0; i < obs_out.size(); ++i)
//           std::cout << "[[replica " << i << "]]\n" << obs_out[i];
//       }
//     }
//     comm.barrier();
//   }

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
  mpi::environment env(argc, argv);
  mpi::communicator world;

  option opt(argc, argv, world.size(), world.rank());
  broadcast(world, opt.valid, 0);
  if (!opt.valid) return -1;

  if (world.rank() == 0) print_copyright(std::clog);

  boost::posix_time::ptime end_time =
    boost::posix_time::second_clock::local_time() + opt.time_limit;

  BOOST_FOREACH(std::string const& file_str, opt.jobfiles) {

    process_helper_mpi process(world, opt.procs_per_clone);

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
      if (is_master()) {
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
      }
      return 0;
    }

    //
    // rum simulations
    //

    if (is_master()) {
      std::clog << log_header() << "starting scheduler on " << alps::hostname() << std::endl;

      if (load_filename(file, file_in_str, file_out_str) != 1) {
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
                << "  number of process(es)      = " << process.num_total_processes() << std::endl
                << "  process(es) per clone      = " << process.num_procs_per_group() << std::endl
                << "  number of process group(s) = " << process.num_groups() << std::endl
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
      check_queue.push(next_taskinfo(opt.checkpoint_interval / 2));
    }

    clone_mpi* clone_ptr = 0;
    while (true) {

      // server process
      if (is_master()) {

        clone_proxy_mpi proxy(process.comm_down());

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
            BOOST_FOREACH(task& t, tasks) t.suspend_clones(proxy);
            process.halt();
          }
        }

        while (true) {
          boost::optional<mpi::status> status
            = process.comm_up().iprobe(mpi::any_source, mpi::any_tag);
          if (status) {
            if (status->tag() == mcmp_tag::clone_info
                || status->tag() == mcmp_tag::clone_checkpoint) {
              clone_info_msg_t msg;
              process.comm_up().recv(mpi::any_source, status->tag(), msg);
              if (tasks[msg.task_id].on_memory() && tasks[msg.task_id].is_running(msg.clone_id)) {
                tasks[msg.task_id].info_updated(msg.clone_id, msg.info);
                if (msg.info.progress() < 1) {
                  // tasks[msg.task_id].save();
                  if (status->tag() == mcmp_tag::clone_checkpoint) {
                    check_queue.push(next_checkpoint(msg.task_id, msg.clone_id, msg.group_id,
                      opt.checkpoint_interval));
                    std::clog << log_header() << "regular checkpoint: "
                              << clone_name(msg.task_id, msg.clone_id) << " is " << msg.info.phase()
                              << " (" << precision(msg.info.progress() * 100, 3) << "% done)\n";
                  } else  {
                    check_queue.push(next_check(msg.task_id, msg.clone_id, msg.group_id, opt.min_check_interval,
                      opt.max_check_interval, msg.info.elapsed(), msg.info.progress()));
                    std::clog << log_header() << "progress report: "
                              << clone_name(msg.task_id, msg.clone_id) << " is " << msg.info.phase()
                              << " (" << precision(msg.info.progress() * 100, 3) << "% done)\n";
                  }
                } else {
                  tasks[msg.task_id].halt_clone(proxy, msg.clone_id);
                  std::clog << log_header() << clone_name(msg.task_id, msg.clone_id) << " finished"
                            << " on " << pg_name(msg.group_id) << std::endl;
                }
              }
            } else if (status->tag() == mcmp_tag::clone_suspend) {
              clone_info_msg_t msg;
              process.comm_up().recv(mpi::any_source, status->tag(), msg);
              tasks[msg.task_id].clone_suspended(msg.clone_id, msg.info);
              // tasks[msg.task_id].save();
              if (tasks[msg.task_id].num_running() == 0) {
                tasks[msg.task_id].save();
                tasks[msg.task_id].halt();
                save_tasks(file_out, simname, file_in_str, file_out_str, tasks);
              }
              process.release(msg.group_id);
              std::clog << log_header() << clone_name(msg.task_id, msg.clone_id)
                        << " suspended (" << precision(msg.info.progress() * 100, 3) << "% done)"
                        << " on " << pg_name(msg.group_id) << std::endl;
            } else if (status->tag() == mcmp_tag::clone_halt) {
              clone_halt_msg_t msg;
              process.comm_up().recv(mpi::any_source, status->tag(), msg);
              task_status_t old_status = tasks[msg.task_id].status();
              tasks[msg.task_id].clone_halted(msg.clone_id);
              // tasks[msg.task_id].save();
              if (old_status != task_status::Continuing && old_status != task_status::Idling &&
                  (tasks[msg.task_id].status() == task_status::Continuing ||
                   tasks[msg.task_id].status() == task_status::Idling))
                ++num_finished_tasks;
              if (tasks[msg.task_id].num_running() == 0) {
                tasks[msg.task_id].save();
                tasks[msg.task_id].halt();
                save_tasks(file_out, simname, file_in_str, file_out_str, tasks);
              }
              process.release(msg.group_id);
            } else if (status->tag() == mcmp_tag::scheduler_halt) {
              //// 2008-11-14 ST workaround for bug in process_mpi (or boost.MPI ?)
              break;
            } else {
              std::clog << "Warning: ignoring a message with an unknown tag " << status->tag()
                        << std::endl;
            }
          } else if (!process.is_halting() && process.num_free() && task_queue.size()) {
            // dispatch a new/suspended clone
            process_group g = process.allocate();
            tid_t tid = task_queue.top().task_id;
            task_queue.pop();
            bool is_new;
            cid_t cid;
            boost::tie(cid, is_new) = tasks[tid].dispatch_clone(proxy, g);
            check_queue.push(next_checkpoint(tid, cid, g.group_id, opt.checkpoint_interval));
            check_queue.push(next_check(tid, cid, g.group_id, opt.min_check_interval));
            if (tasks[tid].can_dispatch())
              task_queue.push(task_queue_t::value_type(tasks[tid]));
            if (is_new)
              std::clog << log_header() << "dispatching a new " << clone_name(tid, cid)
                        << " on " << pg_name(g.group_id) << std::endl;
            else
              std::clog << log_header() << "resuming a suspended " << clone_name(tid, cid)
                        << " on " << pg_name(g.group_id) << std::endl;
          } else if (!process.is_halting() && check_queue.size() && check_queue.top().due()) {
            check_queue_t::value_type q = check_queue.top();
            check_queue.pop();
            if (q.type == check_type::taskinfo) {
              print_taskinfo(std::clog, tasks);
              BOOST_FOREACH(task& t, tasks) {
                if (t.status() == task_status::Running || t.status() == task_status::Continuing)
                  t.save();
              }
              save_tasks(file_out, simname, file_in_str, file_out_str, tasks);
              check_queue.push(next_taskinfo(opt.checkpoint_interval));
            } else if (q.type == check_type::checkpoint) {
              // regular checkpoint
              if (tasks[q.task_id].on_memory() && tasks[q.task_id].is_running(q.clone_id))
                tasks[q.task_id].checkpoint(proxy, q.clone_id);
            } else {
              // regular progress check
              if (tasks[q.task_id].on_memory() && tasks[q.task_id].is_running(q.clone_id))
                tasks[q.task_id].update_info(proxy, q.clone_id);
            }
          } else {
            break;
          }
        }
      }

      if (clone_ptr) {

        // work some
        clone_ptr->run();
        if (clone_ptr->halted()) {
          delete clone_ptr;
          clone_ptr = 0;
        }

      } else {

        // create a clone
        if (process.comm_down().iprobe(0, mcmp_tag::clone_create)) {
          clone_create_msg_t msg;
          process.comm_down().recv(0, mcmp_tag::clone_create, msg);
          clone_ptr = new clone_mpi(process.comm_down(), process.comm_up(), process.comm_work(),
            msg);
        }

      }

      // check if all processes are halted
      if (process.check_halted()) {
        if (is_master()) {
          print_taskinfo(std::clog, tasks);
          std::clog << log_header() << "all processes halted\n";
          if (opt.auto_evaluate) {
            std::clog << log_header() << "starting evaluation on " << alps::hostname() << std::endl;
            BOOST_FOREACH(task& t, tasks) t.evaluate();
            std::clog << scheduler::log_header() << "all tasks evaluated\n";
          }
        }
        break;
      }
    }

    if (is_master()) master_lock.release();
  }
  return 0;
}

} // namespace scheduler
} // namespace parapack
} // namespace alps
