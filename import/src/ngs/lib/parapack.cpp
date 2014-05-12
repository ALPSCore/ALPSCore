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

#include <alps/ngs/parapack/parapack.h>
#include <alps/ngs/parapack/clone.h>
#include <alps/ngs/parapack/clone_proxy.h>
#include <alps/ngs/parapack/job_p.h>
#include <alps/parapack/filelock.h>
#include <alps/parapack/logger.h>
#include <alps/parapack/queue.h>
#include <alps/parapack/staging.h>
#include <alps/parapack/version.h>

#include <alps/config.h>
#include <alps/stop_callback.hpp>
#include <alps/utility/copyright.hpp>
#include <alps/utility/os.hpp>
#include <alps/osiris/comm.h>

// some file (probably a python header) defines a tolower macro ...
#undef tolower
#undef toupper

#include <boost/config.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/foreach.hpp>
#include <boost/regex.hpp>
#include <boost/timer.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <time.h>

#if defined(ALPS_HAVE_UNISTD_H)
# include <unistd.h>
#elif defined(ALPS_HAVE_WINDOWS_H)
# include <windows.h>
#endif

#ifdef _OPENMP
# include <omp.h>
#endif

namespace alps {

namespace ngs_parapack {

int start_impl(int argc, char **argv) {
  #ifndef BOOST_NO_EXCEPTIONS
  try {
  #endif

    alps::parapack::option opt(argc, argv);
    if (!opt.valid) {
      std::cerr << "Error: unknown command line option(s)\n";
      opt.print(std::cerr);
      return -1;
    }
    int ret;
    if (opt.jobfiles.size() == 0) {
      if (!opt.use_mpi) {
        if (opt.show_help) {
          opt.print(std::cout);
          return 0;
        }
        if (opt.show_license) {
          print_copyright(std::cout);
          print_license(std::cout);
          return 0;
        }
        ret = run_sequential(argc, argv);
      } else {
#ifdef ALPS_HAVE_MPI
        if (opt.show_help || opt.show_license) {
          boost::mpi::environment env(argc, argv);
          boost::mpi::communicator world;
          if (world.rank() == 0) {
            if (opt.show_help) {
              opt.print(std::cout);
            } else {
              print_copyright(std::cout);
              print_license(std::cout);
            }
          }
          return 0;
        }
        ret = run_sequential_mpi(argc, argv);
#else
        std::cerr << "ERROR: MPI is not supported\n";
        return -1;
#endif
      }
    } else {
      if (!opt.use_mpi) {
        ret = start_sgl(argc, argv);
      } else {
#ifdef ALPS_HAVE_MPI
        ret = start_mpi(argc, argv);
#else
        std::cerr << "ERROR: MPI is not supported\n";
        return -1;
#endif
      }
    }
    return ret;

  #ifndef BOOST_NO_EXCEPTIONS
  }
  catch (const std::exception& excp) {
    std::cerr << excp.what() << std::endl;
  }
  catch (...) {
    std::cerr << "Unknown exception occurred!" << std::endl;
  }
  return -1;
  #endif
}

// int evaluate(int argc, char **argv) {
//   #ifndef BOOST_NO_EXCEPTIONS
//   try {
//   #endif

//     evaluate_option opt(argc, argv);
//     if (!opt.valid) {
//       std::cerr << "Error: unknown command line option(s)\n";
//       opt.print(std::cerr);
//       return -1;
//     }
//     if (opt.show_help) {
//       opt.print(std::cout);
//       return 0;
//     }
//     if (opt.show_license) {
//       print_copyright(std::cout);
//       print_license(std::cout);
//       return 0;
//     }

//     BOOST_FOREACH(std::string const& file_str, opt.jobfiles) {
//       boost::filesystem::path file = complete(boost::filesystem::path(file_str)).normalize();
//       if (!exists(file)) {
//         std::cerr << "Error: file not found: " << file << std::endl;
//         return -1;
//       }
//       boost::filesystem::path basedir = file.branch_path();
//       std::string file_in_str;
//       std::string file_out_str;
//       std::vector<task> tasks;

//       std::cout << logger::header() << "starting evaluation on "
//                 << alps::hostname() << std::endl;
//       int t = load_filename(file, file_in_str, file_out_str);
//       if (t == 1) {
//         // process all tasks
//         boost::filesystem::path file_in = complete(boost::filesystem::path(file_in_str), basedir);
//         boost::filesystem::path file_out = complete(boost::filesystem::path(file_out_str), basedir);
//         std::string simname;
//         load_tasks(file_in, file_out, basedir, simname, tasks, false, opt.write_xml);
//         std::cout << "  master input file  = " << file_in.string() << std::endl
//                   << "  master output file = " << file_out.string() << std::endl;
//         print_taskinfo(std::cout, tasks, opt.task_range);
//         BOOST_FOREACH(task& t, tasks) {
//           if (!opt.task_range.valid() || opt.task_range.is_included(t.task_id()+1))
//             t.evaluate(opt.write_xml);
//         }
//       } else {
//         // process one task
//         task t(file);
//         if (!opt.task_range.valid() || opt.task_range.is_included(t.task_id()+1))
//           t.evaluate(opt.write_xml);
//       }
//       std::cout << logger::header() << "all tasks evaluated\n";
//     }
//     return 0;

//   #ifndef BOOST_NO_EXCEPTIONS
//   }
//   catch (const std::exception& excp) {
//     std::cerr << excp.what() << std::endl;
//   }
//   catch (...) {
//     std::cerr << "known exception occurred!" << std::endl;
//   }
//   return -1;
//   #endif
// }

void print_copyright(std::ostream& os) {
  // worker_factory::print_copyright(os);
  os << std::endl << "using " << parapack_copyright() << std::endl;
  alps::print_copyright(os);
}

void print_license(std::ostream& os) {
  os << "Please look at the file LICENSE for the license conditions.\n";
}

std::string alps_version() {
  return version_string() + "; configured on " + config_host() +
    " by " + config_user() + "; compiled on " + compile_date();
}

void print_taskinfo(std::ostream& os, std::vector<alps::ngs_parapack::task> const& tasks,
  task_range_t const& task_range) {
  uint32_t num_new = 0;
  uint32_t num_running = 0;
  uint32_t num_continuing = 0;
  uint32_t num_suspended = 0;
  uint32_t num_finished = 0;
  uint32_t num_completed = 0;
  uint32_t num_skipped = 0;
  BOOST_FOREACH(alps::ngs_parapack::task const& t, tasks) {
    if (!task_range.valid() || task_range.is_included(t.task_id()+1)) {
      switch (t.status()) {
      case alps::task_status::NotStarted :
        ++num_new;
        break;
      case alps::task_status::Running :
        ++num_running;
        break;
      case alps::task_status::Continuing :
        ++num_continuing;
        break;
      case alps::task_status::Suspended :
        ++num_suspended;
        break;
      case alps::task_status::Finished :
        ++num_finished;
        break;
      case alps::task_status::Completed :
        ++num_completed;
        break;
      default :
        break;
      }
    } else {
      ++num_skipped;
    }
  }
  os << logger::header() << "task status: "
     << "total number of tasks = " << tasks.size() << std::endl
     << "  new = " << num_new
     << ", running = " << num_running
     << ", continuing = " << num_continuing
     << ", suspended = " << num_suspended
     << ", finished = " << num_finished
     << ", completed = " << num_completed
     << ", skipped = " << num_skipped << std::endl;
}

int load_filename(boost::filesystem::path const& file, std::string& file_in_str,
  std::string& file_out_str) {
  bool is_master;
  alps::ngs_parapack::filename_xml_handler handler(file_in_str, file_out_str, is_master);
  alps::XMLParser parser(handler);
  parser.parse(file);
  if (is_master) {
    if (file_out_str.empty())
      file_out_str = file.filename().string();
    if (file_in_str.empty())
      file_in_str = regex_replace(file_out_str, boost::regex("\\.out\\.xml$"), ".in.xml");
  }
  return is_master ? 1 : 2;
}

void load_version(boost::filesystem::path const& file,
  std::vector<std::pair<std::string, std::string> >& versions) {
  alps::ngs_parapack::version_xml_handler handler(versions);
  alps::XMLParser parser(handler);
  parser.parse(file);
}

void load_tasks(boost::filesystem::path const& file_in, boost::filesystem::path const& file_out,
  boost::filesystem::path const& basedir, std::string& simname, std::vector<alps::ngs_parapack::task>& tasks,
  bool check_parameter, bool write_xml) {
  tasks.clear();
  if (!exists(file_out)) {
    alps::ngs_parapack::job_tasks_xml_handler handler(simname, tasks, basedir);
    alps::XMLParser parser(handler);
    parser.parse(file_in);
  } else {
    alps::ngs_parapack::job_tasks_xml_handler handler_out(simname, tasks, basedir);
    alps::XMLParser parser_out(handler_out);
    parser_out.parse(file_out);

    if (check_parameter) {
      std::vector<alps::ngs_parapack::task> tasks_in;
      alps::ngs_parapack::job_tasks_xml_handler handler_in(simname, tasks_in, basedir);
      alps::XMLParser parser_in(handler_in);
      parser_in.parse(file_in);

      int nc = std::min BOOST_PREVENT_MACRO_SUBSTITUTION (tasks_in.size(), tasks.size());
      for (int i = 0; i < nc; ++i) {
        if (tasks_in[i].file_in_str() != tasks[i].file_in_str() ||
            tasks_in[i].file_out_str() != tasks[i].file_out_str()) {
          std::cout << "Info: input/output XML filename of " << logger::task(i)
                    << " has been modified" << std::endl;
          tasks[i] = tasks_in[i];
        }
        tasks[i].check_parameter(write_xml);
      }
      if (tasks_in.size() > tasks.size()) {
        std::cout << "Info: number of parameter sets has been increased from " << tasks.size()
                  << " to " << tasks_in.size() << std::endl;
        for (int i = tasks.size(); i < tasks_in.size(); ++i) tasks.push_back(tasks_in[i]);
      } else if (tasks_in.size() < tasks.size()) {
        std::cout << "Info: number of parameter sets has been decreased from " << tasks.size()
                  << " to " << tasks_in.size() << std::endl;
        tasks.resize(tasks_in.size());
      }
    }
  }
}


void save_tasks(boost::filesystem::path const& file, std::string const& simname,
  std::string const& file_in_str, std::string const& file_out_str,
  std::vector<alps::ngs_parapack::task>& tasks) {
  alps::ngs_parapack::job_xml_writer(file, simname, file_in_str, file_out_str, alps_version(),
    "unknown", tasks, true);
}

int run_sequential(int argc, char **argv) {

#ifndef BOOST_NO_EXCEPTIONS
  try {
#endif

  alps::ParameterList parameterlist;
  std::cin >> parameterlist;

#ifdef _OPENMP
  // set default number of threads to 1
  char* nt = getenv("OMP_NUM_THREADS");
  if (nt == 0 && omp_get_max_threads() != 1) omp_set_num_threads(1);
#endif

  for (std::size_t i = 0; i < parameterlist.size(); ++i) {
    alps::params p;
    for (alps::Parameters::const_iterator itr = parameterlist[i].begin(); itr != parameterlist[i].end(); ++itr) {
      p[itr->key()] = itr->value();
    }
    boost::timer tm;
    if (!p.defined("DIR_NAME")) p["DIR_NAME"] = ".";
    if (!p.defined("BASE_NAME")) p["BASE_NAME"] = "task" + boost::lexical_cast<std::string>(i+1);
    if (!p.defined("SEED")) p["SEED"] = static_cast<unsigned int>(time(0));
    p["WORKER_SEED"] = p["SEED"];
    p["DISORDER_SEED"] = p["SEED"];
    std::cout << "[input parameters]\n" << p << std::flush;
    boost::shared_ptr<alps::ngs_parapack::abstract_worker>
      worker = alps::ngs_parapack::worker_factory::make_worker(p);
    while (worker->fraction_completed() < 1.0) {
      worker->run(stop_callback(0), boost::function<void (double)>());
    }
    std::cerr << "[speed]\nelapsed time = " << tm.elapsed() << " sec\n";
    // std::cout << "[results]\n" << collect_results(*worker);
    std::cout << std::flush;
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

int start_sgl(int argc, char** argv) {
  alps::parapack::option opt(argc, argv);
  if (!opt.valid) {
    std::cerr << "Error: unknown command line option(s)\n";
    opt.print(std::cerr);
    return -1;
  }
  if (opt.show_help) {
    opt.print(std::cout);
    return 0;
  }
  if (opt.show_license) {
    print_copyright(std::cout);
    print_license(std::cout);
    return 0;
  }
  print_copyright(std::cout);

  boost::posix_time::ptime end_time =
    boost::posix_time::second_clock::local_time() + opt.time_limit;
  int num_total_threads =
    (opt.default_total_threads || opt.auto_total_threads) ? max_threads() : opt.num_total_threads;

  BOOST_FOREACH(std::string const& file_str, opt.jobfiles) {
    process_helper process;
    boost::filesystem::path file = complete(boost::filesystem::path(file_str)).normalize();
    if (!exists(file)) {
      std::cerr << "Error: file not found: " << file << std::endl;
      return -1;
    }
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
    int num_groups = num_total_threads / opt.threads_per_clone;
    if (num_groups < 1) {
      boost::throw_exception(std::runtime_error("Invalid number of threads"));
      return -1;
    }
#if defined(_OPENMP) && defined(ALPS_ENABLE_OPENMP_WORKER)
    omp_set_nested(true);
#else
    if (opt.threads_per_clone > 1) {
      std::cerr << "OpenMP worker parallelization is not supported.  Please rebuild ALPS with -DALPS_PARAPACK_ENABLE_OPENMP_WORKER=ON.\n";
      boost::throw_exception(std::runtime_error("OpenMP worker parallelization is not supported"));
      return -1;
    }
#endif

    //
    // evaluation only
    //

    // if (opt.evaluate_only) {
    //   std::cout << logger::header() << "starting evaluation on " << alps::hostname() << std::endl;
    //   int t = load_filename(file, file_in_str, file_out_str);
    //   if (t == 1) {
    //     file_in = complete(boost::filesystem::path(file_in_str), basedir);
    //     file_out = complete(boost::filesystem::path(file_out_str), basedir);
    //     std::string simname;
    //     load_tasks(file_in, file_out, basedir, simname, tasks, false, opt.write_xml);
    //     std::cout << "  master input file  = " << file_in.string() << std::endl
    //               << "  master output file = " << file_out.string() << std::endl;
    //     print_taskinfo(std::cout, tasks, opt.task_range);
    //     // #pragma omp parallel for
    //     for (int t = 0; t < tasks.size(); ++t) {
    //       if (!opt.task_range.valid() || opt.task_range.is_included(t+1))
    //         tasks[t].evaluate(opt.write_xml);
    //     }
    //   } else {
    //     // process one task
    //     task t(file);
    //     if (!opt.task_range.valid() || opt.task_range.is_included(t.task_id()+1))
    //       t.evaluate(opt.write_xml);
    //   }
    //   std::cout << logger::header() << "all tasks evaluated\n";
    //   return 0;
    // }

    //
    // rum simulations
    //

    std::cout << logger::header() << "starting scheduler on " << alps::hostname() << std::endl;

    if (load_filename(file, file_in_str, file_out_str) != 1) {
      std::cerr << "invalid master file: " << file.string() << std::endl;
      process.halt();
    }
    file_in = complete(boost::filesystem::path(file_in_str), basedir);
    file_out = complete(boost::filesystem::path(file_out_str), basedir);
    file_term = complete(boost::filesystem::path(regex_replace(file_out_str,
                  boost::regex("\\.out\\.xml$"), ".term")), basedir);

    master_lock.set_file(file_out);
    master_lock.lock(0);
    if (!master_lock.locked()) {
      std::cerr << "Error: master file (" << file_out.string()
                << ") is being used by other scheduler.  Skip this file.\n";
      continue;
    }

    std::cout << "  master input file  = " << file_in.string() << std::endl
              << "  master output file = " << file_out.string() << std::endl
              << "  termination file   = "
              << (opt.use_termfile ? file_term.string() : "[disabled]") << std::endl
              << "  total number of thread(s) = " << num_total_threads << std::endl
              << "  thread(s) per clone       = " << opt.threads_per_clone << std::endl
              << "  number of thread group(s) = " << num_groups << std::endl;
    opt.print_summary(std::cout, "  ");
    load_tasks(file_in, file_out, basedir, simname, tasks, true, opt.write_xml);
    if (simname != "")
      std::cout << "  simulation name = " << simname << std::endl;
    BOOST_FOREACH(task const& t, tasks) {
      if ((t.status() == task_status::NotStarted || t.status() == task_status::Suspended ||
           t.status() == task_status::Finished) &&
          (!opt.task_range.valid() || opt.task_range.is_included(t.task_id()+1))) {
        task_queue.push(task_queue_t::value_type(t.task_id(), t.weight()));
      } else {
        ++num_finished_tasks;
      }
    }
    print_taskinfo(std::cout, tasks, opt.task_range);
    if (tasks.size() == 0) std::cout << "Warning: no tasks found\n";

    #pragma omp parallel num_threads(num_groups)
    {
      thread_group group(thread_id());
#if defined(_OPENMP) && defined(ALPS_ENABLE_OPENMP_WORKER)
      if (omp_get_max_threads() != opt.threads_per_clone)
        omp_set_num_threads(opt.threads_per_clone);
#endif
      #pragma omp master
      {
        std::cout << logger::header() << "starting " << num_groups << " threadgroup(s)\n";
      }

      check_queue_t check_queue; // check_queue is thread private
      #pragma omp master
      {
        check_queue.push(next_taskinfo(opt.checkpoint_interval / 10));
      } // end omp master

      clone* clone_ptr = 0;
      clone_proxy proxy(clone_ptr, basedir, opt.dump_policy, opt.check_interval);

      while (true) {

        while (true) {
          if (clone_ptr && clone_ptr->halted()) {
            tid_t tid = clone_ptr->task_id();
            cid_t cid = clone_ptr->clone_id();
            #pragma omp critical
            {
              double progress = tasks[tid].progress();
              tasks[tid].info_updated(cid, clone_ptr->info());
              tasks[tid].halt_clone(proxy, opt.write_xml, cid, group);
              if (progress < 1 && tasks[tid].progress() >= 1) ++num_finished_tasks;
              save_tasks(file_out, simname, file_in_str, file_out_str, tasks);
            } // end omp critical
          } else if (clone_ptr && process.is_halting()) {
            tid_t tid = clone_ptr->task_id();
            cid_t cid = clone_ptr->clone_id();
            #pragma omp critical
            {
              tasks[tid].suspend_clone(proxy, opt.write_xml, cid, group);
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
                  task_queue.push(task_queue_t::value_type(tid, tasks[tid].weight()));
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
                std::cout << logger::header() << "checkpointing task files\n";
                for (int t = 0; t < tasks.size(); ++t) {
                  if (tasks[t].on_memory()) tasks[t].save(opt.write_xml);
                }
                std::cerr << "save task\n";
                save_tasks(file_out, simname, file_in_str, file_out_str, tasks);
                print_taskinfo(std::cout, tasks, opt.task_range);
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
              std::cout << logger::header() << "all tasks have been finished\n";
              to_halt = true;
            }
            if (boost::posix_time::second_clock::local_time() > end_time) {
              std::cout << logger::header() << "time limit reached\n";
              to_halt = true;
            }
            signal_info_t s = signals();
            if (s == signal_info::STOP || s == signal_info::TERMINATE) {
              std::cout << logger::header() << "signal received\n";
              to_halt = true;
            }
            if (opt.use_termfile && exists(file_term)) {
              std::cout << logger::header() << "termination file detected\n";
              remove(file_term);
              to_halt = true;
            }
            if (to_halt) {
              #pragma omp critical
              {
                for (int t = 0; t < tasks.size(); ++t)
                  tasks[t].suspend_remote_clones(proxy, opt.write_xml);
              }
              process.halt();
            }
          }
        } // end omp master

        if (clone_ptr) {

          // work some
          clone_ptr->run(stop_callback(0), boost::function<void (double)>());

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
    } // end omp parallel

    print_taskinfo(std::cout, tasks, opt.task_range);
    std::cout << logger::header() << "all threads halted\n";
    // if (opt.auto_evaluate) {
    //   std::cout << logger::header() << "starting evaluation on "
    //             << alps::hostname() << std::endl;
    //   // #pragma omp parallel for
    //   for (int t = 0; t < tasks.size(); ++t) {
    //     if (!opt.task_range.valid() || opt.task_range.is_included(t+1))
    //       tasks[t].evaluate(opt.write_xml);
    //   } // end omp parallel for
    //   std::cout << logger::header() << "all tasks evaluated\n";
    // }

    master_lock.release();
  }
  return 0;
}

#ifdef ALPS_HAVE_MPI

int run_sequential_mpi(int argc, char** argv) {

#ifndef BOOST_NO_EXCEPTIONS
  try {
#endif

#ifdef _OPENMP
  // set default number of threads to 1
  char* p = getenv("OMP_NUM_THREADS");
  if (p == 0 && omp_get_max_threads() != 1) omp_set_num_threads(1);
#endif

  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
  alps::ParameterList parameterlist;
  if (world.rank() == 0) std::cin >> parameterlist;
  broadcast(world, parameterlist, 0);

  for (int i = 0; i < parameterlist.size(); ++i) {
    alps::params p;
    for (alps::Parameters::const_iterator itr = parameterlist[i].begin(); itr != parameterlist[i].end(); ++itr) {
      p[itr->key()] = itr->value();
    }
    world.barrier();
    boost::timer tm;
    if (!p.defined("DIR_NAME")) p["DIR_NAME"] = ".";
    if (!p.defined("BASE_NAME")) p["BASE_NAME"] = "task" + boost::lexical_cast<std::string>(i+1);
    if (!p.defined("SEED")) p["SEED"] = static_cast<unsigned int>(time(0));
    p["WORKER_SEED"] = static_cast<unsigned int>(p["SEED"]) ^ (world.rank() << 11);
    p["DISORDER_SEED"] = p["SEED"];
    if (world.rank() == 0) std::cout << "[input parameters]\n" << p << std::flush;
    boost::shared_ptr<alps::ngs_parapack::abstract_worker>
      worker = parallel_worker_factory::make_worker(world, p);
    while (worker->fraction_completed() < 1.0) {
      worker->run(stop_callback(0), boost::function<void (double)>());
    }
    world.barrier();
    if (world.rank() == 0) {
      std::cerr << "[speed]\nelapsed time = " << tm.elapsed() << " sec" << std::endl;
    }
    for (int r = 0; r < world.size(); ++r) {
      if (world.rank() == r) {
        // std::cout << "[results " << r << "]\n" << collect_results(*worker);
        std::cout << std::flush;
      }
      world.barrier();
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

#else // ALPS_HAVE_MPI

int run_sequential_mpi(int argc, char** argv) {
  std::cerr << "This program has not been compiled for use with MPI\n";
  return -1;
}

#endif // ALPS_HAVE_MPI

#ifdef ALPS_HAVE_MPI

int start_mpi(int argc, char** argv) {
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
  alps::parapack::option opt(argc, argv);
  if (!opt.valid) {
    if (world.rank() == 0) {
      std::cerr << "Error: unknown command line option(s)\n";
      opt.print(std::cerr);
    }
    return -1;
  }
  if (opt.show_help) {
    if (world.rank() == 0) opt.print(std::cout);
    return 0;
  }
  if (opt.show_license) {
    if (world.rank() == 0) {
      print_copyright(std::cout);
      print_license(std::cout);
    }
    return 0;
  }
  if (world.rank() == 0) print_copyright(std::cout);

  boost::posix_time::ptime end_time =
    boost::posix_time::second_clock::local_time() + opt.time_limit;
  int num_total_threads;
  if (opt.default_total_threads)
    num_total_threads = world.size();
  else if (opt.auto_total_threads)
    num_total_threads = max_threads() * world.size();
  else
    num_total_threads = opt.num_total_threads;

  BOOST_FOREACH(std::string const& file_str, opt.jobfiles) {
    process_helper_mpi
      process(world, world.size() * opt.threads_per_clone / num_total_threads);
    boost::filesystem::path file = complete(boost::filesystem::path(file_str)).normalize();
    if (!exists(file)) {
      if (world.rank() == 0)
        std::cerr << "Error: file not found: " << file << std::endl;
      return -1;
    }
    boost::filesystem::path basedir = file.branch_path();

    // only for master scheduler
    signal_handler signals;
    std::string file_in_str;
    std::string file_out_str;
    std::string file_chp_str;
    boost::filesystem::path file_in;   // xxx.in.xml
    boost::filesystem::path file_out;  // xxx.out.xml
    boost::filesystem::path file_term; // xxx.term
    boost::filesystem::path file_chp; // xxx.checkpoints
    filelock master_lock;
    std::string simname;
    std::vector<task> tasks;
    task_queue_t task_queue;
    check_queue_t check_queue;
    int num_finished_tasks = 0;

    std::queue<alps::parapack::suspended_queue_t> suspended_queue;
    //
    // evaluation only
    //

    // if (opt.evaluate_only) {
    //   if (world.rank() == 0) {
    //     std::cout << logger::header() << "starting evaluation on " << alps::hostname() << std::endl;
    //     int t = load_filename(file, file_in_str, file_out_str);
    //     if (t == 1) {
    //       file_in = complete(boost::filesystem::path(file_in_str), basedir);
    //       file_out = complete(boost::filesystem::path(file_out_str), basedir);
    //       std::string simname;
    //       load_tasks(file_in, file_out, basedir, simname, tasks, true, opt.write_xml);
    //       std::cout << "  master input file  = " << file_in.string() << std::endl
    //                 << "  master output file = " << file_out.string() << std::endl;
    //       print_taskinfo(std::cout, tasks, opt.task_range);
    //       BOOST_FOREACH(task& t, tasks) {
    //         if (!opt.task_range.valid() || opt.task_range.is_included(t.task_id()+1))
    //           t.evaluate(opt.write_xml);
    //       }
    //     } else {
    //       // process one task
    //       task t(file);
    //       if (!opt.task_range.valid() || opt.task_range.is_included(t.task_id()+1))
    //         t.evaluate(opt.write_xml);
    //     }
    //     std::cout << logger::header() << "all tasks evaluated\n";
    //   }
    //   return 0;
    // }

    //
    // rum simulations
    //

    if (world.rank() == 0) {
      std::cout << logger::header() << "starting scheduler on " << alps::hostname() << std::endl;

      if (load_filename(file, file_in_str, file_out_str) != 1) {
        std::cerr << "invalid master file: " << file.string() << std::endl;
        process.halt();
      }
      file_in = complete(boost::filesystem::path(file_in_str), basedir);
      file_out = complete(boost::filesystem::path(file_out_str), basedir);
      file_term = complete(boost::filesystem::path(regex_replace(file_out_str,
                    boost::regex("\\.out\\.xml$"), ".term")), basedir);
      file_chp = complete(boost::filesystem::path(regex_replace(file_out_str,
                    boost::regex("\\.out\\.xml$"), ".checkpoints")), basedir);

      master_lock.set_file(file_out);
      master_lock.lock(0);
      if (!master_lock.locked()) {
        std::cerr << "Error: master file (" << file_out.string()
                  << ") is being used by other scheduler.  Skip this file.\n";
        continue;
      }

      std::cout << "  master input file  = " << file_in.string() << std::endl
                << "  master output file = " << file_out.string() << std::endl
                << "  termination file   = "
                << (opt.use_termfile ? file_term.string() : "[disabled]") << std::endl
                << "  total number of process(es)/thread(s) = "
                << process.num_total_processes() << "/" << num_total_threads << std::endl
                << "  process(es)/thread(s) per clone       = "
                << process.num_procs_per_group() << "/" << opt.threads_per_clone << std::endl
                << "  number of process group(s)            = "
                << process.num_groups() << std::endl;
      opt.print_summary(std::cout, "  ");
      load_tasks(file_in, file_out, basedir, simname, tasks, true, opt.write_xml);
      alps::parapack::load_checkpoints(file_chp, basedir, suspended_queue);
      if (simname != "")
        std::cout << "  simulation name = " << simname << std::endl;
      BOOST_FOREACH(task const& t, tasks) {
        if ((t.status() == task_status::NotStarted || t.status() == task_status::Suspended ||
             t.status() == task_status::Finished) &&
            (!opt.task_range.valid() || opt.task_range.is_included(t.task_id()+1))) {
          task_queue.push(task_queue_t::value_type(t.task_id(), t.weight()));
        } else {
          ++num_finished_tasks;
        }
      }
      print_taskinfo(std::cout, tasks, opt.task_range);
      if (tasks.size() == 0) std::cout << "Warning: no tasks found\n";
      check_queue.push(next_taskinfo(opt.checkpoint_interval / 10));
    }

#if defined(_OPENMP) && defined(__APPLE_CC__) && __GNUC__ == 4 && __GNUC_MINOR__ == 2
    // g++ on Mac OS X Snow Leopard requires the following OpenMP directive
    omp_set_nested(true);
    #pragma omp parallel num_threads(1)
#endif
    {
#ifdef _OPENMP
      int p = num_total_threads / process.num_total_processes();
      if (omp_get_max_threads() != p) omp_set_num_threads(p);
#endif

      clone_mpi* clone_ptr = 0;
      while (true) {

        // server process
        if (world.rank() == 0) {
          clone_proxy_mpi proxy(clone_ptr, process.comm_ctrl(), process.comm_work(), basedir,
            opt.dump_policy, opt.check_interval);

          if (!process.is_halting()) {
            bool to_halt = false;
            if (num_finished_tasks == tasks.size()) {
              std::cout << logger::header() << "all tasks have been finished\n";
              to_halt = true;
            }
            if (boost::posix_time::second_clock::local_time() > end_time) {
              std::cout << logger::header() << "time limit reached\n";
              to_halt = true;
            }
            signal_info_t s = signals();
            if (s == signal_info::STOP || s == signal_info::TERMINATE) {
              std::cout << logger::header() << "signal received\n";
              to_halt = true;
            }
            if (opt.use_termfile && exists(file_term)) {
              std::cout << logger::header() << "termination file detected\n";
              remove(file_term);
              to_halt = true;
            }
            if (to_halt) {
              BOOST_FOREACH(task& t, tasks) t.suspend_remote_clones(proxy, opt.write_xml);
              process.halt();
            }
          }

          while (!process.is_halting() || process.num_allocated() > 0) {
            boost::optional<boost::mpi::status> status
              = process.comm_ctrl().iprobe(boost::mpi::any_source, boost::mpi::any_tag);
            if (status) {
              if (status->tag() == mcmp_tag::clone_info) {
                clone_info_msg_t msg;
                process.comm_ctrl().recv(boost::mpi::any_source, status->tag(), msg);
                if (msg.info.progress() < 1) {
                  std::cout << logger::header() << "progress report: "
                            << logger::clone(msg.task_id, msg.clone_id) << " is "
                            << msg.info.phase()
                            << " (" << precision(msg.info.progress() * 100, 3) << "% done)\n";
                } else {
                  tasks[msg.task_id].info_updated(msg.clone_id, msg.info);
                  tasks[msg.task_id].halt_clone(proxy, opt.write_xml, msg.clone_id,
                                                process_group(msg.group_id));
                }
              } else if (status->tag() == mcmp_tag::clone_checkpoint) {
                clone_info_msg_t msg;
                process.comm_ctrl().recv(boost::mpi::any_source, status->tag(), msg);
                std::cout << logger::header() << "regular checkpoint: "
                          << logger::clone(msg.task_id, msg.clone_id) << " is " << msg.info.phase()
                          << " (" << precision(msg.info.progress() * 100, 3) << "% done)\n";
                if (tasks[msg.task_id].on_memory() && tasks[msg.task_id].is_running(msg.clone_id))
                  tasks[msg.task_id].info_updated(msg.clone_id, msg.info);
              } else if (status->tag() == mcmp_tag::clone_suspend) {
                clone_info_msg_t msg;
                process.comm_ctrl().recv(boost::mpi::any_source, status->tag(), msg);
                tasks[msg.task_id].clone_suspended(msg.clone_id, process_group(msg.group_id),
                                                   msg.info);
                if (tasks[msg.task_id].num_running() == 0) {
                  tasks[msg.task_id].save(opt.write_xml);
                  tasks[msg.task_id].halt();
                  save_tasks(file_out, simname, file_in_str, file_out_str, tasks);
                }
                process.release(msg.group_id);
              } else if (status->tag() == mcmp_tag::clone_halt) {
                clone_halt_msg_t msg;
                process.comm_ctrl().recv(boost::mpi::any_source, status->tag(), msg);
                task_status_t old_status = tasks[msg.task_id].status();
                tasks[msg.task_id].clone_halted(msg.clone_id);
                if (old_status != task_status::Continuing && old_status != task_status::Idling &&
                    (tasks[msg.task_id].status() == task_status::Continuing ||
                     tasks[msg.task_id].status() == task_status::Idling))
                  ++num_finished_tasks;
                if (tasks[msg.task_id].num_running() == 0) {
                  tasks[msg.task_id].save(opt.write_xml);
                  tasks[msg.task_id].halt();
                  save_tasks(file_out, simname, file_in_str, file_out_str, tasks);
                }
                process.release(msg.group_id);
              } else if (status->tag() == mcmp_tag::scheduler_halt) {
                //// 2008-11-14 ST workaround for bug in process_mpi (or boost.MPI ?)
                break;
              } else {
                std::cout << "Warning: ignoring a message with an unknown tag " << status->tag()
                          << std::endl;
              }
            } else if (clone_ptr && clone_ptr->halted()) {
              tid_t tid = clone_ptr->task_id();
              cid_t cid = clone_ptr->clone_id();
              double progress = tasks[tid].progress();
              tasks[tid].info_updated(cid, clone_ptr->info());
              tasks[tid].halt_clone(proxy, opt.write_xml, cid, process_group(0));
              if (progress < 1 && tasks[tid].progress() >= 1) ++num_finished_tasks;
              save_tasks(file_out, simname, file_in_str, file_out_str, tasks);
              process.release(0);
            } else if (clone_ptr && process.is_halting()) {
              tid_t tid = clone_ptr->task_id();
              cid_t cid = clone_ptr->clone_id();
              tasks[tid].suspend_clone(proxy, opt.write_xml, cid, process_group());
              process.release(0);
            } else if (!process.is_halting() && process.num_free() && task_queue.size()) {
              process_group g;
	      if (suspended_queue.empty()) {
		g = process.allocate();
	      } else {
		g = process.allocate(suspended_queue.front());
		suspended_queue.pop();
	      }
              tid_t tid = task_queue.top().task_id;
              task_queue.pop();
              boost::optional<cid_t> cid = tasks[tid].dispatch_clone(proxy, g);
              if (cid) {
                check_queue.push(next_checkpoint(tid, *cid, g.group_id, opt.checkpoint_interval));
                check_queue.push(next_report(tid, *cid, g.group_id, opt.report_interval));
                if (tasks[tid].can_dispatch())
                  task_queue.push(task_queue_t::value_type(tid, tasks[tid].weight()));
              } else {
                process.release(g);
              }
            } else if (!process.is_halting() && check_queue.size() && check_queue.top().due()) {
              check_queue_t::value_type q = check_queue.top();
              check_queue.pop();
              if (q.type == check_type::taskinfo) {
                std::cout << logger::header() << "checkpointing task files\n";
                BOOST_FOREACH(task const& t, tasks) { if (t.on_memory()) t.save(opt.write_xml); }
                save_tasks(file_out, simname, file_in_str, file_out_str, tasks);
                print_taskinfo(std::cout, tasks, opt.task_range);
                check_queue.push(next_taskinfo(opt.checkpoint_interval));
              } else if (q.type == check_type::checkpoint) {
                if (tasks[q.task_id].on_memory() && tasks[q.task_id].is_running(q.clone_id)) {
                  tasks[q.task_id].checkpoint(proxy, q.clone_id);
                  check_queue.push(next_checkpoint(q.task_id, q.clone_id, q.group_id,
                                                   opt.checkpoint_interval));
                }
              } else {
                if (tasks[q.task_id].on_memory() && tasks[q.task_id].is_running(q.clone_id)) {
                  tasks[q.task_id].report(proxy, q.clone_id);
                  check_queue.push(next_report(q.task_id, q.clone_id, q.group_id,
                                               opt.report_interval));
                }
              }
            } else {
              break;
            }
          }
        }

        if (clone_ptr) {

          // work some
          clone_ptr->run(stop_callback(0), boost::function<void (double)>());
          if (world.rank() != 0 && clone_ptr->halted()) {
            delete clone_ptr;
            clone_ptr = 0;
          }

        } else {

          if (world.rank() != 0 && process.comm_ctrl().iprobe(0, mcmp_tag::clone_create)) {
            // create a clone
            clone_create_msg_t msg;
            process.comm_ctrl().recv(0, mcmp_tag::clone_create, msg);
            clone_ptr = new clone_mpi(process.comm_ctrl(), process.comm_work(), basedir,
                                      opt.dump_policy, opt.check_interval, msg);
          }

        }

        // check if all processes are halted
        if (process.check_halted()) {
          if (world.rank() == 0) {
            print_taskinfo(std::cout, tasks, opt.task_range);
            std::cout << logger::header() << "all processes halted\n";
            // if (opt.auto_evaluate) {
            //   std::cout << logger::header() << "starting evaluation on "
            //             << alps::hostname() << std::endl;
            //   BOOST_FOREACH(task& t, tasks) {
            //     if (!opt.task_range.valid() || opt.task_range.is_included(t.task_id()+1))
            //       t.evaluate(opt.write_xml);
            //   }
            //   std::cout << logger::header() << "all tasks evaluated\n";
            // }
          }
          break;
        }
      }
    }

    if (world.rank() == 0) master_lock.release();
  }
  return 0;
}

#else // ALPS_HAVE_MPI

int start_mpi(int, char**) {
  boost::throw_exception(std::runtime_error(
    "This program has not been compiled for use with MPI"));
  return -1;
}

#endif // ALPS_HAVE_MPI

} // end namespace parapack
} // end namespace alps
