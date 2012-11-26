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

#include <alps/ngs/parapack/clone_info.h>
#include <alps/parapack/util.h>

#include <alps/utility/os.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/foreach.hpp>

namespace alps {
namespace ngs_parapack {

//
// clone_phase
//

clone_phase::clone_phase(std::vector<std::string> const& hosts, std::string const& user,
  std::string const& phase) :
  hosts_(hosts), user_(user) {
  start(phase);
}

void clone_phase::start(std::string const& phase) {
  phase_ = phase;
  startt_ = boost::posix_time::second_clock::local_time();
  stopt_ = startt_;
}

void clone_phase::stop() {
  stopt_ = boost::posix_time::second_clock::local_time();
}

void clone_phase::write_xml(oxstream& os) const {
  using namespace alps;
  os << start_tag("EXECUTED")
     << attribute("processes", hosts_.size())
     << attribute("elapsed", to_simple_string(elapsed()));
  if (phase_ != "") os << attribute("phase", phase_);
  os << start_tag("FROM") << no_linebreak
     << to_simple_string(startt_)
     << end_tag("FROM")
     << start_tag("TO") << no_linebreak
     << to_simple_string(stopt_)
     << end_tag("TO");
  int id = 0;
  BOOST_FOREACH(std::string const& h, hosts_)
    os << start_tag("MACHINE") << no_linebreak << attribute("id", ++id)
       << start_tag("NAME") << h << end_tag("NAME")
       << end_tag("MACHINE");
  if (user_ != "")
    os << start_tag("USER") << no_linebreak << user_ << end_tag("USER");
  os << end_tag("EXECUTED");
}

void clone_phase::save(hdf5::archive & ar) const {
  ar << make_pvp("user", user_) << make_pvp("phase", phase_)
     << make_pvp("from", boost::posix_time::to_iso_string(startt_))
     << make_pvp("to", boost::posix_time::to_iso_string(stopt_));
  for (std::size_t i = 0; i < hosts_.size(); ++i)
    ar << make_pvp("machine/" + boost::lexical_cast<std::string>(i) + "/name", hosts_[i]);
}

void clone_phase::load(hdf5::archive & ar) {
  ar >> make_pvp("user", user_) >> make_pvp("phase", phase_);
  std::string start_str, stop_str;
  ar >> make_pvp("from", start_str) >> make_pvp("to", stop_str);
  startt_ = boost::posix_time::from_iso_string(start_str);
  stopt_ = boost::posix_time::from_iso_string(stop_str);
  hosts_.clear();
  for (int i = 0 ; ;++i) {
    std::string p = "machine/" + boost::lexical_cast<std::string>(i) + "/name";
    if (ar.is_data(p)) {
      hosts_.push_back(std::string());
      ar >> make_pvp(p, hosts_.back());
    } else {
      break;
    }
  }
}

} // end namespace ngs_parapack
} // end namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
namespace ngs_parapack {
#endif

alps::oxstream& operator<<(alps::oxstream& os, alps::ngs_parapack::clone_phase const& phase) {
  phase.write_xml(os);
  return os;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // namespace ngs_parapack
} // namespace alps
#endif

namespace alps {
namespace ngs_parapack {

//
// clone_info
//

clone_info::clone_info() : clone_id_(0), progress_(0), is_master_(true) {}

clone_info::clone_info(cid_t cid) : clone_id_(cid), progress_(0), is_master_(true) {}

clone_info::clone_info(cid_t cid, alps::params const& p, std::string const& dump,
  bool initialize) :
  clone_id_(cid), progress_(0), is_master_(true) {
  if (initialize) init(p, dump);
}

bool clone_info::has_seed() const { return worker_seed_.size(); }

seed_t clone_info::worker_seed() const {
  if (worker_seed_.size() == 0) boost::throw_exception(std::logic_error("no seed"));
  return worker_seed_[0];
}

seed_t clone_info::disorder_seed() const {
  if (worker_seed_.size() == 0) boost::throw_exception(std::logic_error("no seed"));
  return disorder_seed_;
}


void clone_info::start(std::string const& phase) {
  if (is_master_) phases_.push_back(clone_phase(hosts_, username(), phase));
}

void clone_info::stop() {
  if (is_master_) {
    if (phases_.size()) {
      phases_.back().stop();
    } else {
      boost::throw_exception(std::logic_error("empty clone_info in clone_info::stop"));
    }
  }
}

boost::posix_time::time_duration clone_info::elapsed() const {
  boost::posix_time::time_duration t;
  BOOST_FOREACH(clone_phase p, phases_) t += p.elapsed();
  return t;
}

void clone_info::write_xml(oxstream& os) const {
  if (is_master_) {
    os << start_tag("MCRUN") << attribute("id", clone_id_+1);
    if (hosts_.size())
      os << attribute("processes", hosts_.size());
    os << attribute("status", clone_status::to_string(clone_status::status(progress_)))
       << attribute("elapsed", to_simple_string(elapsed()))
       << attribute("progress", precision(progress_ * 100, 3) + '%');
    if (has_seed()) {
      os << start_tag("DISORDER_SEED")
         << attribute("value", disorder_seed_)
         << end_tag("DISORDER_SEED");
      for (std::size_t p = 0; p < worker_seed_.size(); ++p)
        os << start_tag("SEED")
           << attribute("id", p+1)
           << attribute("value", worker_seed_[p])
           << end_tag("SEED");
    }
    BOOST_FOREACH(clone_phase const& p, phases_)
      os << p;
    for (std::size_t p = 0; p < dumpfiles_.size(); ++p)
      os << start_tag("CHECKPOINT") << no_linebreak
         << attribute("id", p+1)
         << attribute("format", "osiris")
         << attribute("file", dumpfiles_[p])
         << end_tag("CHECKPOINT");
    os << end_tag("MCRUN");
  }
}

void clone_info::init(alps::params const& p, std::string const& dump) {
  unsigned int np = this->num_processes();
  unsigned int pid = this->process_id();
  std::string f = dump + ".clone" + id2string(clone_id_+1);
  if (np > 1) {
    if (pid == 0)
      for (unsigned int p = 0; p < np; ++p) dumpfiles_.push_back(f + ".worker" + id2string(p+1));
    else
      dumpfiles_.push_back(f + ".worker" + id2string(pid+1));
  } else {
    // no process id suffix for single clone
    dumpfiles_.push_back(f);
  }

  if (p.defined("SEED")) {
    seed_t baseseed = static_cast<seed_t>(p["SEED"]);
    if (p.defined("DISORDER_SEED")) {
      disorder_seed_ = static_cast<seed_t>(p["DISORDER_SEED"]);
    } else {
      disorder_seed_ = baseseed ^ hash(clone_id_ * (np + 1) + 1);
    }
    if (pid == 0)
      for (unsigned int p = 0; p < np; ++p)
        worker_seed_.push_back(baseseed ^ hash(clone_id_ * (np + 1) + p + 2));
    else
      worker_seed_.push_back(baseseed ^ hash(clone_id_ * (np + 1) + pid + 2));
  }

  this->set_hosts(hosts_, is_master_);
}

void clone_info::set_hosts(std::vector<std::string>& hosts, bool& is_master) {
  hosts.push_back(hostname());
  is_master = true;
}

void clone_info::save(hdf5::archive & ar) const {
  ar << make_pvp("clone", clone_id_)
     << make_pvp("progress", progress_)
     << make_pvp("workerseed", worker_seed_)
     << make_pvp("disorderseed", disorder_seed_);
  for (unsigned int i = 0 ;i < phases_.size() ;++i)
    ar << make_pvp(boost::lexical_cast<std::string>(i), phases_[i]);
  for (unsigned int i = 0 ;i < dumpfiles_.size() ;++i)
    ar << make_pvp("dumpfile/" + boost::lexical_cast<std::string>(i), dumpfiles_[i]);
}

void clone_info::load(hdf5::archive & ar) {
  cid_t cid;
  ar >> make_pvp("clone", cid)
     >> make_pvp("progress", progress_)
     >> make_pvp("workerseed", worker_seed_)
     >> make_pvp("disorderseed", disorder_seed_);
  if (clone_id_ != 0 && clone_id_ != cid)
    std::cerr << "Warning: inconsistent clone id in dump file: current = " << clone_id_
              << ", dumped = " << cid << std::endl;
  clone_id_ = cid;
  phases_.clear();
  for (int i = 0; ; ++i) {
    std::string p = boost::lexical_cast<std::string>(i);
    if (ar.is_group(p)) {
      phases_.push_back(clone_phase());
      ar >> make_pvp(p, phases_.back());
    } else {
      break;
    }
  }
  dumpfiles_.clear();
  for (int i = 0; ; ++i) {
    std::string p = "dumpfile/" + boost::lexical_cast<std::string>(i);
    if (ar.is_data(p)) {
      dumpfiles_.push_back(std::string());
      ar >> make_pvp(p, dumpfiles_.back());
    } else {
      break;
    }
  }
}

} // namespace ngs_parapack
} // namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
namespace ngs_parapack {
#endif

alps::oxstream& operator<<(alps::oxstream& os, alps::ngs_parapack::clone_info const& info) {
  info.write_xml(os);
  return os;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // namespace ngs_parapack
} // namespace alps
#endif

#ifdef ALPS_HAVE_MPI

namespace alps {
namespace ngs_parapack {

clone_info_mpi::clone_info_mpi(boost::mpi::communicator const& comm, cid_t cid,
  alps::params const& p, std::string const& dump) :
  clone_info(cid, p, dump, false), comm_(comm) {
  clone_info::init(p, dump);
}

unsigned int clone_info_mpi::num_processes() const { return comm_.size(); }

unsigned int clone_info_mpi::process_id() const { return comm_.rank(); }

void clone_info_mpi::set_hosts(std::vector<std::string>& hosts, bool& is_master) {
  is_master = (comm_.rank() == 0);
  std::string host = alps::hostname();
  if (is_master) {
    hosts.resize(comm_.size());
    gather(comm_, host, hosts, 0);
  } else {
    gather(comm_, host, 0);
  }
}

} // namespace ngs_parapack
} // namespace alps

#endif // ALPS_HAVE_MPI
