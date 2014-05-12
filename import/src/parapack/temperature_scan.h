/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1997-2010 by Synge Todo <wistaria@comp-phys.org>
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

#ifndef PARAPACK_TEMPERATURE_SCAN_H
#define PARAPACK_TEMPERATURE_SCAN_H

#include "mc_worker.h"
#include "montecarlo.h"
#include <alps/expression.h>
#include <alps/osiris.h>
#include <boost/shared_ptr.hpp>

namespace alps {
namespace parapack {

template<typename WORKER>
class temperature_scan_adaptor : public abstract_worker {
private:
  typedef WORKER worker_type;

public:
  static std::string version() { return worker_type::version(); }
  static void print_copyright(std::ostream& out) { worker_type::print_copyright(out); }

  temperature_scan_adaptor(alps::Parameters const& params)
    : steps_(params),
      num_temps_(static_cast<int>(evaluate("NUM_TEMPERATURES", params))),
      temp_init_(evaluate("INITIAL_TEMPERATURE", params)),
      temp_diff_(evaluate("DIFF_TEMPERATURE", params)),
      stage_(0), mcs_(params) {
    if (params.defined("INITIAL_THERMALIZATION"))
      mcs_.set_thermalization(static_cast<int>(evaluate("INITIAL_THERMALIZATION", params)));
    worker_.reset(new worker_type(params));
    std::cout << "TEMPSCAN: number of temperatures = " << num_temps_ << std::endl;
    std::cout << "TEMPSCAN: initial temperature = " << temp_init_ << std::endl;
    std::cout << "TEMPSCAN: final temperature = " << temp_init_ + (num_temps_ - 1) * temp_diff_
              << std::endl;
  }
  virtual ~temperature_scan_adaptor() {}

  void init_observables(alps::Parameters const& params, std::vector<alps::ObservableSet>& obs) {
    obs.resize(num_temps_);
    for (int p = 0; p < num_temps_; ++p) {
      worker_->init_observables(params, obs[p]);
    }
  }

  void run(std::vector<alps::ObservableSet>& obs) {
    bool is_thermalized = mcs_.is_thermalized();
    double progress = mcs_.progress();
    if (stage_ < num_temps_ && progress < 1) {
      ++mcs_;
      worker_->set_beta(1.0 / (temp_init_ + stage_ * temp_diff_));
      worker_->run(obs[stage_]);
      if (!is_thermalized && mcs_.is_thermalized()) obs[stage_].reset(true);
      if (progress < 1 && mcs_.progress() >= 1) {
        ++stage_;
        mcs_ = steps_; // reinitialize mcs_
      }
    }
  }

  void save(alps::ODump& dp) const {
    dp << stage_ << mcs_;
    worker_->save(dp);
  }
  void load(alps::IDump& dp) {
    dp >> stage_ >> mcs_;
    worker_->load(dp);
  }

  bool is_thermalized() const { return stage_ > 0 || mcs_.is_thermalized(); }
  double progress() const { return (stage_ + mcs_.progress()) / num_temps_; }

private:
  mc_steps steps_;
  int num_temps_;
  double temp_init_, temp_diff_;

  int stage_;
  mc_steps mcs_;

  boost::shared_ptr<worker_type> worker_;
};

} // end namespace parapack
} // end namespace alps

#endif // PARAPACK_TEMPERATURE_SCAN_H
