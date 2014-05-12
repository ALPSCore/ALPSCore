/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1997-2011 by Synge Todo <wistaria@comp-phys.org>
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

#ifndef PARAPACK_EXCHANGE_H
#define PARAPACK_EXCHANGE_H

#include "integer_range.h"
#include "mc_worker.h"
#include "parapack.h"
#include "permutation.h"
#include "process.h"
#include <alps/expression.h>
#include <alps/osiris.h>

#include <boost/lexical_cast.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/throw_exception.hpp>

#include <algorithm>
#include <functional>
#include <cmath>
#include <vector>

namespace alps {
namespace parapack {
namespace exmc {

//
// walker labels
//

struct walker_direc {
  enum walker_direc_t { up, down, unlabeled };
};
typedef walker_direc::walker_direc_t walker_direc_t;


//
// default initializer and helpers
//

struct no_initializer { no_initializer(alps::Parameters const&) {} };

template<typename WALKER, typename INITIALIZER> struct initializer_helper;

template<typename WALKER>
struct initializer_helper<WALKER, no_initializer> {
  typedef WALKER walker_type;
  static boost::shared_ptr<walker_type>
  create_walker(alps::Parameters const& params, no_initializer const&) {
    return boost::shared_ptr<walker_type>(new walker_type(params));
  }
  static void init_observables(boost::shared_ptr<walker_type> walker_ptr,
    alps::Parameters const& params, no_initializer const&, alps::ObservableSet& obs) {
    walker_ptr->init_observables(params, obs);
  }
  static void run_walker(boost::shared_ptr<walker_type> walker_ptr, no_initializer const&,
    alps::ObservableSet& obs) {
    walker_ptr->run(obs);
  }
};

template<typename WALKER, typename INITIALIZER>
struct initializer_helper {
  typedef WALKER walker_type;
  typedef INITIALIZER initializer_type;
  static boost::shared_ptr<walker_type>
  create_walker(alps::Parameters const& params, initializer_type const& init) {
    return boost::shared_ptr<walker_type>(new walker_type(params, init));
  }
  static void init_observables(boost::shared_ptr<walker_type> walker_ptr,
    alps::Parameters const& params, initializer_type const& init, alps::ObservableSet& obs) {
    walker_ptr->init_observables(params, init, obs);
  }
  static void run_walker(boost::shared_ptr<walker_type> walker_ptr, initializer_type const& init,
    alps::ObservableSet& obs) {
    walker_ptr->run(init, obs);
  }
};


//
// temperature set
//

class inverse_temperature_set {
public:
  typedef double value_type;
  typedef std::size_t size_type;
  typedef std::vector<double>::iterator iterator;
  typedef std::vector<double>::const_iterator const_iterator;

  inverse_temperature_set(alps::Parameters const& params) { init(params); }
  void init(alps::Parameters const& params) {
    using std::sqrt;
    int dist = -1;
    // 0 : explicit
    // 1 : equidistance in beta
    // 2 : equidistance in 1/beta
    // 3 : equidistance in sqrt(beta)
    if (params.defined("TEMPERATURE_DISTRIBUTION_TYPE")) {
      dist = static_cast<int>(params["TEMPERATURE_DISTRIBUTION_TYPE"]);
    } else if (params.defined("BETA_MAX") && params.defined("BETA_MIN")) {
      dist = 1;
    } else if (params.defined("T_MAX") && params.defined("T_MIN")) {
      dist = 2;
    }
    if (params.defined("INVERSE_TEMPERATURE_SET") || params.defined("TEMPERATURE_SET")) dist = 0;
    if (dist < 0 || dist > 3)
      boost::throw_exception(std::invalid_argument("illegal temperature distribution"));
    if (params.defined("INVERSE_TEMPERATURE_SET")) {
      read_vector_resize(params["INVERSE_TEMPERATURE_SET"], beta_);
    } else if (params.defined("TEMPERATURE_SET")) {
      read_vector_resize(params["TEMPERATURE_SET"], beta_);
      for (int i = 0; i < beta_.size(); ++i) beta_[i] = 1 / beta_[i];
    } else if (params.defined("NUM_REPLICAS")) {
      int n = static_cast<int>(evaluate("NUM_REPLICAS", params));
      double b_max = 0;
      double b_min = 0;
      if (params.defined("BETA_MAX") && params.defined("BETA_MIN")) {
        b_max = evaluate("BETA_MAX", params);
        b_min = evaluate("BETA_MIN", params);
      } else if (params.defined("T_MAX") && params.defined("T_MIN")) {
        b_max = 1 / evaluate("T_MIN", params);
        b_min = 1 / evaluate("T_MAX", params);
      } else {
        boost::throw_exception(std::invalid_argument("MAX/MIN values of BETA (or T) not defined"));
      }
      beta_.resize(n);
      if (dist == 2) {
        b_max = 1 / b_max;
        b_min = 1 / b_min;
      } else if (dist == 3) {
        b_max = sqrt(b_max);
        b_min = sqrt(b_min);
      }
      double step = (n > 1) ? (b_max - b_min) / (n - 1) : 0;
      for (int i = 0; i < n; ++i) beta_[i] = b_max - step * i;
      if (dist == 2) {
        for (int i = 0; i < n; ++i) beta_[i] = 1 / beta_[i];
      } else if (dist == 3) {
        for (int i = 0; i < n; ++i) beta_[i] = beta_[i] * beta_[i];
      }
    } else {
      boost::throw_exception(std::invalid_argument("temperature set not defined"));
    }
    std::sort(beta_.begin(), beta_.end());
  }
  std::size_t size() const { return beta_.size(); }
  double operator[](std::size_t i) const { return beta_[i]; }

  bool optimize(std::vector<double> const& population) {
    // population : population ratio of UPWARD walkers at each level
    using std::pow; using std::sqrt;
    for (int p = 0; p < size() - 1; ++p)
      if (population[p+1] - population[p] <= 0) return false;
    diffusivity_.resize(size() - 1);
    for (int p = 0; p < size() - 1; ++p)
      diffusivity_[p] = pow(beta_[p+1] - beta_[p], 2.) / (population[p+1] - population[p]);
    // optimize inverse temperatures
    accum_.resize(size() - 1);
    double c = 0;
    for (int p = 0; p < size() - 1; ++p) {
      c += (beta_[p+1] - beta_[p]) / sqrt(diffusivity_[p]);
      accum_[p] = c;
    }
    c /= (size() - 1);
    oldbeta_.resize(size());
    std::copy(beta_.begin(), beta_.end(), oldbeta_.begin());
    beta_.front() = oldbeta_.front();
    int q = 0;
    for (int p = 1; p < size() - 1; ++p) {
      double t = c * p;
      for (;; ++q) if (accum_[q] > t) break;
      beta_[p] = oldbeta_[q+1] - sqrt(diffusivity_[q]) * (accum_[q] - t);
    }
    beta_.back() = oldbeta_.back();
    return true;
  }

  bool optimize2(std::vector<double> const& population) {
    // population : population ratio of UPWARD walkers at each level
    using std::pow; using std::sqrt;
    for (int p = 0; p < size() - 1; ++p)
      if (population[p+1] - population[p] <= 0) return false;
    // original temperature set
    std::vector<double> T;
    std::vector<double> deltaT;
    // optimized temperature set
    std::vector<double> T_prime;
    std::vector<double> deltaT_prime;
    // some more stuff
    std::vector<double> Fraction;
    std::vector<double> Derivative;
    std::vector<double> Diffusivity;
    for (int p = 0; p < size(); ++p) {
      Fraction.push_back(population[size() - p - 1]);
      T.push_back(1. / beta_[size() - p - 1]);
    }
    // determine deltaT
    for(int i=0; i<T.size()-1; ++i) {
      deltaT.push_back(T[i+1]-T[i]);
    }
    // determine derivative of fraction
    for(int i=0; i<T.size()-1; ++i) {
      // regression with 3 points
      double a, b;
      boost::tie(a, b) = linear_regression(3, ((i == 0) ? 0 : i - 1), T, Fraction);
      Derivative.push_back(b);
    }
    // check if reweighting is possible
    for(int i=0; i<Derivative.size(); ++i)
      if(Derivative[i]>=0) return false;
    // feedback of local diffusivity
    const int N = T.size()-1;  // number of intervals
    for(int t=0; t<N; ++t) {
      deltaT_prime.push_back( 1./sqrt( -1. / deltaT[t] * Derivative[t] ) );
    }
    // determine normalization
    double C = .0;
    if(deltaT.size() != N) { std::cerr << "FEEDBACK ERROR: Inconsistent size of deltaT" << std::endl; exit(1); }
    for(int t=0; t<N; ++t) {
      C += deltaT[t] / deltaT_prime[t];
    }
    C = N/C;
    // determine new temperature set
    T_prime.push_back( T[0] );
    double n = 0.;
    double n_prime = 0.;
    int    t = 0;
    double this_deltaT = 0.;
    do {
      if(n_prime + C * deltaT[t] / deltaT_prime[t] >= n+1) {
        double tau = deltaT_prime[t]/C * (n - n_prime + 1);
        this_deltaT += tau;
        T_prime.push_back( T_prime.back() + this_deltaT );
        n++;
        n_prime += C * tau/deltaT_prime[t];
        deltaT[t] -= tau;
        this_deltaT = 0.;
      }
      else {
        n_prime += C * deltaT[t] / deltaT_prime[t];
        this_deltaT += deltaT[t];
        t++;
      }
    } while(n<N && t<N);
    if(t==N) {
      T_prime.push_back( T[N] );
    }

    for (int p = 0; p < size(); ++p) beta_[p] = 1./T_prime[size() - p - 1];
    return true;
  }

  template<typename WALKER>
  void optimize_h1999(std::vector<typename WALKER::weight_parameter_type> const& wp) {
    // optimization of temperature set based on energy mesurement (Hukushima PRE 60, 3606 (1999)
    typedef WALKER walker_type;
    typedef typename walker_type::weight_parameter_type weight_parameter_type;
    const int num_rep = 32;
    const double convergence = 0.01;
    oldbeta_.resize(size());
    std::copy(beta_.begin(), beta_.end(), oldbeta_.begin());
    for (int rep = 0; rep < 2 * num_rep; ++rep) {
      for (int i = (rep % 2) + 1; i < (size() - 1); i += 2) {
        const weight_parameter_type gm = interpolate<walker_type>(oldbeta_, wp, beta_[i-1]);
        const weight_parameter_type gp = interpolate<walker_type>(oldbeta_, wp, beta_[i+1]);
        double bl = beta_[i-1];
        double b0 = beta_[i];
        double br = beta_[i+1];
        const double thresh = convergence * (br - bl);
        weight_parameter_type g0 = interpolate<walker_type>(oldbeta_, wp, b0);
        if (walker_type::log_weight(gm, beta_[i-1]) < walker_type::log_weight(g0, beta_[i-1]) &&
            walker_type::log_weight(g0, beta_[i+1]) < walker_type::log_weight(gp, beta_[i+1])) {
          while (true) {
            if (((walker_type::log_weight(gm, beta_[i-1]) + walker_type::log_weight(g0, b0)) -
                 (walker_type::log_weight(gm, b0) + walker_type::log_weight(g0, beta_[i-1]))) <
                ((walker_type::log_weight(g0, b0) + walker_type::log_weight(gp, beta_[i+1])) -
                 (walker_type::log_weight(g0, beta_[i+1]) + walker_type::log_weight(gp, b0)))) {
              bl = b0;
              b0 = (b0 + br) / 2;
            } else {
              br = b0;
              b0 = (bl + b0) / 2;
            }
            if (br - bl < thresh) break;
            g0 = interpolate<walker_type>(oldbeta_, wp, b0);
          }
          beta_[i] = b0;
        }
      }
    }
  }

  void save(alps::ODump& dp) const { dp << beta_; }
  void load(alps::IDump& dp) { dp >> beta_; }

  static std::pair<double, double> linear_regression(const int num_points, const int offset,
    std::vector<double> const& x, std::vector<double> const& y) {
    double ss_xy = 0.0;
    double ss_xx = 0.0;
    double mean_x = 0.0;
    double mean_y = 0.0;
    for(int i = offset; i < offset + num_points; ++i) {
      mean_x += x[i];
      mean_y += y[i];
    }
    mean_x /= num_points;
    mean_y /= num_points;
    for(int i = offset; i < offset + num_points; ++i) {
      ss_xy += x[i]*y[i];
      ss_xx += x[i]*x[i];
    }
    ss_xy -= num_points * mean_x * mean_y;
    ss_xx -= num_points * mean_x * mean_x;
    const double b = ss_xy / ss_xx;       // slope
    const double a = mean_y - b * mean_x; // offset
    return std::make_pair(a, b);
  }

  template<typename WALKER>
  static typename WALKER::weight_parameter_type interpolate(std::vector<double> const& beta,
    std::vector<typename WALKER::weight_parameter_type> const& func, double b) {
    if (b < beta.front() && b > beta.back())
      boost::throw_exception(std::range_error("interpolate"));
    int n = std::lower_bound(beta.begin(), beta.end(), b) - beta.begin();
    if (n == 0) ++n;
    return (((beta[n] - b) / (beta[n] - beta[n-1])) * func[n-1] +
            ((b - beta[n-1]) / (beta[n] - beta[n-1])) * func[n]);
  }

private:
  std::vector<double> beta_;
  // working space
  std::vector<double> diffusivity_;
  std::vector<double> accum_;
  std::vector<double> oldbeta_;
};

} // end namespace exmc
} // end namespace parapack
} // end namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
namespace parapack {
namespace exmc {
#endif

inline alps::ODump& operator<<(alps::ODump& dp,
  alps::parapack::exmc::inverse_temperature_set const& beta) {
  beta.save(dp);
  return dp;
}

inline alps::IDump& operator>>(alps::IDump& dp,
  alps::parapack::exmc::inverse_temperature_set& beta) {
  beta.load(dp);
  return dp;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace exmc
} // end namespace parapack
} // end namespace alps
#endif

namespace alps {
namespace parapack {
namespace exmc {

class exchange_steps {
public:
  typedef integer_range<unsigned int> range_type;

  enum opttype_t { rate, population };

  exchange_steps(alps::Parameters const& p) : mcs_(0), stage_(0), count_(0),
    exchange_(!static_cast<bool>(evaluate(p.value_or_default("NO_EXCHANGE", 0)))),
    interval_(static_cast<int>(evaluate(p.value_or_default("EXCHANGE_INTERVAL", 1)))),
    random_(static_cast<bool>(evaluate(p.value_or_default("RANDOM_EXCHANGE", 0)))),
    optimize_(static_cast<bool>(p.value_or_default("OPTIMIZE_TEMPERATURE", 0)) ||
              static_cast<bool>(p.value_or_default("TEMPERATURE_OPTIMIZATION", 0))),
    sweep_(p.value_or_default("SWEEPS", "[65536:]"), p),
    therm_(p.defined("THERMALIZATION") ?
           static_cast<unsigned int>(alps::evaluate("THERMALIZATION", p)) :
           (sweep_.min BOOST_PREVENT_MACRO_SUBSTITUTION () >> 3)),
    optstep_(0), iteration_(0) {

    if (exchange_ && optimize_) {
      if (!p.defined("OPTIMIZATION_TYPE") || p["OPTIMIZATION_TYPE"] == "rate") {
        opttype_ = rate;
      } else if (p["OPTIMIZATION_TYPE"] == "population") {
        opttype_ = population;
      } else {
        boost::throw_exception(
          std::runtime_error("unknown OPIMIZATION_TYPE: " + p["OPTIMIZATION_TYPE"]));
      }

      if (p.defined("NUM_CLONES") &&
          integer_range<uint32_t>(p["NUM_CLONES"]).max BOOST_PREVENT_MACRO_SUBSTITUTION () > 1)
        boost::throw_exception(std::logic_error("temperature optimization and multipile clones"
                                                " are exclusive"));

      int m;
      if (opttype_ == rate) {
        factor_ = p.value_or_default("BLOCK_SWEEP_FACTOR", 1);
        iteration_ =
          static_cast<unsigned int>(p.value_or_default("OPTIMIZATION_ITERATIONS", 1)) + 1;
        m = p.value_or_default("INITIAL_BLOCK_SWEEPS", therm_);
      } else {
        factor_ = p.value_or_default("BLOCK_SWEEP_FACTOR", 2);
        iteration_ =
          static_cast<unsigned int>(p.value_or_default("OPTIMIZATION_ITERATIONS", 7)) + 1;
        m = p.value_or_default("INITIAL_BLOCK_SWEEPS", (2 << 8));
      }
      optstep_ = 0;
      block_.resize(iteration_);
      for (int i = 0; i < iteration_; ++i, m = static_cast<int>(factor_ * m)) {
        block_[i] = m;
        optstep_ += m;
      }
    }
  }

  bool exchange() const { return exchange_; }
  unsigned int interval() const { return interval_; }
  bool random_exchange() const { return random_; }
  bool optimize() const { return optimize_; }
  opttype_t optimization_type() const { return opttype_; }

  exchange_steps& operator++() { ++mcs_; ++count_; return *this; }
  exchange_steps operator++(int) { exchange_steps tmp = *this; this->operator++(); return tmp; }

  void continue_thermalization() { therm_ = static_cast<unsigned>(factor_ * therm_); }
  void continue_stage() {
    for (int s = stage_; s < iteration_; ++s) {
      unsigned int oldstep = block_[s];
      block_[s] = static_cast<unsigned int>(factor_ * oldstep);
      optstep_ += (block_[s] - oldstep);
    }
  }
  void next_stage() {
    if (!doing_optimization())
      boost::throw_exception(std::logic_error("exchange_steps::next_stage()"));
    ++stage_;
    count_ = 0;
  }

  unsigned int operator()() const { return mcs_; }
  unsigned int stage() const { return stage_; }
  unsigned int stage_count() const { return count_; }
  bool can_work() const {
    return !is_thermalized() &&
      (mcs_ - optstep_ - therm_) < sweep_.max BOOST_PREVENT_MACRO_SUBSTITUTION ();
  }
  bool is_thermalized() const { return mcs_ >= optstep_ && (mcs_ - optstep_) >= therm_; }
  double progress() const {
    return static_cast<double>(mcs_) /
      (optstep_ + therm_ + sweep_.min BOOST_PREVENT_MACRO_SUBSTITUTION ());
  }
  bool doing_optimization() const { return perform_optimization() && (stage_ < iteration_); }

  int thermalization() const { return therm_; }
  range_type sweeps() const { return sweep_; }
  unsigned int stage_sweeps() const { return block_[stage_]; }
  unsigned int optimization_sweeps() const { return optstep_; }
  unsigned int iterations() const { return iteration_; }
  bool perform_optimization() const { return optimize_; }

  void save(alps::ODump& dp) const {
    dp << mcs_ << stage_ << count_ << exchange_ << interval_ << optimize_ << factor_ << sweep_
       << therm_ << optstep_ << block_ << iteration_;
  }
  void load(alps::IDump& dp) {
    dp >> mcs_ >> stage_ >> count_ >> exchange_ >> interval_ >> optimize_ >> factor_ >> sweep_
       >> therm_ >> optstep_ >> block_ >> iteration_;
  }

private:
  int mcs_;                         // MC count from the first
  unsigned int stage_;              // present stage of optimization [0,iteration_)
  unsigned int count_;              // MC count in the present stage [0,block_[stage_])

  bool exchange_;                   // whether replica exchange is performed or not
  unsigned int interval_;           // interval between replica exchange
  bool random_;                     // whether exchange is performed in random order or not
  bool optimize_;                   // whether optimization is performed or not
  opttype_t opttype_;               // optimization type (rate, population)
  double factor_;                   // increment factor of MC steps in the next optimization stage

  range_type sweep_;                // MC steps for measurement
  unsigned int therm_;              // MC steps for thermazliation (after optimization)
  unsigned int optstep_;            // total MC steps for optimization
  std::vector<unsigned int> block_; // MC steps for each optimization stage
  unsigned int iteration_;          // number of optimization iterations
};

} // end namespace exmc
} // end namespace parapack
} // end namespace alps


#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
namespace parapack {
namespace exmc {
#endif

inline alps::ODump& operator<<(alps::ODump& dp, alps::parapack::exmc::exchange_steps const& mcs) {
  mcs.save(dp);
  return dp;
}

inline alps::IDump& operator>>(alps::IDump& dp, alps::parapack::exmc::exchange_steps& mcs) {
  mcs.load(dp);
  return dp;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace exmc
} // end namespace parapack
} // end namespace alps
#endif


namespace alps {
namespace parapack {

template<typename WALKER, typename INITIALIZER = exmc::no_initializer>
class single_exchange_worker : public mc_worker {
private:
  typedef mc_worker super_type;
  typedef WALKER walker_type;
  typedef typename walker_type::weight_parameter_type weight_parameter_type;
  typedef INITIALIZER initializer_type;
  typedef exmc::initializer_helper<walker_type, initializer_type> helper;
  typedef exmc::walker_direc walker_direc;

public:
  static std::string version() { return walker_type::version(); }
  static void print_copyright(std::ostream& out) { walker_type::print_copyright(out); }

  single_exchange_worker(alps::Parameters const& params)
    : super_type(params), init_(params), beta_(params), mcs_(params), num_returnee_(0) {

    int nrep = beta_.size();
    std::cout << "EXMC: number of replicas = " << nrep << std::endl;
    std::cout << "EXMC: initial inverse temperature set = " << write_vector(beta_, " ", 5)
              << std::endl;

    // initialize walkers
    walker_.resize(nrep);
    tid_.resize(nrep);
    alps::Parameters wp(params);
    for (int p = 0; p < nrep; ++p) {
      for (int j = 1; j < 3637 /* 509th prime number */; ++j) engine()();
      wp["WORKER_SEED"] = engine()(); // different seed for each walker
      walker_[p] = helper::create_walker(wp, init_); // same DISORDER_SEED for all walkers
      tid_[p] = p;
    }

    // initialize walker labels
    wid_.resize(nrep);
    for (int p = 0; p < nrep; ++p) wid_[p] = p;
    if (mcs_.exchange()) {
      direc_.resize(nrep);
      direc_[0] = walker_direc::down;
      for (int p = 1; p < nrep; ++p) direc_[p] = walker_direc::unlabeled;
    }

    // working space
    if (mcs_.exchange()) {
      wp_.resize(nrep);
      upward_.resize(nrep);
      accept_.resize(nrep - 1);
      if (mcs_.random_exchange()) permutation_.resize(nrep - 1);
    }
    if (mcs_.exchange()) {
      weight_parameters_.resize(nrep);
      for (int p = 0; p < nrep; ++p) weight_parameters_[p] = weight_parameter_type(0);
    }
  }
  virtual ~single_exchange_worker() {}

  void init_observables(alps::Parameters const& params, std::vector<alps::ObservableSet>& obs) {
    int nrep = beta_.size();
    obs.resize(nrep);
    for (int p = 0; p < nrep; ++p) {
      helper::init_observables(walker_[0], params, init_, obs[p]);
      obs[p] << SimpleRealObservable("EXMC: Temperature")
             << SimpleRealObservable("EXMC: Inverse Temperature");
      if (mcs_.exchange()) {
        obs[p] << SimpleRealObservable("EXMC: Ratio of Upward-Moving Walker")
               << SimpleRealObservable("EXMC: Ratio of Downward-Moving Walker")
               << SimpleRealObservable("EXMC: Inverse Round-Trip Time");
        obs[p]["EXMC: Ratio of Upward-Moving Walker"].reset(true);
        obs[p]["EXMC: Ratio of Downward-Moving Walker"].reset(true);
        if (p != nrep - 1) {
          obs[p] << SimpleRealObservable("EXMC: Acceptance Rate");
          obs[p]["EXMC: Acceptance Rate"].reset(true);
        }
      }
    }
    if (mcs_.exchange()) obs[0] << SimpleRealObservable("EXMC: Average Inverse Round-Trip Time");
  }

  void run(std::vector<alps::ObservableSet>& obs) {
    ++mcs_;

    int nrep = beta_.size();
    for (int p = 0; p < nrep; ++p) {
      add_constant(obs[p]["EXMC: Temperature"], 1 / beta_[p]);
      add_constant(obs[p]["EXMC: Inverse Temperature"], beta_[p]);
    }

    // MC update of each replica
    for (int w = 0; w < nrep; ++w) {
      int p = tid_[w];
      walker_[w]->set_beta(beta_[p]);
      helper::run_walker(walker_[w], init_, obs[p]);
    }

    // replica exchange process
    if (mcs_.exchange() && (mcs_() % mcs_.interval()) == 0) {

      for (int w = 0; w < nrep; ++w) {
        int p = tid_[w];
        wp_[w] = walker_[w]->weight_parameter();
        weight_parameters_[p] += wp_[w];
      }

      if (mcs_.random_exchange()) {
        // random exchange
        for (int p = 0; p < nrep - 1; ++p) permutation_[p] = p;
        alps::random_shuffle(permutation_.begin(), permutation_.end(), generator_01());

        for (int i = 0; i < nrep - 1; ++i) {
          int p = permutation_[i];
          int w0 = wid_[p];
          int w1 = wid_[p+1];
          double logp = ((walker_type::log_weight(wp_[w1], beta_[p]  ) +
                          walker_type::log_weight(wp_[w0], beta_[p+1])) -
                         (walker_type::log_weight(wp_[w1], beta_[p+1]) +
                          walker_type::log_weight(wp_[w0], beta_[p]  )));
          //// std::cerr << mcs_() << ' ' << w0 << ' ' << w1 << ' ' << logp << std::endl;
          if (uniform_01() < std::exp(logp)) {
            std::swap(tid_[w0], tid_[w1]);
            std::swap(wid_[p], wid_[p+1]);
            obs[p]["EXMC: Acceptance Rate"] << 1.;
          } else {
            obs[p]["EXMC: Acceptance Rate"] << 0.;
          }
        }

      } else {
        // alternating exchange
        int start = (mcs_() / mcs_.interval()) % 2;
        for (int p = start; p < nrep - 1; p += 2) {
          int w0 = wid_[p];
          int w1 = wid_[p+1];
          double logp = ((walker_type::log_weight(wp_[w1], beta_[p]  ) +
                          walker_type::log_weight(wp_[w0], beta_[p+1])) -
                         (walker_type::log_weight(wp_[w1], beta_[p+1]) +
                          walker_type::log_weight(wp_[w0], beta_[p]  )));
          //// std::cerr << mcs_() << ' ' << w0 << ' ' << w1 << ' ' << logp << std::endl;
          if (uniform_01() < std::exp(logp)) {
            std::swap(tid_[w0], tid_[w1]);
            std::swap(wid_[p], wid_[p+1]);
            obs[p]["EXMC: Acceptance Rate"] << 1.;
          } else {
            obs[p]["EXMC: Acceptance Rate"] << 0.;
          }
        }
      }

      int wtop = wid_.front();
      for (int w = 0; w < nrep; ++w) {
        if (w == wtop && direc_[w] == walker_direc::up) {
          obs[w]["EXMC: Inverse Round-Trip Time"] << 1.;
        } else {
          obs[w]["EXMC: Inverse Round-Trip Time"] << 0.;
        }
      }
      if (direc_[wtop] == walker_direc::up) {
        obs[0]["EXMC: Average Inverse Round-Trip Time"] << 1. / nrep;
        ++num_returnee_;
      } else {
        obs[0]["EXMC: Average Inverse Round-Trip Time"] << 0.;
      }
      direc_[wtop] = walker_direc::down;
      if (direc_[wid_.back()] == walker_direc::down) direc_[wid_.back()] = walker_direc::up;
      for (int p = 0; p < nrep; ++p) {
        obs[p]["EXMC: Ratio of Upward-Moving Walker"] <<
          (direc_[wid_[p]] == walker_direc::up ? 1. : 0.);
        obs[p]["EXMC: Ratio of Downward-Moving Walker"] <<
          (direc_[wid_[p]] == walker_direc::down ? 1. : 0.);
      }

      if (mcs_.doing_optimization() && mcs_.stage_count() == mcs_.stage_sweeps()) {

        if (mcs_.optimization_type() == exmc::exchange_steps::rate) {

          for (int p = 0; p < nrep - 1; ++p)
            accept_[p] =
              reinterpret_cast<SimpleRealObservable&>(obs[p]["EXMC: Acceptance Rate"]).mean();
          for (int p = 0; p < nrep; ++p) wp_[p] = weight_parameters_[p] / mcs_.stage_count();
          std::cout << "EXMC stage " << mcs_.stage() << ": acceptance rate = "
                    << write_vector(accept_, " ", 3) << std::endl;

          if (mcs_.stage() != 0) {
            beta_.optimize_h1999<walker_type>(wp_);
            std::cout << "EXMC stage " << mcs_.stage() << ": optimized inverse temperature set = "
                      << write_vector(beta_, " ", 5) << std::endl;
          }
          mcs_.next_stage();

          for (int p = 0; p < nrep - 1; ++p) {
            obs[p]["EXMC: Acceptance Rate"].reset(true);
          }
          for (int p = 0; p < nrep; ++p) {
            obs[p]["EXMC: Ratio of Upward-Moving Walker"].reset(true);
            obs[p]["EXMC: Ratio of Downward-Moving Walker"].reset(true);
            weight_parameters_[p] = weight_parameter_type(0);
          }

        } else {

          bool success = (num_returnee_ >= nrep);

          int nu = 0;
          for (int p = 0; p < nrep; ++p) if (direc_[p] == walker_direc::unlabeled) ++nu;
          if (nu > 0) success = false;

          for (int p = 0; p < nrep; ++p) {
            double up = reinterpret_cast<SimpleRealObservable&>(
              obs[p]["EXMC: Ratio of Upward-Moving Walker"]).mean();
            double down = reinterpret_cast<SimpleRealObservable&>(
              obs[p]["EXMC: Ratio of Downward-Moving Walker"]).mean();
            upward_[p] = (up + down > 0) ? up / (up + down) : alps::nan();
          }

          for (int p = 0; p < nrep - 1; ++p)
            accept_[p] = reinterpret_cast<SimpleRealObservable&>(
              obs[p]["EXMC: Acceptance Rate"]).mean();

          std::cout << "EXMC stage " << mcs_.stage()
                    << ": stage count = " << mcs_.stage_count() << '\n'
                    << "EXMC stage " << mcs_.stage()
                    << ": number of returned walkers = " << num_returnee_ << '\n'
                    << "EXMC stage " << mcs_.stage()
                    << ": number of unlabeled walkers = " << nu << '\n'
                    << "EXMC stage " << mcs_.stage()
                    << ": population ratio of upward-moving walkers "
                    << write_vector(upward_, " ", 5) << '\n'
                    << "EXMC stage " << mcs_.stage()
                    << ": acceptance rate " << write_vector(accept_, " ", 3) << std::endl;

          // preform optimization
          if (mcs_.stage() != 0 && success) success = beta_.optimize2(upward_);

          if (success) {
            std::cout << "EXMC stage " << mcs_.stage() << ": DONE" << std::endl;
            if (mcs_.stage() > 0)
              std::cout << "EXMC stage " << mcs_.stage() << ": optimized inverse temperature set = "
                        << write_vector(beta_, " ", 5) << std::endl;
            mcs_.next_stage();
            for (int p = 0; p < nrep - 1; ++p) {
              obs[p]["EXMC: Acceptance Rate"].reset(true);
            }
            for (int p = 0; p < nrep; ++p) {
              obs[p]["EXMC: Ratio of Upward-Moving Walker"].reset(true);
              obs[p]["EXMC: Ratio of Downward-Moving Walker"].reset(true);
            }
            num_returnee_ = 0;
          } else {
            // increase stage sweeps
            mcs_.continue_stage();
            std::cout << "EXMC stage " << mcs_.stage() << ": NOT FINISHED\n"
                      << "EXMC stage " << mcs_.stage() << ": increased number of sweeps to "
                      << mcs_.stage_sweeps() << std::endl;
          }
        }
      }

      // check whether all the replicas have revisited the highest temperature or not
      if (!mcs_.perform_optimization() && mcs_() == mcs_.thermalization()) {
        int nu = 0;
        for (int p = 0; p < nrep; ++p) if (direc_[p] == walker_direc::unlabeled) ++nu;
        std::cout << "EXMC: thermalization count = " << mcs_() << '\n'
                  << "EXMC: number of returned walkers = " << num_returnee_ << '\n'
                  << "EXMC: number of unlabeled walkers = " << nu << std::endl;
        if ((num_returnee_ >= nrep) && (nu == 0)) {
          std::cout << "EXMC: thermzlization DONE" << std::endl;
        } else {
          mcs_.continue_stage();
          std::cout << "EXMC: thermalization NOT FINISHED\n"
                    << "EXMC: increased number of thermalization sweeps to "
                    << mcs_.thermalization() << std::endl;
        }
      }
    }
  }

  void save(alps::ODump& dp) const {
    dp << beta_ << mcs_ << tid_ << wid_ << direc_ << num_returnee_ << weight_parameters_;
    for (int i = 0; i < walker_.size(); ++i) walker_[i]->save(dp);
  }
  void load(alps::IDump& dp) {
    dp >> beta_ >> mcs_ >> tid_ >> wid_ >> direc_ >> num_returnee_ >> weight_parameters_;
    for (int i = 0; i < walker_.size(); ++i) walker_[i]->load(dp);
    permutation_.resize(beta_.size() - 1);
    upward_.resize(beta_.size());
  }

  bool is_thermalized() const { return mcs_.is_thermalized(); }
  double progress() const { return mcs_.progress(); }

  static void evaluate_observable(alps::ObservableSet& obs) {
    walker_type::evaluate_observable(obs);
  }

private:
  initializer_type init_;
  std::vector<boost::shared_ptr<walker_type> > walker_;

  exmc::inverse_temperature_set beta_;
  exmc::exchange_steps mcs_;
  std::vector<int> tid_;     // temperature id of each walker (replica)
  std::vector<int> wid_;     // id of walker (replica) at each temperature
  std::vector<int> direc_;   // direction of each walker
  int num_returnee_;         // number of walkers returned to highest temperature
  std::vector<weight_parameter_type> weight_parameters_;

  // working space
  std::vector<weight_parameter_type> wp_;
  std::vector<double> upward_;
  std::vector<double> accept_;
  std::vector<int> permutation_;
};

#ifdef ALPS_HAVE_MPI

template<typename WALKER, typename INITIALIZER = exmc::no_initializer>
class parallel_exchange_worker : public mc_worker {
private:
  typedef mc_worker super_type;
  typedef WALKER walker_type;
  typedef typename walker_type::weight_parameter_type weight_parameter_type;
  typedef INITIALIZER initializer_type;
  typedef exmc::initializer_helper<walker_type, initializer_type> helper;
  typedef exmc::walker_direc walker_direc;

public:
  static std::string version() { return walker_type::version(); }
  static void print_copyright(std::ostream& out) { walker_type::print_copyright(out); }

  parallel_exchange_worker(boost::mpi::communicator const& comm, alps::Parameters const& params)
    : super_type(params), comm_(comm), init_(params), beta_(params), mcs_(params),
      num_returnee_(0) {

    int nrep = beta_.size();
    boost::tie(nrep_local_, offset_local_) = calc_nrep(comm_.rank());
    if (nrep_local_ == 0) {
      std::cerr << "Error: number of replicas is smaller than number of processes\n";
      boost::throw_exception(std::runtime_error(
        "number of replicas is smaller than number of processes"));
    }
    if (comm_.rank() == 0) {
      nreps_.resize(comm_.size());
      offsets_.resize(comm_.size());
      nrep_max_ = 0;
      for (int p = 0; p < comm_.size(); ++p) {
        boost::tie(nreps_[p], offsets_[p]) = calc_nrep(p);
        nrep_max_ = std::max(nrep_max_, nreps_[p]);
      }
      std::cout << "EXMC: number of replicas = " << nrep << std::endl
                << "EXMC: number of replicas on each process = "
                << write_vector(nreps_) << std::endl
                << "EXMC: initial inverse temperature set = "
                << write_vector(beta_, " ", 5) << std::endl;
    }

    // initialize walkers
    walker_.resize(nrep_local_);
    tid_local_.resize(nrep_local_);
    alps::Parameters wp(params);
    for (int p = 0; p < nrep_local_; ++p) {
      // different WORKER_SEED for each walker, same DISORDER_SEED for all walkers
      for (int j = 1; j < 3637 /* 509th prime number */; ++j) engine()();
      wp["WORKER_SEED"] = engine()();
      walker_[p] = helper::create_walker(wp, init_);
      tid_local_[p] = p + offset_local_;
    }
    if (comm_.rank() == 0) {
      tid_.resize(nrep);
      for (int p = 0; p < nrep; ++p) tid_[p] = p;
    }

    if (comm_.rank() == 0 && mcs_.exchange()) {
      weight_parameters_.resize(nrep);
      for (int p = 0; p < nrep; ++p) weight_parameters_[p] = weight_parameter_type(0);
    }

    // initialize walker labels
    if (comm_.rank() == 0) {
      wid_.resize(nrep);
      for (int p = 0; p < nrep; ++p) wid_[p] = p;
      if (mcs_.exchange()) {
        direc_.resize(nrep);
        direc_[0] = walker_direc::down;
        for (int p = 1; p < nrep; ++p) direc_[p] = walker_direc::unlabeled;
      }
    }

    // working space
    if (mcs_.exchange()) {
      wp_local_.resize(nrep_local_);
      if (comm_.rank() == 0) {
        wp_.resize(nrep);
        upward_.resize(nrep);
        accept_.resize(nrep - 1);
        if (mcs_.random_exchange()) permutation_.resize(nrep - 1);
      }
    }
  }
  virtual ~parallel_exchange_worker() {}

  void init_observables(alps::Parameters const& params, std::vector<alps::ObservableSet>& obs) {
    int nrep = beta_.size();
    obs.resize(nrep);
    for (int p = 0; p < nrep; ++p)
      helper::init_observables(walker_[0], params, init_, obs[p]);
    if (comm_.rank() == 0) {
      for (int p = 0; p < nrep; ++p) {
        obs[p] << SimpleRealObservable("EXMC: Temperature")
               << SimpleRealObservable("EXMC: Inverse Temperature");
        if (mcs_.exchange()) {
          obs[p] << SimpleRealObservable("EXMC: Ratio of Upward-Moving Walker")
                 << SimpleRealObservable("EXMC: Ratio of Downward-Moving Walker")
                 << SimpleRealObservable("EXMC: Inverse Round-Trip Time");
          obs[p]["EXMC: Ratio of Upward-Moving Walker"].reset(true);
          obs[p]["EXMC: Ratio of Downward-Moving Walker"].reset(true);
          if (p != nrep - 1) {
            obs[p] << SimpleRealObservable("EXMC: Acceptance Rate");
            obs[p]["EXMC: Acceptance Rate"].reset(true);
          }
        }
      }
      if (mcs_.exchange()) obs[0] << SimpleRealObservable("EXMC: Average Inverse Round-Trip Time");
    }
  }

  void run(std::vector<alps::ObservableSet>& obs) {
    ++mcs_;

    int nrep = beta_.size();

    if (comm_.rank() == 0) {
      for (int p = 0; p < nrep; ++p) {
        add_constant(obs[p]["EXMC: Temperature"], 1. / beta_[p]);
        add_constant(obs[p]["EXMC: Inverse Temperature"], beta_[p]);
      }
    }

    // MC update of each replica
    for (int w = 0; w < nrep_local_; ++w) {
      int p = tid_local_[w];
      walker_[w]->set_beta(beta_[p]);
      helper::run_walker(walker_[w], init_, obs[p]);
    }

    // replica exchange process
    if (mcs_.exchange() && (mcs_() % mcs_.interval()) == 0) {

      bool continue_stage = false;
      bool next_stage = false;

      for (int w = 0; w < nrep_local_; ++w) wp_local_[w] = walker_[w]->weight_parameter();
      if (comm_.rank() == 0) {
        std::copy(wp_local_.begin(), wp_local_.begin() + nrep_local_, wp_.begin());
        for (int p = 1; p < nreps_.size(); ++p)
          comm_.recv(p, 0, &wp_[offsets_[p]], nreps_[p]);
        for (int w = 0; w < nrep; ++w) {
          int p = tid_[w];
          weight_parameters_[p] += wp_[w];
        }
      } else {
        comm_.send(0, 0, &wp_local_[0], nrep_local_);
      }

      if (comm_.rank() == 0) {
        if (mcs_.random_exchange()) {
          // random exchange
          for (int p = 0; p < nrep - 1; ++p) permutation_[p] = p;
          alps::random_shuffle(permutation_.begin(), permutation_.end(), generator_01());

          for (int i = 0; i < nrep - 1; ++i) {
            int p = permutation_[i];
            int w0 = wid_[p];
            int w1 = wid_[p+1];
            double logp = ((walker_type::log_weight(wp_[w1], beta_[p]  ) +
                            walker_type::log_weight(wp_[w0], beta_[p+1])) -
                           (walker_type::log_weight(wp_[w1], beta_[p+1]) +
                            walker_type::log_weight(wp_[w0], beta_[p]  )));
            if (logp > 0 || uniform_01() < std::exp(logp)) {
              std::swap(tid_[w0], tid_[w1]);
              std::swap(wid_[p], wid_[p+1]);
              obs[p]["EXMC: Acceptance Rate"] << 1.;
            } else {
              obs[p]["EXMC: Acceptance Rate"] << 0.;
            }
          }
        } else {
          // alternating exchange
          int start = (mcs_() / mcs_.interval()) % 2;
          for (int p = start; p < nrep - 1; p += 2) {
            int w0 = wid_[p];
            int w1 = wid_[p+1];
            double logp = ((walker_type::log_weight(wp_[w1], beta_[p]  ) +
                            walker_type::log_weight(wp_[w0], beta_[p+1])) -
                           (walker_type::log_weight(wp_[w1], beta_[p+1]) +
                            walker_type::log_weight(wp_[w0], beta_[p]  )));
            if (logp > 0 || uniform_01() < std::exp(logp)) {
              std::swap(tid_[w0], tid_[w1]);
              std::swap(wid_[p], wid_[p+1]);
              obs[p]["EXMC: Acceptance Rate"] << 1.;
            } else {
              obs[p]["EXMC: Acceptance Rate"] << 0.;
            }
          }
        }

        int wtop = wid_.front();
        for (int w = 0; w < nrep; ++w) {
          if (w == wtop && direc_[w] == walker_direc::up) {
            obs[w]["EXMC: Inverse Round-Trip Time"] << 1.;
          } else {
            obs[w]["EXMC: Inverse Round-Trip Time"] << 0.;
          }
        }
        if (direc_[wtop] == walker_direc::up) {
          obs[0]["EXMC: Average Inverse Round-Trip Time"] << 1. / nrep;
          ++num_returnee_;
        } else {
          obs[0]["EXMC: Average Inverse Round-Trip Time"] << 0.;
        }
        direc_[wtop] = walker_direc::down;
        if (direc_[wid_.back()] == walker_direc::down) direc_[wid_.back()] = walker_direc::up;
        for (int p = 0; p < nrep; ++p) {
          obs[p]["EXMC: Ratio of Upward-Moving Walker"] <<
            (direc_[wid_[p]] == walker_direc::up ? 1. : 0.);
          obs[p]["EXMC: Ratio of Downward-Moving Walker"] <<
            (direc_[wid_[p]] == walker_direc::down ? 1. : 0.);
        }

        if (mcs_.doing_optimization() && mcs_.stage_count() == mcs_.stage_sweeps()) {

          if (mcs_.optimization_type() == exmc::exchange_steps::rate) {

            for (int p = 0; p < nrep - 1; ++p)
              accept_[p] =
                reinterpret_cast<SimpleRealObservable&>(obs[p]["EXMC: Acceptance Rate"]).mean();
            for (int p = 0; p < nrep; ++p) wp_[p] = weight_parameters_[p] / mcs_.stage_count();
            std::cout << "EXMC stage " << mcs_.stage() << ": acceptance rate = "
                      << write_vector(accept_, " ", 5) << std::endl;

            if (mcs_.stage() != 0) {
              beta_.optimize_h1999<walker_type>(wp_);
              std::cout << "EXMC stage " << mcs_.stage() << ": optimized inverse temperature set = "
                        << write_vector(beta_, " ", 5) << std::endl;
            }
            next_stage = true;

            for (int p = 0; p < nrep - 1; ++p) {
              obs[p]["EXMC: Acceptance Rate"].reset(true);
            }
            for (int p = 0; p < nrep; ++p) {
              obs[p]["EXMC: Ratio of Upward-Moving Walker"].reset(true);
              obs[p]["EXMC: Ratio of Downward-Moving Walker"].reset(true);
              weight_parameters_[p] = weight_parameter_type(0);
            }

          } else {

            bool success = (num_returnee_ >= nrep);

            int nu = 0;
            for (int p = 0; p < nrep; ++p) if (direc_[p] == walker_direc::unlabeled) ++nu;
            if (nu > 0) success = false;

            for (int p = 0; p < nrep; ++p) {
              double up = reinterpret_cast<SimpleRealObservable&>(
                obs[p]["EXMC: Ratio of Upward-Moving Walker"]).mean();
              double down = reinterpret_cast<SimpleRealObservable&>(
                obs[p]["EXMC: Ratio of Downward-Moving Walker"]).mean();
              upward_[p] = (up + down > 0) ? up / (up + down) : alps::nan();
            }

            for (int p = 0; p < nrep - 1; ++p)
              accept_[p] = reinterpret_cast<SimpleRealObservable&>(
                obs[p]["EXMC: Acceptance Rate"]).mean();

            std::cout << "EXMC stage " << mcs_.stage()
                      << ": stage count = " << mcs_.stage_count() << '\n'
                      << "EXMC stage " << mcs_.stage()
                      << ": number of returned walkers = " << num_returnee_ << '\n'
                      << "EXMC stage " << mcs_.stage()
                      << ": number of unlabeled walkers = " << nu << '\n'
                      << "EXMC stage " << mcs_.stage()
                      << ": population ratio of upward-moving walkers "
                      << write_vector(upward_, " ", 5) << '\n'
                      << "EXMC stage " << mcs_.stage()
                      << ": acceptance rate " << write_vector(accept_, " ", 3) << std::endl;

            // preform optimization
            if (mcs_.stage() != 0 && success) success = beta_.optimize2(upward_);

            if (success) {
              std::cout << "EXMC stage " << mcs_.stage() << ": DONE" << std::endl;
              if (mcs_.stage() > 0)
                std::cout << "EXMC stage " << mcs_.stage() << ": optimized inverse temperature set = "
                          << write_vector(beta_, " ", 5) << std::endl;
              next_stage = true;
              for (int p = 0; p < nrep - 1; ++p) {
                obs[p]["EXMC: Acceptance Rate"].reset(true);
              }
              for (int p = 0; p < nrep; ++p) {
                obs[p]["EXMC: Ratio of Upward-Moving Walker"].reset(true);
                obs[p]["EXMC: Ratio of Downward-Moving Walker"].reset(true);
              }
              num_returnee_ = 0;
            } else {
              // increase stage sweeps
              continue_stage = true;
              std::cout << "EXMC stage " << mcs_.stage() << ": NOT FINISHED\n"
                        << "EXMC stage " << mcs_.stage() << ": increased number of sweeps to "
                        << mcs_.stage_sweeps() << std::endl;
            }
          }

          // check whether all the replicas have revisited the highest temperature or not
          if (!mcs_.perform_optimization() && mcs_() == mcs_.thermalization()) {
            int nu = 0;
            for (int p = 0; p < nrep; ++p) if (direc_[p] == walker_direc::unlabeled) ++nu;
            std::cout << "EXMC: thermalization count = " << mcs_() << '\n'
                      << "EXMC: number of returned walkers = " << num_returnee_ << '\n'
                      << "EXMC: number of unlabeled walkers = " << nu << std::endl;
            if ((num_returnee_ >= nrep) && (nu == 0)) {
              std::cout << "EXMC: thermzlization DONE" << std::endl;
            } else {
              continue_stage = true;
              std::cout << "EXMC: thermalization NOT FINISHED\n"
                        << "EXMC: increased number of thermalization sweeps to "
                        << mcs_.thermalization() << std::endl;
            }
          }
        }
      }

      // broadcast EXMC results
      broadcast(comm_, continue_stage, 0);
      broadcast(comm_, next_stage, 0);
      if (continue_stage) mcs_.continue_stage();
      if (next_stage) mcs_.next_stage();
      if (comm_.rank() == 0) {
        for (int p = 1; p < nreps_.size(); ++p)
          comm_.send(p, 0, &tid_[offsets_[p]], nreps_[p]);
        std::copy(tid_.begin(), tid_.begin() + nrep_local_, tid_local_.begin());
      } else {
        comm_.recv(0, 0, &tid_local_[0], nrep_local_);
      }
    }
  }

  void save(alps::ODump& dp) const {
    dp << beta_ << mcs_ << tid_local_;
    if (comm_.rank() == 0) dp << tid_ << wid_ << direc_ << num_returnee_ << weight_parameters_;
    for (int i = 0; i < nrep_local_; ++i) walker_[i]->save(dp);
  }
  void load(alps::IDump& dp) {
    dp >> beta_ >> mcs_ >> tid_local_;
    if (comm_.rank() == 0) dp >> tid_ >> wid_ >> direc_ >> num_returnee_ >> weight_parameters_;
    for (int i = 0; i < nrep_local_; ++i) walker_[i]->load(dp);
  }

  bool is_thermalized() const { return mcs_.is_thermalized(); }
  double progress() const { return mcs_.progress(); }

  static void evaluate_observable(alps::ObservableSet& obs) {
    walker_type::evaluate_observable(obs);
  }

protected:
  std::pair<int, int> calc_nrep(int id) const {
    int nrep = beta_.size();
    int n = nrep / comm_.size();
    int f;
    if (id < nrep - n * comm_.size()) {
      ++n;
      f = n * id;
    } else {
      f = (nrep - n * comm_.size()) + n * id;
    }
    return std::make_pair(n, f);
  }

private:
  boost::mpi::communicator comm_;

  int nrep_local_;           // number of walkers (replicas) on this process
  int offset_local_;         // first (global) id of walker on this process
  int nrep_max_;             // [master only] maximum number of walkers (replicas) on a process
  std::vector<int> nreps_;   // [master only] number of walkers (replicas) on each process
  std::vector<int> offsets_; // [master only] first (global) id of walker on each process

  initializer_type init_;
  std::vector<boost::shared_ptr<walker_type> > walker_; // [0..nrep_local_)

  exmc::inverse_temperature_set beta_;
  exmc::exchange_steps mcs_;
  std::vector<int> tid_local_; // temperature id of each walker (replica)
  std::vector<int> tid_;       // [master only] temperature id of each walker (replica)
  std::vector<int> wid_;       // [master only] walker (replica) id at each temperature
  std::vector<int> direc_;     // [master only] direction of each walker (replica)
  int num_returnee_;           // [master only] number of walkers returned to highest temperature
  std::vector<weight_parameter_type> weight_parameters_; // [master only]

  // working space
  std::vector<weight_parameter_type> wp_local_;
  std::vector<weight_parameter_type> wp_; // [master only]
  std::vector<double> upward_;            // [master only]
  std::vector<double> accept_;            // [master only]
  std::vector<int> permutation_;          // [master only]
};

#endif // ALPS_HAVE_MPI

} // end namespace parapack
} // end namespace alps

#endif // PARAPACK_EXCHANGE_H
