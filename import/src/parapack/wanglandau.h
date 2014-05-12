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

#ifndef PARAPACK_WANGLANDAU_H
#define PARAPACK_WANGLANDAU_H

#include "exp_number.h"
#include "worker_factory.h"
#include "integer_range.h"
#include "montecarlo.h"
#include <alps/alea.h>
#include <alps/numeric/double2int.hpp>
#include <alps/osiris.h>
#include <boost/filesystem/operations.hpp>

namespace alps {

namespace wanglandau {
  struct learn {};
  struct measure {};
  struct reweight {};

  template<typename WL_TYPE>
  struct is_learning_phase { BOOST_STATIC_CONSTANT(bool, value = false); };
  template<>
  struct is_learning_phase<learn> { BOOST_STATIC_CONSTANT(bool, value = true); };

  template<typename WL_TYPE>
  struct is_measurement_phase { BOOST_STATIC_CONSTANT(bool, value = false); };
  template<>
  struct is_measurement_phase<measure> { BOOST_STATIC_CONSTANT(bool, value = true); };

  template<typename WL_TYPE>
  struct is_reweighting_phase { BOOST_STATIC_CONSTANT(bool, value = false); };
  template<>
  struct is_reweighting_phase<reweight> { BOOST_STATIC_CONSTANT(bool, value = true); };

  typedef std::map<std::string, std::vector<double> > observable_map_type;
}


class wanglandau_weight {
private:
  BOOST_STATIC_CONSTANT(int, unknown = 0);
  BOOST_STATIC_CONSTANT(int, ascending = 1);
  BOOST_STATIC_CONSTANT(int, descending = 2);

public:
  typedef integer_range<int> range_t;

  wanglandau_weight(range_t const& walk)
    : walk_range_(walk), measure_range_(walk), weight_(walk_range_.size(), 1.0),
      last_direction_(unknown), direction_(unknown), penalty_(8) {
  }
  wanglandau_weight(range_t const& walk, range_t const& measure)
    : walk_range_(walk), measure_range_(measure), weight_(walk_range_.size(), 1.0),
      last_direction_(unknown), direction_(unknown), penalty_(8) {
  }

  void init_observables(ObservableSet& obs) const {
    std::vector<std::string> label;
    for (int b = bin_min(); b <= bin_max(); ++b)
      label.push_back(boost::lexical_cast<std::string>(b));
    obs << SimpleRealVectorObservable("Log(g)", label)
        << IntHistogramObservable("Final Histogram", 0, size())
        << IntHistogramObservable("Overall Histogram", 0, size())
        << IntHistogramObservable("Histogram of Ascending Walker", 0, size())
        << IntHistogramObservable("Measurement Histogram", 0, size())
        << SimpleRealObservable("Inverse Round-trip Time")
        << SimpleRealObservable("Lower Bound of Random Walk")
        << SimpleRealObservable("Upper Bound of Random Walk")
        << SimpleRealObservable("Lower Bound of Measurement")
        << SimpleRealObservable("Upper Bound of Measurement");
    obs["Final Histogram"].reset(true);
  }

  void set_penalty(double p) { penalty_ = p; }

  int size() const { return walk_range_.size(); }
  int bin_min() const { return walk_range_.min(); }
  int bin_max() const { return walk_range_.max(); }
  range_t const& walk_range() const { return walk_range_; }
  range_t const& measure_range() const { return measure_range_; }
  int32_t bin2index(int bin) const { return bin - bin_min(); }
  int index2bin(int index) const { return index + bin_min(); }

  exp_double relative_weight(int bin, int bin_new) const {
    if (bin < bin_min())
      return (bin_new >= bin && bin_new <= walk_range_.max()) ? 1.0 : 0.0;
    else if (bin > bin_max())
      return (bin_new >= walk_range_.min() && bin_new <= bin) ? 1.0 : 0.0;
    else
      return walk_range_.is_included(bin_new) ? ((*this)[bin] / (*this)[bin_new]) : exp_double(0);
  }
  exp_double operator[](int bin) const {
    return walk_range_.is_included(bin) ? weight_[bin2index(bin)] : exp_double(0);
  }
  void set_weight(int bin, exp_double w) {
    if (walk_range_.is_included(bin)) {
      weight_[bin2index(bin)] = w;
    } else {
      boost::throw_exception(std::range_error("wanglandau_weight::set_weight()"));
    }
  }

  void visit(ObservableSet& obs, int bin, exp_double const& factor) {
    if (walk_range_.is_included(bin)) {
      if (bin == bin_min())
        direction_ = ascending;
      else if (bin == bin_max())
        direction_ = descending;
      obs["Final Histogram"] << bin2index(bin);
      obs["Overall Histogram"] << bin2index(bin);
      if (direction_ == ascending) obs["Histogram of Ascending Walker"] << bin2index(bin);
    }
    if (walk_range_.is_included(bin)) {
      if (measure_range_.is_included(bin))
        weight_[bin2index(bin)] *= factor;
      else
        weight_[bin2index(bin)] *= pow(factor, penalty_);
    }
  }
  void measure(ObservableSet& obs, int bin) {
    if (measure_range_.is_included(bin))
      obs["Measurement Histogram"] << bin2index(bin);
    if (walk_range_.is_included(bin)) {
      obs["Inverse Round-trip Time"]
        << ((last_direction_ == ascending && direction_ == descending) ? 1.0 : 0.0);
      last_direction_ = direction_;
    }
  }

  void write_observables(ObservableSet& obs) const {
    std::valarray<double> w(size());
    for (int i = 0; i < size(); ++i) w[i] = log(weight_[i]);
    obs["Log(g)"] << w;
    obs["Lower Bound of Random Walk"] << 1.0 * walk_range_.min();
    obs["Upper Bound of Random Walk"] << 1.0 * walk_range_.max();
    obs["Lower Bound of Measurement"] << 1.0 * measure_range_.min();
    obs["Upper Bound of Measurement"] << 1.0 * measure_range_.max();
  }

  static void evaluate_observable(std::vector<ObservableSet>& obs) { evaluate_observable(obs[0]); }
  static void evaluate_observable(ObservableSet& obs) {
    if (obs.has("Inverse Round-trip Time")) {
      RealObsevaluator irt = obs["Inverse Round-trip Time"];
      RealObsevaluator rt("Round-trip Time");
      rt = 1.0 / irt;
      obs.addObservable(rt);
    }
  }

  static void save_weight(std::vector<ObservableSet> const& obs, Parameters const& params) {
    save_weight(obs[0], params);
  }
  static void save_weight(ObservableSet const& obs, Parameters const& params) {
    typedef RealObsevaluator eval_t;
    range_t walk(numeric::double2int(eval_t(obs["Lower Bound of Random Walk"]).mean()),
                 numeric::double2int(eval_t(obs["Upper Bound of Random Walk"]).mean()));
    range_t measure(numeric::double2int(eval_t(obs["Lower Bound of Measurement"]).mean()),
                    numeric::double2int(eval_t(obs["Upper Bound of Measurement"]).mean()));
    std::valarray<double> weight = RealVectorObsevaluator(obs["Log(g)"]).mean();
    std::valarray<double> logg(measure.size());
    for (int bin = measure.min(); bin <= measure.max(); ++bin)
      logg[bin - measure.min()] = weight[bin - walk.min()];
    OXDRFileDump dp(weight_dumpfiles(params)[0]);
    dp << measure << logg;
  }
  void load_weight(Parameters const& params) {
    range_t load_range;
    std::vector<boost::filesystem::path> dumpfiles = weight_dumpfiles(params);
    BOOST_FOREACH(boost::filesystem::path const& path, dumpfiles) {
      IXDRFileDump dp(path);
      range_t r;
      std::valarray<double> logg;
      dp >> r >> logg;
      if (load_range.empty()) {
        // loading first time
        load_range = overlap(measure_range_, r);
        for (int b = load_range.min(); b <= load_range.max(); ++b)
          weight_[bin2index(b)] = exp_value(logg[b - r.min()]);
      } else if (overlap(measure_range_, r).empty()) {
        // no overlap with measure_range, so skip loading
      } else {
        // calculate average difference in the overlapped region
        range_t v = overlap(load_range, r);
        if (v.empty()) {
          std::cerr << "no overlap\n";
          boost::throw_exception(std::runtime_error("wanglandau_weight::load_weight()"));
        }
        int count = 0;
        double diff = 0;
        const double epsilon = 1e-5;
        for (int b = v.min(); b <= v.max(); ++b) {
          if (logg[b - r.min()] > epsilon && log(weight_[bin2index(b)]) > epsilon) {
            ++count;
            diff += logg[b - r.min()] - log(weight_[bin2index(b)]);
          } else if (logg[b - r.min()] > epsilon || log(weight_[bin2index(b)]) > epsilon) {
            std::cerr << "inconsistent weight: " << b << ' ' << logg[b - r.min()] << ' '
                      << log(weight_[bin2index(b)]) << std::endl;
            boost::throw_exception(std::runtime_error("wanglandau_weight::load_weight()"));
          }
        }
        if (count == 0) {
          std::cerr << "no count: " << count << std::endl;
          boost::throw_exception(std::runtime_error("wanglandau_weight::load_weight()"));
        }
        diff /= count;
        // read weight into weight_
        if (diff > 0) {
          for (int b = load_range.min(); b <= load_range.max(); ++b)
            if (log(weight_[bin2index(b)]) > epsilon) weight_[bin2index(b)] *= exp_value(diff);
          diff = 0;
        }
        load_range = unify(load_range, overlap(measure_range_, r));
        for (int b = overlap(measure_range_, r).min(); b <= overlap(measure_range_, r).max(); ++b) {
          if (logg[b - r.min()] > epsilon) {
            if (v.is_included(b)) {
              weight_[bin2index(b)] *= exp_value(logg[b - r.min()] - diff);
              weight_[bin2index(b)] = sqrt(weight_[bin2index(b)]);
            } else {
              weight_[bin2index(b)] = exp_value(logg[b - r.min()] - diff);
            }
          }
        }
      }
    }

    // check whether the input weight covers the whole range
    if (overlap(measure_range_, load_range) != measure_range_) {
      std::cerr << measure_range_ << ' ' << load_range << std::endl;
      boost::throw_exception(std::runtime_error("wanglandau_weight::load_weight()"));
    }
  }

  void save(ODump& dp) const {
    dp << walk_range_ << measure_range_ << weight_ << last_direction_ << direction_ << penalty_;
  }
  void load(IDump& dp) {
    dp >> walk_range_ >> measure_range_ >> weight_ >> last_direction_ >> direction_ >> penalty_;
  }

  static boost::filesystem::path observable_dumpfile(Parameters const& params) {
      boost::filesystem::path basedir = complete(boost::filesystem::path(static_cast<std::string>(params["DIR_NAME"])));
    boost::filesystem::path dumpfile;
    if (params.defined("OBSERVABLE_DUMP_FILE"))
      dumpfile = complete(boost::filesystem::path(static_cast<std::string>(params["OBSERVABLE_DUMP_FILE"])), basedir);
    else
      dumpfile = complete(boost::filesystem::path(static_cast<std::string>(params["BASE_NAME"]) + ".observable"), basedir);
    return dumpfile;
  }

  static std::vector<boost::filesystem::path> weight_dumpfiles(Parameters const& params) {
    boost::filesystem::path basedir = complete(boost::filesystem::path(static_cast<std::string>(params["DIR_NAME"])));
    std::vector<std::string> files;
    if (params.defined("WEIGHT_DUMP_FILE")) {
      read_vector_resize(params["WEIGHT_DUMP_FILE"], files);
    } else {
      files.push_back(params["BASE_NAME"] + ".weight");
    }
    std::vector<boost::filesystem::path> dumpfiles;
    BOOST_FOREACH(std::string const& f, files) {
      dumpfiles.push_back(complete(boost::filesystem::path(f), basedir));
    }
    return dumpfiles;
  }

private:
  range_t walk_range_;
  range_t measure_range_;
  std::vector<exp_double> weight_;
  int last_direction_;
  int direction_;
  double penalty_;
};

} // end namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

inline alps::ODump& operator<<(alps::ODump& dp, alps::wanglandau_weight const& hist) {
  hist.save(dp);
  return dp;
}

inline alps::IDump& operator>>(alps::IDump& dp, alps::wanglandau_weight& hist) {
  hist.load(dp);
  return dp;
}

inline alps::ObservableSet& operator<<(alps::ObservableSet& obs,
  alps::wanglandau_weight const& hist) {
  hist.write_observables(obs);
  return obs;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif


namespace alps {

template<typename WL_TYPE> class wanglandau_steps;

template<>
class wanglandau_steps<wanglandau::learn> {
public:
  typedef mc_steps::range_type range_type;

  wanglandau_steps() {}
  wanglandau_steps(Parameters const& p)
    : mcs_(p), stage_(0), interval_(16384), factor_(std::exp(1.0)), final_(1.00000001), flatness_(0.95) {
    if (p.defined("CHECK_INTERVAL"))
      interval_ = numeric::double2int(evaluate("CHECK_INTERVAL", p));
    if (p.defined("INITIAL_UPDATE_FACTOR"))
      factor_ = evaluate("INITIAL_UPDATE_FACTOR", p);
    if (p.defined("FINAL_UPDATE_FACTOR"))
      final_ = evaluate("FINAL_UPDATE_FACTOR", p);
    if (p.defined("FLATNESS_THRESHOLD"))
      flatness_ = evaluate("FLATNESS_THRESHOLD", p);
    mcs_.set_thermalization(0);
    mcs_.set_sweeps(interval_);
  }

  wanglandau_steps& operator++() { ++mcs_; return *this; }
  wanglandau_steps operator++(int) { wanglandau_steps tmp = *this; ++mcs_; return tmp; }

  unsigned int operator()() const { return mcs_(); }
  bool can_work() const { return mcs_.can_work(); }
  bool is_thermalized() const { return stage_ > 0 && mcs_.is_thermalized(); }
  double progress() const { return mcs_.progress(); }

  int thermalization() const { return mcs_.thermalization(); }
  range_type sweeps() const { return mcs_.sweeps(); }

  exp_double const& factor() const { return factor_; }

  bool check_flatness(ObservableSet& obs, wanglandau_weight const& weight) {
    bool verbose = true;
    int num = 0;
    double mean = 0;
    double hist_min = -1;
    IntHistogramObsevaluator hist = obs["Final Histogram"];
    for (int bin = weight.measure_range().min(); bin < weight.measure_range().max(); ++bin) {
      int index = bin - weight.walk_range().min();
      if (hist[index] > 0) {
        ++num;
        mean += hist[index];
        hist_min = (hist_min < 0) ? hist[index] : std::min(hist_min, 1.0 * hist[index]);
      }
    }
    if (num > 0) mean /= num;
    if (num == 0) {
      if (verbose) std::cout << "stage " << stage_ << ": " << mcs_()
                             << ": no count in the target energy range)\n";
      mcs_.set_sweeps(mcs_.sweeps().min() + interval_);
      return false;
    } else if (hist_min < flatness_ * mean) {
      if (verbose) std::cout << "stage " << stage_ << ": " << mcs_()
                             << ": flatness check FAILED (mean = " << mean << ", Hmin = "
                             << hist_min << ", ratio = " << hist_min / mean << ")\n";
      mcs_.set_sweeps(mcs_.sweeps().min() + interval_);
      return false;
    } else {
      if (verbose) std::cout << "stage " << stage_ << ": " << mcs_()
                             << ": flatness check PASSED (mean = " << mean << ", Hmin = "
                             << hist_min << ", ratio = " << hist_min / mean << ")\n";
      if (factor_ > final_) {
        factor_ = sqrt(factor_);
        if (verbose) std::cout << "stage " << stage_ << ": " << mcs_()
                               << ": update factor is reduced to " << factor_
                               << " (target = " << final_ << ")\n";
        mcs_.set_sweeps(mcs_.sweeps().min() + interval_);
        obs["Final Histogram"].reset(true);
        ++stage_;
        return false;
      } else {
        if (verbose) std::cout << "stage " << stage_ << ": " << mcs_()
                               << ": Wang-Landau optimization done\n";
        ++stage_;
        factor_ = 1;
      }
    }
    return true;
  }
  template<typename COMMUNICATOR>
  bool check_flatness(ObservableSet& obs, wanglandau_weight& weight, COMMUNICATOR& comm) {
    bool finished = false;
    if (comm.rank() == 0) {
      finished = check_flatness(obs, weight);
      broadcast(comm, finished, 0);
    } else {
      broadcast(comm, finished, 0);
      if (!finished)
        mcs_.set_sweeps(mcs_.sweeps().min() + interval_);
    }
    return finished;
  }

  void save(ODump& dp) const {
    dp << mcs_ << stage_ << interval_ << factor_ << final_ << flatness_;
  }
  void load(IDump& dp) {
    dp >> mcs_ >> stage_ >> interval_ >> factor_ >> final_ >> flatness_;
  }

private:
  mc_steps mcs_;
  unsigned int stage_;
  unsigned int interval_;
  exp_double factor_;
  exp_double final_;
  double flatness_;
};

template<>
class wanglandau_steps<wanglandau::measure> : public mc_steps {
public:
  wanglandau_steps() : mc_steps() {}
  wanglandau_steps(Parameters const& p) : mc_steps(p) {}
  exp_double factor() const { return exp_double(1); }
  bool check_flatness(ObservableSet&, wanglandau_weight const&, bool /* verbose */ = true) { return true; }
};

} // end namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

template<typename WL_TYPE>
alps::ODump& operator<<(alps::ODump& dp, alps::wanglandau_steps<WL_TYPE> const& mcs) {
  mcs.save(dp);
  return dp;
}

template<typename WL_TYPE>
alps::IDump& operator>>(alps::IDump& dp, alps::wanglandau_steps<WL_TYPE>& mcs) {
  mcs.load(dp);
  return dp;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif

namespace alps {

template<typename WL_TYPE>
class wanglandau_evaluator : public parapack::abstract_evaluator {
public:
  wanglandau_evaluator(Parameters const& params) : params_(params), count_(0) {}
  void load(std::vector<ObservableSet> const& obs_in,
            std::vector<ObservableSet>& obs_out) {
    if (wanglandau::is_learning_phase<WL_TYPE>::value) {
      // learning phase
      if (obs_out.size() != obs_in.size()) obs_out.resize(obs_in.size());
      for (int i = 0; i < obs_in.size(); ++i) obs_out[i] << obs_in[i];
    } else {
      // measurement phase
      if (obs_out.size() != 1) obs_out.resize(1);
      int size = obs_in.size();
      wanglandau::observable_map_type obs_map;
      IntHistogramObsevaluator hist = obs_in[0]["Measurement Histogram"];
      if (hist.count() > 0) {
        obs_map["Measurement Histogram"] = std::vector<double>();
        std::vector<double>& hist_array = obs_map["Measurement Histogram"];
        hist_array.resize(size);
        for (int i = 0; i < size; ++i) hist_array[i] = hist[i];
        for (std::map<std::string, Observable*>::const_iterator itr = obs_in[0].begin();
             itr != obs_in[0].end(); ++itr) {
          std::string name = itr->second->name();
          if (obs_in[0].has(name) && obs_in[1].has(name)) {
            obs_map[name] = std::vector<double>();
            std::vector<double>& obs_array = obs_map[name];
            obs_array.resize(size);
            for (int i = 0; i < size; ++i) {
              if (dynamic_cast<const RealObsevaluator*>(&obs_in[i][name])!=0 ||
                  dynamic_cast<const RealObservable*>(&obs_in[i][name])!=0 ||
                  dynamic_cast<const SimpleRealObservable*>(&obs_in[i][name])!=0) {
                RealObsevaluator eval = obs_in[i][name];
                obs_array[i] = (eval.count() ? eval.mean() : 0.0);
              } else {
                obs_array[i] = 0.0;
              }
            }
          } else {
            obs_out[0] << obs_in[0][name];
          }
        }
        OXDRFileDump dp(wanglandau_weight::observable_dumpfile(params_), (count_ > 0));
        dp << true << obs_map;
        ++count_;
      } else {
        std::cout << "No count in observable. Skipped. ";
      }
    }
  }
  void evaluate(std::vector<ObservableSet>& obs) const {
    wanglandau_weight::evaluate_observable(obs);
    if (wanglandau::is_learning_phase<WL_TYPE>::value) {
      // learning phase
      wanglandau_weight::save_weight(obs, params_);
    } else {
      // measurement phase
      OXDRFileDump dp(wanglandau_weight::observable_dumpfile(params_), (count_ > 0));
      dp << false;
    }
  }
private:
  Parameters params_;
  unsigned int count_;
};

} // end namespace alps

#endif // PARAPACK_WANGLANDAU_H
