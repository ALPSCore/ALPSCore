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

#include "rng_helper.h"

namespace alps {

rng_helper::rng_helper(const Parameters& p) {
  int nr = max_threads();
  engines_.resize(nr);
  generators_.resize(nr);
#if !(defined(__APPLE_CC__) && __GNUC__ == 4 && __GNUC_MINOR__ == 2)
  // g++ with -fopenmp on Mac OS X Snow Leopard crashes with the following directive
  #pragma omp parallel num_threads(nr)
#endif
  {
    for (int r = 0; r < nr; ++r) {
      if (r == thread_id()) {
        engines_[r].reset(rng_factory.create(p.value_or_default("RNG", "mt19937")));
        generators_[r].reset(new generator_type(*engines_[r], boost::uniform_real<>()));
      }
    }
  }
  init(p);
}

void rng_helper::init(const Parameters& p) {
  seed = static_cast<uint32_t>(p["WORKER_SEED"]);
  disorder_seed = static_cast<uint32_t>(p["DISORDER_SEED"]);
  alps::pseudo_des inigen(seed);
  for (int r = 0; r < engines_.size(); ++r) {
    engines_[r]->seed(inigen);
  }
  Disorder::seed(disorder_seed);
}

void rng_helper::load(IDump& dp) {
  std::string state;
  dp >> seed >> disorder_seed;
  for (int r = 0; r < engines_.size(); ++r) {
    dp >> state;
    std::stringstream rngstream(state);
    engines_[r]->read_all(rngstream);
  }
  Disorder::seed(disorder_seed);
}

void rng_helper::save(ODump& dp) const {
  dp << seed << disorder_seed;
  std::ostringstream rngstream;
  for (int r = 0; r < engines_.size(); ++r) {
    engines_[r]->write_all(rngstream);
    dp << rngstream.str();
  }
}

} // end namespace alps
