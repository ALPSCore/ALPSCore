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

#include "mc_worker.h"

namespace alps {
namespace parapack {

//
// dumb_worker
//

dumb_worker::~dumb_worker() {}

void dumb_worker::print_copyright(std::ostream& out) {
  out << "ALPS/parapack dumb worker\n";
}

void dumb_worker::init_observables(Parameters const&, ObservableSet&) {}

void dumb_worker::run(ObservableSet&) {}

void dumb_worker::load(IDump&) {}

void dumb_worker::save(ODump&) const {}

bool dumb_worker::is_thermalized() const { return true; }

double dumb_worker::progress() const { return 1; }

//
// mc_worker
//

mc_worker::mc_worker(Parameters const& params)
  : abstract_worker(), rng_helper(params) {
}

mc_worker::~mc_worker() {}

void mc_worker::load_worker(IDump& dump) {
  abstract_worker::load_worker(dump);
  rng_helper::load(dump);
}

void mc_worker::save_worker(ODump& dump) const {
  abstract_worker::save_worker(dump);
  rng_helper::save(dump);
}

} // end namespace parapack
} // end namespace alps
