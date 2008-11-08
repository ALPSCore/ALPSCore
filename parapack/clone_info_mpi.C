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

#include "clone_info_mpi.h"

namespace mpi = boost::mpi;

namespace alps {

clone_info_mpi::clone_info_mpi(mpi::communicator const& comm, cid_t cid,
  Parameters const& params, std::string const& dump) :
  clone_info(cid, params, dump, false), comm_(comm) {
  clone_info::init(params, dump);
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

} // namespace alps
