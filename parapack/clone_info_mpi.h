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

#ifndef PARAPACK_CLONE_INFO_MPI_H
#define PARAPACK_CLONE_INFO_MPI_H

#include "clone_info.h"
#include "process_mpi.h"

namespace alps {

//
// clone_info_mpi
//

class clone_info_mpi : public clone_info {
public:
  // interprocess communication is required
  clone_info_mpi(boost::mpi::communicator const& comm, cid_t cid, Parameters const& params,
    std::string const& base);
private:
  boost::mpi::communicator comm_;
  virtual unsigned int num_processes() const;
  virtual unsigned int process_id() const;
  virtual void set_hosts(std::vector<std::string>& hosts, bool& is_master);
};

} // end namespace alps

#endif // PARAPACK_CLONE_INFO_MPI_H
