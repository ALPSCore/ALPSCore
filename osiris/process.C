/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2002 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Synge Todo <wistaria@comp-phys.org>
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

/* $Id$ */

#include <alps/osiris/comm.h>
#include <alps/osiris/process.h>
#include <alps/osiris/dump.h>
#include <alps/osiris/std/string.h>
#include <alps/osiris/std/vector.h>

#include <string>
#include <algorithm>
#include <functional>

#ifdef ALPS_MPI
#include <mpi.h>
#endif

namespace alps {

//=======================================================================
// Host
//
// derived from Osiris
//
// is used to reference a Host computer
//-----------------------------------------------------------------------
  
Host::Host()
  : name_(),
    speed_(1.),
    id_(-1)
{
}

Host::Host(int32_t i, const std::string& n, double s)
  : name_(n), 
    speed_(s), 
    id_(i)
{
}

void Host::save(ODump& dump) const
{
  dump << name_ << speed_ << id_;
}

void Host::load(IDump& dump)
{
  dump >> name_ >> speed_ >> id_;
}

bool Host::valid() const
{
  return (id_ >= 0);
}


//=======================================================================
// Process
//
// derived from Host
//
// describes a Process on a specified Host
//-----------------------------------------------------------------------


Process::Process(const Host& h, int32_t i)
  : Host(h),
    tid(i)
{
}


Process::Process(int32_t i)
  : Host(detail::invalid_id()),
    tid(i)
{
}

void Process::save(ODump& dump) const
{
  Host::save(dump);
  dump << tid;
}


void Process::load(IDump& dump)
{
  Host::load(dump);
  dump >> tid;
}


bool Process::local() const
{
  // is it the currently running Process ?
  return (tid==detail::local_id());
}

bool Process::valid() const
{
#ifdef ALPS_PVM

  return (tid >= 0);

#else
#ifdef ALPS_MPI

  int total;
  MPI_Comm_size(MPI_COMM_WORLD,&total);
  return ((tid>=0) && (tid < total));
   
#else

  return (tid==0);
  
#endif
#endif
}

} // end namespace alps
