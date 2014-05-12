/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2010 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

#include <alps/config.h>

#ifdef ALPS_HAVE_MPI
# undef SEEK_SET
# undef SEEK_CUR
# undef SEEK_END  
# include <mpi.h>
#endif
#include <alps/osiris/comm.h>
#include <alps/osiris/process.h>
#include <alps/osiris/dump.h>
#include <string>
#include <algorithm>
#include <functional>

namespace alps {

Process::Process(int i)
 : tid(i)
{
}

void Process::save(ODump& dump) const
{
  dump << tid;
}


void Process::load(IDump& dump)
{
  dump >> tid;
}


bool Process::local() const
{
  return (tid==detail::local_id());
}

bool Process::valid() const
{
#ifdef ALPS_HAVE_MPI

  int total;
  MPI_Comm_size(MPI_COMM_WORLD,&total);
  return ((tid>=0) && (tid < total));

#else

  return (tid==0);

#endif
}

} // end namespace alps
