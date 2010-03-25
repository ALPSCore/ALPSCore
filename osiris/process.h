/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2005 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

#ifndef OSIRIS_PROCESS_H
#define OSIRIS_PROCESS_H

#include <alps/config.h>
#include <alps/osiris/std/vector.h>
#include <alps/osiris/dump.h>

#include <cstdlib>
#include <string>

namespace alps {

 
/** a process descriptor. */
    
class Process
{
public:
  
  // CONSTRUCTORS
  
  explicit Process(int); // constructor for process on unknown host
  Process() : tid(-1) {} // invalid Process
  
  // MEMBER FUNCTIONS
  
  void load(IDump&); // load from a dump
  void save(ODump&) const; // save into a dump

  bool valid() const; // is this a valid Process descriptor ?
  bool local() const; // is this the current Process?
  
  inline operator int () const {return tid;}
  
  bool operator==(const Process& p) const
  { return (tid==p.tid);}

  bool operator!=(const Process& p)  const
  { return (tid!=p.tid);}

  /// sorting criterion for two processes?
  bool operator<(const Process& p) const
  { return (tid<p.tid);}

private:
  int tid; // the unique Process id
};

typedef std::vector<Process> ProcessList;
}


#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

inline std::ostream& operator<<(std::ostream& out, const alps::Process& p)
{
  out << int(p);
  return out;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif

#endif // OSIRIS_PROCESS_H
