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

#ifndef PARAPACK_FILELOCK_H
#define PARAPACK_FILELOCK_H

#include <alps/config.h>

#include <boost/filesystem/path.hpp>

namespace alps {

class filelock {
public:
  filelock();
  filelock(boost::filesystem::path const& file, bool lock_now = false, int wait = -1,
    bool auto_release = true);
  ~filelock();

  void set_file(boost::filesystem::path const& file);

  void lock(int wait = -1); // wait = -1 means waiting forever
  void release();

  bool locking() const;
  bool locked() const;

private:
  std::string file_;
  boost::filesystem::path lock_;
  bool auto_release_;
  bool is_locking_;
};

} // end namespace alps

#endif // PARAPACK_FILELOCK_H
