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

#ifndef PARAPACK_CLONE_PROXY_H
#define PARAPACK_CLONE_PROXY_H

#include "clone.h"

namespace alps {

class clone_proxy {
public:
  clone_proxy(clone*& clone_ptr, clone_timer::duration_t const& check_interval)
    : clone_ptr_(clone_ptr), interval_(check_interval) {}

  bool is_local(Process const&) const { return true; }

  void start(tid_t tid, cid_t cid, thread_group const&, Parameters const& params,
    boost::filesystem::path const& basedir, std::string const& base, bool is_new) {
    clone_ptr_ = new clone(tid, cid, params, basedir, base, interval_, is_new);
  }

  clone_info const& info(Process const&) const {
    if (!clone_ptr_)
      boost::throw_exception(std::logic_error("clone_proxy::info()"));
    return clone_ptr_->info();
  }

  void checkpoint(Process const&) { if (clone_ptr_) clone_ptr_->checkpoint(); }

  void update_info(Process const&) const {}

  void suspend(Process const&) { if (clone_ptr_) clone_ptr_->suspend(); }

  void halt(Process const&) { /* if (clone_ptr_) clone_ptr_->halt(); */ }

  void destroy(Process const&) {
    if (clone_ptr_) {
      delete clone_ptr_;
      clone_ptr_ = 0;
    }
  }

private:
  clone*& clone_ptr_;
  clone_timer::duration_t interval_;
};

} // end namespace alps

#endif // PARAPACK_CLONE_PROXY_H
