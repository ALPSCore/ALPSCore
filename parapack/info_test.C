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

#include <alps/parapack/clone_info.h>
#include <alps/osiris/comm.h>
#if defined(ALPS_HAVE_UNISTD_H)
# include <unistd.h>
#elif defined(ALPS_HAVE_WINDOWS_H)
# include <windows.h>
#endif

int main(int argc, char **argv) {
  alps::comm_init(argc, argv);
  alps::Parameters params;
  params["SEED"] = 29832;
  alps::clone_info info(0, params, "info_test");
  info.start("test 1");
  #if defined(ALPS_HAVE_UNISTD_H)
    sleep(1); // sleep 1 Sec
  #elif defined(ALPS_HAVE_WINDOWS_H)
    Sleep(1000); // sleep 1000 mSec
  #endif
  info.stop();
  #if defined(ALPS_HAVE_UNISTD_H)
    sleep(1); // sleep 1 Sec
  #elif defined(ALPS_HAVE_WINDOWS_H)
    Sleep(1000); // sleep 1000 mSec
  #endif
  info.start("test 2");
  #if defined(ALPS_HAVE_UNISTD_H)
    sleep(1); // sleep 1 Sec
  #elif defined(ALPS_HAVE_WINDOWS_H)
    Sleep(1000); // sleep 1000 mSec
  #endif
  info.stop();
  info.set_progress(0.593483);
  if (alps::is_master()) {
    alps::oxstream oxs;
    oxs << info;
  }
  info.set_progress(1);
  if (alps::is_master()) {
    alps::oxstream oxs;
    oxs << info;
  }
  alps::comm_exit();
  return 0;
}
