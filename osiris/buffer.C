/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2004 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

#include <alps/osiris/buffer.h>
#include <alps/osiris/dump.h>
#include <alps/osiris/std/vector.h>

#include <boost/throw_exception.hpp>
#include <stdexcept>

namespace alps {
namespace detail {

// write a few bytes
void Buffer::write_buffer(const void* p, size_type n)
{
  size_type write_pos=size();
  resize(write_pos+n);
  memcpy(&((*this)[write_pos]),p,n);
}


// read a few bytes and update position
void Buffer::read_buffer(void* p, size_type n)
{
  if(read_pos+n>size())
    boost::throw_exception(std::runtime_error("read past buffer"));
  memcpy(p,&((*this)[read_pos]),n);
  read_pos+=n;
}

} // end namespace detail
} // end namespace alps
