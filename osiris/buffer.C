/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2003 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

// load Buffer from dump

Buffer::Buffer(IDump& dump)
{
  load(dump);
}

// save the Buffer contents

void Buffer::save(ODump& dump) const
{
  dump << static_cast<const std::vector<uint8_t>&>(*this) << read_pos << write_pos;
}


// load the Buffer contents

void Buffer::load(IDump& dump)
{
  dump >> static_cast<std::vector<uint8_t>&>(*this) >> read_pos >> write_pos;
}


// write a few bytes and update length

void Buffer::write_buffer(const void* p, size_type n)
{
  if(write_pos+n>size())
    resize((((write_pos+n)/buffer_grow_steps)+1)*buffer_grow_steps);
  memcpy(&((*this)[write_pos]),p,n);
  write_pos+=n;
}


// read a few bytes and update position

void Buffer::read_buffer(void* p, size_type n)
{
  if(read_pos+n>size())
    boost::throw_exception(std::runtime_error("read past buffer"));
  memcpy(p,&((*this)[read_pos]),n);
  read_pos+=n;
}

#ifndef BOOST_NO_INCLASS_MEMBER_INITIALIZATION
const Buffer::size_type Buffer::buffer_grow_steps;
#endif
} // end namespace detail
} // end namespace alps
