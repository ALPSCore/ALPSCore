/***************************************************************************
* PALM++/osiris library
*
* osiris/buffer.C      dumps for object serialization
*
* $Id$
*
* Copyright (C) 1994-2002 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Synge Todo <wistaria@comp-phys.org>,
*
* This program is free software; you can redistribute it and/or
* modify it under the terms of the GNU General Public License
* as published by the Free Software Foundation; either version 2
* of the License, or (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
**************************************************************************/

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

} // end namespace detail
} // end namespace alps
