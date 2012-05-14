/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2003 by Synge Todo <wistaria@comp-phys.org>
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
#include <cstddef>
#include <iostream>

int main() {

#define DO_TYPE(T) \
  std::cout << "size of "#T" is " << sizeof(T) << std::endl;

  DO_TYPE(bool)
  DO_TYPE(char)
  DO_TYPE(short)
  DO_TYPE(int)
  DO_TYPE(long)
  DO_TYPE(long long)
  DO_TYPE(float)
  DO_TYPE(double)
  DO_TYPE(long double)

  DO_TYPE(alps::int8_t)
  DO_TYPE(alps::int16_t)
  DO_TYPE(alps::int32_t)
  DO_TYPE(alps::int64_t)

  DO_TYPE(std::size_t)
  DO_TYPE(std::ptrdiff_t)

  return 0;
}
