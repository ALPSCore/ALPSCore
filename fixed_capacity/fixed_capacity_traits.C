/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2002-2003 by Synge Todo <wistaria@comp-phys.org>
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

#include <alps/fixed_capacity_traits.h>
#include <alps/fixed_capacity_vector.h>

#include <iostream>
#include <list>
#include <queue>
#include <stack>
#include <vector>

template<class T>
class capacity_checker
{
public:
  capacity_checker() {
    output<T, alps::fixed_capacity_traits<T>::capacity_is_fixed>();
  }

private:
  template<class U, bool B> struct output;
  template<class U> struct output<U, true> {
    output() {
      std::cout << "capacity is fixed (static_max_size = "
                << alps::fixed_capacity_traits<U>::static_max_size << ")\n";
    }
  };
  template<class U> struct output<U, false> {
    output() { std::cout << "capacity is not fixed\n"; }
  };

}; // capacity_checker


int main()
{
  std::cout << "T = std::vector<int>\n";
  capacity_checker<std::vector<int> >();
  std::cout << std::endl;

  std::cout << "T = std::list<int>\n";
  capacity_checker<std::list<int> >();
  std::cout << std::endl;

  std::cout << "T = alps::fixed_capacity_vector<int,8>\n";
  capacity_checker<alps::fixed_capacity_vector<int,8> >();
  std::cout << std::endl;

  std::cout << "T = alps::fixed_capacity_deque<int,8>\n";
  capacity_checker<alps::fixed_capacity_deque<int,8> >();
  std::cout << std::endl;

  std::cout << "T = std::stack<int, alps::fixed_capacity_vector<int,4> >\n";
  capacity_checker<std::stack<int, alps::fixed_capacity_vector<int,4> > >();
  std::cout << std::endl;

  std::cout << "T = std::stack<int>\n";
  capacity_checker<std::stack<int> >();
  std::cout << std::endl;

  std::cout << "T = std::queue<int, alps::fixed_capacity_deque<int,6> >\n";
  capacity_checker<std::queue<int, alps::fixed_capacity_deque<int,6> > >();
  std::cout << std::endl;

  std::cout << "T = std::queue<int>\n";
  capacity_checker<std::queue<int> >();
  std::cout << std::endl;

  std::cout 
    << "T = std::priority_queue<int, alps::fixed_capacity_vector<int,16> >\n";
  capacity_checker<std::priority_queue<int, 
    alps::fixed_capacity_vector<int,16> > >();
  std::cout << std::endl;

  std::cout << "T = std::priority_queue<int>\n";
  capacity_checker<std::priority_queue<int> >();
  std::cout << std::endl;

  std::cout << "T = double\n";
  capacity_checker<double>();

  return 0;
}
