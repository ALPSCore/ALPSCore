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

#include <alps/fixed_capacity_deque.h>
#include <iostream>
#include <list>
#include <vector>

#ifdef BOOST_NO_ARGUMENT_DEPENDENT_LOOKUP
using namespace alps;
#endif

int main()
{
  typedef alps::fixed_capacity_deque<float,6> Array;

  std::cout << "initialize fixed_capacity_deque of size = 3\n";
  Array a(3);
  
  std::cout << "assign(2)\n";
  a.assign(2);
  
  std::cout << "push_back(4)\n";
  a.push_back(4);
  
  std::cout << "a.erase(a.begin() + 1)\n";
  a.erase(a.begin() + 1);

  std::cout << "a.insert(a.begin() + 2, 5)\n";
  a.insert(a.begin() + 2, 5);
  
  std::cout << "a.insert(a.begin() + 1, 2, 6)\n";
  a.insert(a.begin() + 1, 2, 6);
  
  std::cout << "a.pop_back()\n";
  a.pop_back();
  
  std::cout << "size:     " << a.size() << std::endl;
  std::cout << "empty:    " << (a.empty() ? "true" : "false") << std::endl;
  std::cout << "max_size: " << a.max_size() << std::endl;
  std::cout << "front:    " << a.front() << std::endl;
  std::cout << "back:     " << a.back() << std::endl;

  std::cout << "elems:    ";
  for (Array::iterator pos = a.begin(); pos != a.end(); ++pos) {
    std::cout << *pos << ' ';
  }
  std::cout << std::endl;
  
  std::cout << "a.pop_back() two times\n";
  a.pop_back();
  a.pop_back();

  std::cout << "a.push_front(1) three times\n";
  a.push_front(1);
  a.push_front(1);
  a.push_front(1);

  std::cout << "elems:    ";
  for (Array::const_iterator pos = a.begin(); pos != a.end(); ++pos) {
    std::cout << *pos << ' ';
  }
  std::cout << std::endl;

  std::cout << "a.erase(a.begin(), a.end())\n";
  a.erase(a.begin(), a.end());

  std::cout << "elems:    ";
  for (Array::const_iterator pos = a.begin(); pos != a.end(); ++pos) {
    std::cout << *pos << ' ';
  }
  std::cout << std::endl;

  std::cout << "resize(2,1)\n";
  a.resize(2, 1);

  // insert from std::vector
  std::cout << "insert of std::vector (2,3,4,5) at begin() + 1\n";
  std::vector<float> vec;
  vec.push_back(2);
  vec.push_back(3);
  vec.push_back(4);
  vec.push_back(5);
  a.insert(a.begin() + 1, vec.begin(), vec.end());
  
  std::cout << "elems:    ";
  for (Array::const_iterator pos = a.begin(); pos != a.end(); ++pos) {
    std::cout << *pos << ' ';
  }
  std::cout << std::endl;

  std::cout << "erase(begin()+3, begin()+6)\n";
  a.erase(a.begin()+3, a.begin()+6);

  // insert from std::list
  std::cout << "insert of std::list (6,7,8) at begin()\n";
  std::list<float> lst;
  lst.push_back(7);
  lst.push_back(8);
  lst.push_front(6);
  a.insert(a.begin(), lst.begin(), lst.end());
  
  std::cout << "elems:    ";
  for (Array::const_iterator pos = a.begin(); pos != a.end(); ++pos) {
    std::cout << *pos << ' ';
  }
  std::cout << std::endl;

  // check copy constructor and assignment operator
  Array b(a);
  Array c;
  c = a;
  if (a==b && a==c) {
    std::cout << "copy construction and copy assignment are OK"
              << std::endl;
  }
  else {
    std::cout << "copy construction and copy assignment FAILED"
              << std::endl;
  }
  
  // check copy constructor and assignment operator
  Array d(a);
  Array e;
  e = a;
  if (a == d && a==e) {
    std::cout << "copy construction and copy assignment for different size are OK\n";
  }
  else {
    std::cout << "copy construction and copy assignment for different size FAILED\n";
  }

  std::cout << "elems in reverse order:    ";
  Array::const_reverse_iterator iter_e = a.rend();
  for (Array::const_reverse_iterator pos = a.rbegin();
       pos != iter_e; ++pos) {
    std::cout << *pos << ' ';
  }
  std::cout << std::endl;

  b.clear();
  b.push_front(12);
  b.push_front(11);
  b.push_front(10);

  std::cout << "a elems:    ";
  for (Array::const_iterator pos = a.begin(); pos != a.end(); ++pos) {
    std::cout << *pos << ' ';
  }
  std::cout << std::endl;

  std::cout << "b elems:    ";
  for (Array::const_iterator pos = b.begin(); pos != b.end(); ++pos) {
    std::cout << *pos << ' ';
  }
  std::cout << std::endl;

  std::cout << "swap a and b\n";
  swap(a,b);

  std::cout << "a elems:    ";
  for (Array::const_iterator pos = a.begin(); pos != a.end(); ++pos) {
    std::cout << *pos << ' ';
  }
  std::cout << std::endl;

  std::cout << "b elems:    ";
  for (Array::const_iterator pos = b.begin(); pos != b.end(); ++pos) {
    std::cout << *pos << ' ';
  }
  std::cout << std::endl;

  std::cout << "swap again\n";
  swap(a,b);
    
  std::cout << "a elems:    ";
  for (Array::const_iterator pos = a.begin(); pos != a.end(); ++pos) {
    std::cout << *pos << ' ';
  }
  std::cout << std::endl;

  std::cout << "b elems:    ";
  for (Array::const_iterator pos = b.begin(); pos != b.end(); ++pos) {
    std::cout << *pos << ' ';
  }
  std::cout << std::endl;

  return 0;
}
