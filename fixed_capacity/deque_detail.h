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

#ifndef ALPS_FIXED_CAPACITY_DEQUE_DETAIL_H
#define ALPS_FIXED_CAPACITY_DEQUE_DETAIL_H

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>

namespace alps {

template<class T, std::size_t N, class C>
template<class InputIterator>
void fixed_capacity_deque<T, N, C>::insert_dispatch(
  typename fixed_capacity_deque<T, N, C>::iterator pos, InputIterator first,
  InputIterator last, std::random_access_iterator_tag) {
  // for random access iterator
  const size_type n = last - first;
  T* const p = &(*pos);
  checker::capacity_check(size() + n);
  checker::pointer_check(base(), first_, last_, p);
  if (2 * size_type(pos - begin()) <= size()) {
    // move elements after pos
    if (last_ >= p) {
      if (last_ + n > top()) {
        if (p + n > top()) {
          std::uninitialized_copy(p, last_, p + n - M);
          
          std::copy(first, first + (last_ - p), p);
          first += last_ - p;
          std::uninitialized_copy(first, first + (top() - last_), last_);
          first += top() - last_;
          std::uninitialized_copy(first, last, base());
        } else {
          if (p + n > last_) {
            std::uninitialized_copy(top() - n, last_, base());
            std::uninitialized_copy(p, top() - n, p + n);
            
            std::copy(first, first + (last_ - p), p);
            first += last_ - p;
            std::uninitialized_copy(first, last, last_);
          } else {
            std::uninitialized_copy(top() - n, last_, base());
            std::uninitialized_copy(last_ - n, top() - n, last_);
            std::copy_backward(p, last_ - n, last_);
            
            std::copy(first, last, p);
          }
        }
      } else {
        if (p + n > last_) {
          std::uninitialized_copy(p, last_, p + n);
          
          std::copy(first, first + (last_ - p), p);
          first += last_ - p;
          std::uninitialized_copy(first, last, last_);
        } else {
          std::uninitialized_copy(last_ - n, last_, last_);
          std::copy_backward(p, last_ - n, last_);
          
          std::copy(first, last, p);
        }
      }
    } else {
      if (base() + n > last_) {
        if (p + n - M > last_) {
          std::uninitialized_copy(base(), last_, base() + n);
          std::uninitialized_copy(p, top(), p + n - M);
          
          std::copy(first, first + (top() - p), p);
          first += top() - p;
          std::copy(first, first + (last_ - base()), base());
          first += last_ - base();
          std::uninitialized_copy(first, last, last_);
        } else {
          if (p + n - M > base()) {
            std::uninitialized_copy(base(), last_, base() + n);
            std::uninitialized_copy(last_ + M - n, top(), last_);
            std::copy(p, last_ + M - n, p + n - M);
            
            std::copy(first, first + (top() - p), p);
            first += top() - p;
            std::copy(first, last, base());
          } else {
            std::uninitialized_copy(base(), last_, base() + n);
            std::uninitialized_copy(last_ + M - n, top(), last_);
            std::copy(top() - n, last_ + M - n, base());
            std::copy_backward(p, top() - n, top());
            
            std::copy(first, last, p);
          }
        }
      } else {
        if (p + n - M > base()) {
          std::uninitialized_copy(last_ - n, last_, last_);
          std::copy_backward(base(), last_ - n, last_);
          std::copy(p, top(), p + n - M);
          
          std::copy(first, first + (top() - p), p);
          first += top() - p;
          std::copy(first, last, base());
        } else {
          std::uninitialized_copy(last_ - n, last_, last_);
          std::copy_backward(base(), last_ - n, last_);
          std::copy(top() - n, top(), base());
          std::copy_backward(p, top() - n, top());
          
          std::copy(first, last, p);
        }
      }
    }
    last_ += n;
    if (last_ >= top()) last_ -= M;
  } else {
    // move elements before pos
    if (p >= first_) {
      if (first_ - n < base()) {
        if (p - n < base()) {
          std::uninitialized_copy(first_, p, first_ - n + M);
          
          std::uninitialized_copy(first, first + (base() - p + n), p - n + M);
          first += base() - p + n;
          std::uninitialized_copy(first, first + (first_ - base()), base());
          first += first_ - base();
          std::copy(first, last, first_);
        } else {
          if (p - n < first_) {
            std::uninitialized_copy(first_, base() + n, first_ - n + M);
            std::uninitialized_copy(base() + n, p, base());
            
            std::uninitialized_copy(first, first + (first_ - p + n), p - n);
            first += (first_ - p + n);
            std::copy(first, last, first_);
          } else {
            std::uninitialized_copy(first_, base() + n, first_ - n + M);
            std::uninitialized_copy(base() + n, first_ + n, base());
            std::copy(first_ + n, p, first_);
            
            std::copy(first, last, p - n);
          }
        }
      } else {
        if (p - n < first_) {
          std::uninitialized_copy(first_, p, first_ - n);
          
          std::uninitialized_copy(first, first + (first_ - p + n), p - n);
          first += (first_ - p + n);
          std::copy(first, last, first_);
        } else {
          std::uninitialized_copy(first_, first_ + n, first_ - n);
          std::copy(first_ + n, p, first_);
          
          std::copy(first, last, p - n);
        }
      }
    } else {
      if (top() - n < first_) {
        if (p - n + M < first_) {
          std::uninitialized_copy(first_, top(), first_ - n);
          std::uninitialized_copy(base(), p, top() - n);
          
          std::uninitialized_copy(first, first + (first_ - p + n - M),
                                  p - n + M);
          first += (first_ - p + n - M);
          std::copy(first, first + (top() - first_), first_);
          first += (top() - first_);
          std::copy(first, last, base());
        } else {
          if (p - n < base()) {
            std::uninitialized_copy(first_, top(), first_ - n);
            std::uninitialized_copy(base(), first_ + n - M, top() - n);
            std::copy(first_ + n - M, p, first_);
            
            std::copy(first, first + (base() - p + n), p - n + M);
            first += (base() - p + n);
            std::copy(first, last, base());
          } else {
            std::uninitialized_copy(first_, top(), first_ - n);
            std::uninitialized_copy(base(), first_ + n - M, top() - n);
            std::copy(first_ + n - M, base() + n, first_);
            std::copy(base() + n, p, base());
            
            std::copy(first, last, p - n);
          }
        }
      } else {
        if (p - n < base()) {
          std::uninitialized_copy(first_, first_ + n, first_ - n);
          std::copy(first_ + n, top(), first_);
          std::copy(base(), p, top() - n);
          
          std::copy(first, first + (base() - p + n), p - n + M);
          first += (base() - p + n);
          std::copy(first, last, base());
        } else {
          std::uninitialized_copy(first_, first_ + n, first_ - n);
          std::copy(first_ + n, top(), first_);
          std::copy(base(), base() + n, top() - n);
          std::copy(base() + n, p, base());
          
          std::copy(first, last, p - n);
        }
      }
    }
    first_ -= n;
    if (first_ < base()) first_ += M;
  }
}

template<class T, std::size_t N, class C>
typename fixed_capacity_deque<T, N, C>::iterator
fixed_capacity_deque<T, N, C>::insert_n(
  typename fixed_capacity_deque<T, N, C>::iterator pos, size_type n, const T& x) {
  T* const p = &(*pos);
  checker::capacity_check(size() + n);
  checker::pointer_check(base(), first_, last_, p);
  if (2 * size_type(pos - begin()) <= size()) {
    // move elements after pos
    if (last_ >= p) {
      if (last_ + n > top()) {
        if (p + n > top()) {
          std::uninitialized_copy(p, last_, p + n - M);
          
          std::fill(p, last_, x);
          std::uninitialized_fill(last_, top(), x);
          std::uninitialized_fill(base(), p + n - M, x);
        } else {
          if (p + n > last_) {
            std::uninitialized_copy(top() - n, last_, base());
            std::uninitialized_copy(p, top() - n, p + n);
            
            std::fill(p, last_, x);
            std::uninitialized_fill(last_, p + n, x);
          } else {
            std::uninitialized_copy(top() - n, last_, base());
            std::uninitialized_copy(last_ - n, top() - n, last_);
            std::copy_backward(p, last_ - n, last_);
            
            std::fill(p, p + n, x);
          }
        }
      } else {
        if (p + n > last_) {
          std::uninitialized_copy(p, last_, p + n);
          
          std::fill(p, last_, x);
          std::uninitialized_fill(last_, p + n, x);
        } else {
          std::uninitialized_copy(last_ - n, last_, last_);
          std::copy_backward(p, last_ - n, last_);
          
          std::fill(p, p + n, x);
        }
      }
    } else {
      if (base() + n > last_) {
        if (p + n - M > last_) {
          std::uninitialized_copy(base(), last_, base() + n);
          std::uninitialized_copy(p, top(), p + n - M);
          
          std::fill(p, top(), x);
          std::fill(base(), last_, x);
          std::uninitialized_fill(last_, p + n - M, x);
        } else {
          if (p + n - M > base()) {
            std::uninitialized_copy(base(), last_, base() + n);
            std::uninitialized_copy(last_ + M - n, top(), last_);
            std::copy(p, last_ + M - n, p + n - M);
            
            std::fill(p, top(), x);
            std::fill(base(), p + n - M, x);
          } else {
            std::uninitialized_copy(base(), last_, base() + n);
            std::uninitialized_copy(last_ + M - n, top(), last_);
            std::copy(top() - n, last_ + M - n, base());
            std::copy_backward(p, top() - n, top());
            
            std::fill(p, p + n, x);
          }
        }
      } else {
        if (p + n - M > base()) {
          std::uninitialized_copy(last_ - n, last_, last_);
          std::copy_backward(base(), last_ - n, last_);
          std::copy(p, top(), p + n - M);
          
          std::fill(p, top(), x);
          std::fill(base(), p + n - M, x);
        } else {
          std::uninitialized_copy(last_ - n, last_, last_);
          std::copy_backward(base(), last_ - n, last_);
          std::copy(top() - n, top(), base());
          std::copy_backward(p, top() - n, top());
          
          std::fill(p, p + n, x);
        }
      }
    }
    last_ += n;
    if (last_ >= top()) last_ -= M;
    return pos;
  } else {
    // move elements before pos
    if (p >= first_) {
      if (first_ - n < base()) {
        if (p - n < base()) {
          std::uninitialized_copy(first_, p, first_ - n + M);
          
          std::uninitialized_fill(p - n + M, top(), x);
          std::uninitialized_fill(base(), first_, x);
          std::fill(first_, p, x);
        } else {
          if (p - n < first_) {
            std::uninitialized_copy(first_, base() + n, first_ - n + M);
            std::uninitialized_copy(base() + n, p, base());
            
            std::uninitialized_fill(p - n, first_, x);
            std::fill(first_, p, x);
          } else {
            std::uninitialized_copy(first_, base() + n, first_ - n + M);
            std::uninitialized_copy(base() + n, first_ + n, base());
            std::copy(first_ + n, p, first_);
            
            std::fill(p - n, p, x);
          }
        }
      } else {
        if (p - n < first_) {
          std::uninitialized_copy(first_, p, first_ - n);
          
          std::uninitialized_fill(p - n, first_, x);
          std::fill(first_, p, x);
        } else {
          std::uninitialized_copy(first_, first_ + n, first_ - n);
          std::copy(first_ + n, p, first_);
          
          std::fill(p - n, p, x);
        }
      }
    } else {
      if (top() - n < first_) {
        if (p - n + M < first_) {
          std::uninitialized_copy(first_, top(), first_ - n);
          std::uninitialized_copy(base(), p, top() - n);
          
          std::uninitialized_fill(p - n + M, first_, x);
          std::fill(first_, top(), x);
          std::fill(base(), p, x);
        } else {
          if (p - n < base()) {
            std::uninitialized_copy(first_, top(), first_ - n);
            std::uninitialized_copy(base(), first_ + n - M, top() - n);
            std::copy(first_ + n - M, p, first_);
            
            std::fill(p - n + M, top(), x);
            std::fill(base(), p, x);
          } else {
            std::uninitialized_copy(first_, top(), first_ - n);
            std::uninitialized_copy(base(), first_ + n - M, top() - n);
            std::copy(first_ + n - M, base() + n, first_);
            std::copy(base() + n, p, base());
            
            std::fill(p - n, p, x);
          }
        }
      } else {
        if (p - n < base()) {
          std::uninitialized_copy(first_, first_ + n, first_ - n);
          std::copy(first_ + n, top(), first_);
          std::copy(base(), p, top() - n);
          
          std::fill(p - n + M, top(), x);
          std::fill(base(), p, x);
        } else {
          std::uninitialized_copy(first_, first_ + n, first_ - n);
          std::copy(first_ + n, top(), first_);
          std::copy(base(), base() + n, top() - n);
          std::copy(base() + n, p, base());
          
          std::fill(p - n, p, x);
        }
      }
    }
    first_ -= n;
    if (first_ < base()) first_ += M;
    return pos - n;
  }
}

template<class T, std::size_t N, class C>
typename fixed_capacity_deque<T, N, C>::iterator
fixed_capacity_deque<T, N, C>::erase_n(
  typename fixed_capacity_deque<T, N, C>::iterator pos, size_type n) {
  T* const p = &(*pos);
  checker::pointer_check(base(), first_, last_, p);
  checker::pointer_check(base(), first_, last_, &(*(pos + n)));
  if (size_type(pos - begin()) <= size_type(end() - pos) - n) {
    // move before pos
    if (p + n <= top()) {
      if (p - first_ >= 0) {
        std::copy_backward(first_, p, p + n);
        destroy(first_, first_ + n);
      } else {
        if (first_ + n < top()) {
          std::copy_backward(base(), p, p + n);
          std::copy(top() - n, top(), base());
          std::copy_backward(first_, top() - n, top());
          destroy(first_, first_ + n);
        } else {
          std::copy_backward(base(), p, p + n);
          std::copy(first_, top(), first_ + n - M);
          destroy(base(), first_ + n - M);
          destroy(first_, top());
        }
      }
    } else {
      if (first_ + n > top()) {
        std::copy(first_, p, first_ + n - M);
        destroy(base(), first_ + n - M);
        destroy(first_, top());
      } else {
        std::copy(top() - n, p, base());
        std::copy_backward(first_, top() - n, top());
        destroy(first_, first_ + n);
      }
    }
    first_ += n;
    if (first_ >= top()) first_ -= M;
    return pos + n;
  } else {
    // move after pos
    if (p + n <= top()) {
      if (last_ - p >= 0) {
        std::copy(p + n, last_, p);
        destroy(last_ - n, last_);
      } else {
        if (last_ - n >= base()) {
          std::copy(p + n, top(), p);
          std::copy(base(), base() + n, top() - n);
          std::copy(base() + n, last_, base());
          destroy(last_ - n, last_);
        } else {
          std::copy(p + n, top(), p);
          std::copy(base(), last_, top() - n);
          destroy(base(), last_);
          destroy(last_ - n + M, top());
        }
      }
    } else {
      if (last_ - n > base()) {
        std::copy(p + n - M, base() + n, p);
        std::copy(base() + n, last_, base());
        destroy(last_ - n, last_);
      } else {
        std::copy(p + n - M, last_, p);
        destroy(base(), last_);
        destroy(last_ - n + M, top());
      }
    }
    last_ -= n;
    if (last_ < base()) last_ += M;
    return pos;
  }
}

} // namespace alps

#endif // ALPS_FIXED_CAPACITY_DEQUE_DETAIL_H
