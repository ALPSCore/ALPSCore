// Copyright 2006 Matthias Troyer

// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <alps/random/parallel/detail/get_prime.hpp>
#include <alps/random/parallel/detail/primelist_64.hpp>
#include <boost/throw_exception.hpp>
#include <boost/assert.hpp>
#include <stdexcept>
#include <string>

// taken from SPRNG implementation

#define MAXPRIME 3037000501U  /* largest odd # < sqrt(2)*2^31+2 */
#define MINPRIME 55108   /* sqrt(MAXPRIME) */
#define MAXPRIMEOFFSET 146138719U /* Total number of available primes */
#define NPRIMES 10000
#define PRIMELISTSIZE1 1000
#define STEP 10000

namespace alps { namespace random { namespace detail {

struct primes_storage
{
  primes_storage()
   : obtained(0)
  {
    int i, j;
    bool isprime;
  
    for(i=3; i < MINPRIME; i += 2) {
      isprime = true;
    
      for(j=0; j < obtained; j++)
        if(i%primes_[j] == 0) {
          isprime = false;
          break;
        }
        else if(primes_[j]*primes_[j] > i)
          break;

      if(isprime) {
        primes_[obtained] = i;
        obtained++;
      }
    }
  }
  
  int operator[](int i) const
  {
    return primes_[i];
  }
  
  int size() const
  {
    return obtained;
  }
  
private:
  int primes_[NPRIMES];
  int obtained;
};

static primes_storage primes;


boost::uint64_t get_prime_64(unsigned int offset)
{
  BOOST_ASSERT(offset <= MAXPRIMEOFFSET);

  if(offset<PRIMELISTSIZE1) 
    return primelist_64[offset];

  unsigned int largest = MAXPRIME;
  
  int index = (unsigned int) ((offset-PRIMELISTSIZE1+1)/STEP) + PRIMELISTSIZE1 -  1;
  largest = primelist_64[index] + 2;
  offset -= (index-PRIMELISTSIZE1+1)*STEP + PRIMELISTSIZE1 - 1;
  
  while(largest > MINPRIME)
  {
    bool isprime = true;
    largest -= 2;
    for(int i=0; i<primes.size(); i++)
      if(largest%primes[i] == 0) {
        isprime = false;
        break;
      }
    
    if(isprime && offset > 0)
      offset--;
    else if(isprime)
      return largest;
  }
  
  // Casting to std::string is a workaround for Fujitsu FCC Compiler
  boost::throw_exception(std::runtime_error(std::string("Insufficient number of primes")));
  return 0; // dummy return

}


} } } // namespace alps::random::detail
