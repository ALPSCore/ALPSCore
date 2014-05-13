/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1999-2003 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Andreas Laeuchli <laeuchli@itp.phys.ethz.ch>,
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

/// \file bitops.h
/// \brief bit manipulation functions
/// 
/// This header contains bit operations modeled after Cray and Fortran
/// intrinsics.  On Cray machines they are replaced by the intrinsic
/// functions with the same name.

#ifndef ALPS_UTILITY_BITOPS_HPP
#define ALPS_UTILITY_BITOPS_HPP

#include <alps/config.h>

#ifdef cray
# include <intrinsics.h>
#endif

namespace alps {

//
// Cray intrinsic bit operations : gbit, gbits, maskr, popcnt
//

#ifdef cray
# define gbit   _gbit
# define gbits  _gbits
# define maskr  _maskr
# define popcnt _popcnt
#else


#if __GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 4)
namespace detail{
    inline long gcc_popcount(unsigned int x) { return __builtin_popcount(x); }
    inline long gcc_popcount(unsigned long x) { return __builtin_popcountl(x); }
    inline long gcc_popcount(unsigned long long x) { return __builtin_popcountll(x); }
}
#endif


/** \brief extract a bit from a word.
    @param x the word
    @param n position of the bit to be extracted
    @return the n-th bit of the word x */
template <class T, class N>
inline T gbit(T x, N n) { return (x>>n)&1; }

/** \brief extract bits from a word.
    @param x the word
    @param m the number of bits to be extracted
    @param n position of the first bit to be extracted
    @return the m bits starting at bit n  */
template <class T, class N>
inline T gbits(T x, N m, long n) { return (x>>n)&((1<<m)-1); }

/** \brief create a right-justified N
    @param i the number of bits to be set to 1
    @return a word with the rightmost bits set to 1 */
inline uint32_t maskr(uint16_t i) {return (1u<<i)-1;}
inline static int BX_(long x) { return ((x) - (((x)>>1)&0x77777777)
                             - (((x)>>2)&0x33333333)
                             - (((x)>>3)&0x11111111)); }

// TODO:
// use http://graphics.stanford.edu/~seander/bithacks.html
// for windwos use http://msdn.microsoft.com/en-us/library/bb385231.aspx

inline bool poppar(uint32_t x) {
    x ^= x >> 1;
    x ^= x >> 2;
    x = (x & 0x11111111U) * 0x11111111U;
    return (x >> 28) & 1;
}

inline bool poppar(uint64_t x) {
    x ^= x >> 1;
    x ^= x >> 2;
    x = (x & 0x1111111111111111UL) * 0x1111111111111111UL;
    return (x >> 60) & 1;
}

/// \brief count the 1-bits in a word
/// @param x the 32-bit word of which 1-bits should be counted
/// @return the number of 1-bits in the word
#if __GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 4)
inline long popcnt(uint32_t x)
{ return detail::gcc_popcount(x); }
#else
inline long popcnt(uint32_t x)
{ return (((BX_(x)+(BX_(x)>>4)) & 0x0F0F0F0F) % 255); }
#endif

/// \brief count the 1-bits in a word
/// @param x the 64-bit word of which 1-bits should be counted
/// @return the number of 1-bits in the word
#if __GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 4)
inline long popcnt(uint64_t x)
{ return detail::gcc_popcount(x); }
#else
inline long popcnt(uint64_t x)
{ return popcnt(static_cast<uint32_t>(x)) + popcnt(static_cast<uint32_t>(x>>32)); }
#endif

#endif

//
// Fortran-style bit operations : btest, ibits, ibclr, ibset
//

// btest : check p-th bit of i (return value : true for 1 and false for 0)

/// \brief test if a bit is set
/// \param i the integer to be tested
/// \param p the position of the bit to be tested
/// \return true if the p-th bit of i is set, false otherwise
template <class T, class U>
inline bool btest(T i, U p) { return i & (1<<p); }

/// brief extract a bit from an integer
/// \param i the integer from which a bit will be extracted
/// \param p the position of the bit to be extracted
/// \return 1 if the p-th bit of i is set, 0 otherwise
template <class T, class U>
inline T ibits(T i, U p) { return gbit(i, p); }

/// \brief extract several bits from an integer
/// \param i the integer from which bits will be extracted
/// \param p the position of the first bit to be extracted
/// \param n the number of bits to be extracted
/// \return the bits at positions [p,p+n-1]
template <class T, class U, class V>
inline T ibits(T i, U p, V n) { return gbits(i, p, n); }

/// brief brief clear a bit in an integer
/// \param i the integer of which a bit will be cleared
/// \param p the position of the bit to be cleared
/// \return the integer i with the bit at position p set to zero
template <class T, class U>
inline T ibclr(T i, U p) { return i & (~(1 << p)); }

/// brief set a bit in an integer
/// \param i the integer of which a bit will be set
/// \param p the position of the bit to be set
/// \return the integer i with the bit at position p set to one
template <class T, class U>
inline T ibset(T i, U p) { return i | (1 << p); }

/// brief set a bit in an integer to a specified valus
/// \param i the integer of which a bit will be set
/// \param p the position of the bit to be set
/// \param b the value of the bit to be set
/// \return the integer i with the bit at position p set to to the specified value
template <class T, class U, class V>
inline T ibset(T i, U p, V b) { return (i & (~(1 << p))) | ((b & 1) << p); }

} // end namespace alps
#endif // ALPS_UTILITY_BITOPS_HPP
