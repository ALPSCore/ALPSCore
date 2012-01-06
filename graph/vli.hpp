/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2012 by Lukas Gamper <gamperl@gmail.com>                   *
 *                                                                                 *
 * This software is part of the ALPS libraries, published under the ALPS           *
 * Library License; you can use, redistribute it and/or modify it under            *
 * the terms of the license, either version 1 or (at your option) any later        *
 * version.                                                                        *
 *                                                                                 *
 * You should have received a copy of the ALPS Library License along with          *
 * the ALPS Libraries; see the file LICENSE.txt. If not, the license is also       *
 * available from http://alps.comp-phys.org/.                                      *
 *                                                                                 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        *
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT       *
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE       *
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,     *
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER     *
 * DEALINGS IN THE SOFTWARE.                                                       *
 *                                                                                 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <boost/array.hpp>
#include <boost/cstdint.hpp>

#include <cstdlib>
#include <cstring>
#include <iostream>

#ifndef ALPS_GRAPH_VLI
#define ALPS_GRAPH_VLI

namespace alps {
	namespace graph {

		namespace detail {
// = uint
			template<std::size_t P, std::size_t N> struct vli_set {
				static inline void apply(boost::array<boost::uint64_t, N> & lhs, boost::uint64_t) {
					lhs[P] = 0UL;
					vli_set<P + 1, N>::apply(lhs, 0UL);
				}
			};
			template<std::size_t N> struct vli_set<1, N> {
				static inline void apply(boost::array<boost::uint64_t, N> & lhs, boost::uint64_t rhs) {
					lhs[1] = (rhs & 0xC000000000000000ULL) >> 62;
					vli_set<2, N>::apply(lhs, 0UL);
				}
			};
			template<std::size_t N> struct vli_set<0, N> {
				static inline void apply(boost::array<boost::uint64_t, N> & lhs, boost::uint64_t rhs) {
					lhs.front() = rhs & 0x3FFFFFFFFFFFFFFFULL;
					vli_set<1, N>::apply(lhs, 0UL);
				}
			};
			template<std::size_t N> struct vli_set<N, N> {
				static inline void apply(boost::array<boost::uint64_t, N> &, boost::uint64_t) {}
			};

// !=
			template<std::size_t P, std::size_t N> struct vli_neq {
				static inline bool apply(boost::array<boost::uint64_t, N> const & lhs, boost::array<boost::uint64_t, N> const & rhs) {
					return lhs[P] != rhs[P] || vli_neq<P - 1, N>::apply(lhs, rhs);
				}
			};
			template<std::size_t N> struct vli_neq<0, N> {
				static inline bool apply(boost::array<boost::uint64_t, N> const & lhs, boost::array<boost::uint64_t, N> const & rhs) {
					return lhs.front() != rhs.front();
				}
			};

// <
			template<std::size_t P, std::size_t N> struct vli_less {
				static inline bool apply(boost::array<boost::uint64_t, N> const & lhs, boost::array<boost::uint64_t, N> const & rhs) {
					return lhs[P] < rhs[P] || vli_less<P - 1, N>::apply(lhs, rhs);
				}
			};
			template<std::size_t N> struct vli_less<0, N> {
				static inline bool apply(boost::array<boost::uint64_t, N> const & lhs, boost::array<boost::uint64_t, N> const & rhs) {
					return lhs.front() < rhs.front();
				}
			};
			
// +=
			template<std::size_t P, std::size_t N> struct vli_add_eq {
				static inline void apply(boost::array<boost::uint64_t, N> & lhs, boost::array<boost::uint64_t, N> const & rhs) {
					lhs[P - 1] += rhs[P - 1];
					lhs[P    ] += (lhs[P - 1] & 0xC000000000000000ULL) >> 62;
					lhs[P - 1] &= 0x3FFFFFFFFFFFFFFFULL;
					vli_add_eq<P + 1, N>::apply(lhs, rhs);
				}
			};
			template<std::size_t N> struct vli_add_eq<N, N> {
				template<typename T> static inline void apply(boost::array<boost::uint64_t, 0> & lhs, boost::array<boost::uint64_t, 0> const & rhs) {
					((lhs.back() &= 0x3FFFFFFFFFFFFFFFULL) += (rhs.back() & 0x3FFFFFFFFFFFFFFFULL)) &= 0x3FFFFFFFFFFFFFFFULL;
				}
			};

// -=
			template<std::size_t P, std::size_t N> struct vli_sub_eq {
				static inline void apply(boost::array<boost::uint64_t, N> & lhs, boost::array<boost::uint64_t, N> const & rhs) {
					lhs[P - 1] -= rhs[P - 1] - ((lhs[P - 2] & 0x4000000000000000ULL) >> 62);
					lhs[P - 1] &= 0x3FFFFFFFFFFFFFFFULL;
					vli_sub_eq<P + 1, N>::apply(lhs, rhs);
				}
			};
			template<std::size_t N> struct vli_sub_eq<N, N> {
				template<typename T> static inline void apply(boost::array<boost::uint64_t, N> & lhs, boost::array<boost::uint64_t, N> const & rhs) {
					lhs.back() -= rhs.back() - ((lhs[N - 2] & 0x4000000000000000ULL) >> 62);
					lhs[N - 2] &= 0x3FFFFFFFFFFFFFFFULL;
					lhs.back() &= 0xBFFFFFFFFFFFFFFFULL;
				}
			};
			template<std::size_t N> struct vli_sub_eq<1, N> {
				template<typename T> static inline void apply(boost::array<boost::uint64_t, N> & lhs, boost::array<boost::uint64_t, N> const & rhs) {
					lhs.front() -= rhs.front();
					vli_sub_eq<2, N>::apply(lhs, rhs);
				}
			};
			template<> struct vli_sub_eq<1, 1> {
				template<typename T> static inline void apply(boost::array<boost::uint64_t, 1> & lhs, boost::array<boost::uint64_t, 1> const & rhs) {
					lhs.front() -= rhs.front();
				}
			};
// *=
			template<std::size_t P, std::size_t Q, std::size_t N, std::size_t Np1, std::size_t> struct vli_mul_eq {
				static inline void apply(boost::array<boost::uint64_t, N> & lhs, boost::array<boost::uint64_t, N> const & arg1, boost::array<boost::uint64_t, N> const & arg2) {
					lhs[P + Q]	   +=    (arg1[P] & 0x000000007FFFFFFFULL)        *   (arg2[Q] & 0x000000007FFFFFFFULL)
								   +  ((((arg1[P] & 0x3FFFFFFF80000000ULL) >> 31) *   (arg2[Q] & 0x000000007FFFFFFFULL)       ) << 31)
								   +  (( (arg1[P] & 0x000000007FFFFFFFULL)        *  ((arg2[Q] & 0x3FFFFFFF80000000ULL) >> 31)) << 31)
								   ;
					lhs[P + Q]     &= 0x3FFFFFFFFFFFFFFFULL;
					lhs[P + Q + 1] += ((((arg1[P] & 0x3FFFFFFF80000000ULL) >> 31) *   (arg2[Q] & 0x000000007FFFFFFFULL)       ) >> 31)
								   +  (( (arg1[P] & 0x000000007FFFFFFFULL)        *  ((arg2[Q] & 0x3FFFFFFF80000000ULL) >> 31)) >> 31)
								   +    ((arg1[P] & 0x3FFFFFFF80000000ULL) >> 31) *  ((arg2[Q] & 0x3FFFFFFF80000000ULL) >> 31)
								   ;
					lhs[P + Q + 1] &= 0x3FFFFFFFFFFFFFFFULL;
					vli_mul_eq<P + 1, Q    , N, Np1, P + Q + 2>::apply(lhs, arg1, arg2);
					vli_mul_eq<P    , Q + 1, N, Np1, P + Q + 2>::apply(lhs, arg1, arg2);
					vli_mul_eq<P + 1, Q + 1, N, Np1, P + Q + 3>::apply(lhs, arg1, arg2);
				}
			};
			template<std::size_t P, std::size_t Q, std::size_t N, std::size_t Np1> struct vli_mul_eq<P, Q, N, Np1, N> {
				static inline void apply(boost::array<boost::uint64_t, N> & lhs, boost::array<boost::uint64_t, N> const & arg1, boost::array<boost::uint64_t, N> const & arg2) {
					lhs[P + Q]	   +=    (arg1[P] & 0x000000007FFFFFFFULL)        *   (arg2[Q] & 0x000000007FFFFFFFULL)
								   +  ((((arg1[P] & 0x3FFFFFFF80000000ULL) >> 31) *   (arg2[Q] & 0x000000007FFFFFFFULL)       ) << 31)
								   +  (( (arg1[P] & 0x000000007FFFFFFFULL)        *  ((arg2[Q] & 0x3FFFFFFF80000000ULL) >> 31)) << 31)
								   ;
					lhs[P + Q]     &= 0x3FFFFFFFFFFFFFFFULL;
				}
			};
			template<std::size_t P, std::size_t Q, std::size_t N, std::size_t Np1> struct vli_mul_eq<P, Q, N, Np1, Np1> {
				static inline void apply(boost::array<boost::uint64_t, N> &, boost::array<boost::uint64_t, N> const &, boost::array<boost::uint64_t, N> const &) {}
			};
		}

		// 62 bits in each segment: vli<N> represents an N * 62 bit integer
		template<std::size_t N> class vli {

			public:

// Constructor
				inline vli() {
					detail::vli_set<0, N>::apply(data, 0UL);
				}

				inline vli(boost::int64_t arg) {
					detail::vli_set<0, N>::apply(data, std::abs(arg));
					data.back() |= (arg < 0 ? 0x8000000000000000ULL : 0x0000000000000000ULL);
				}

				inline vli(vli<N> const & arg) {
					data = arg.data;
				}
// Assign
				inline vli<N> & operator=(vli<N> const & arg) {
					data = arg.data;
				}

// ==
				inline bool operator==(vli<N> const & arg) const {
					return !(*this != arg);
				}
				inline bool operator!=(vli<N> const & arg) const {
					return detail::vli_neq<N - 1, N>::apply(data, arg.data);
				}

// <
				inline bool operator>(vli<N> const & arg) const {
					return arg < *this;
				}
				inline bool operator>=(vli<N> const & arg) const {
					return arg <= *this;
				}
				inline bool operator<(vli<N> const & arg) const {
					return !!((data.back() ^ arg.data.back()) & 0x8000000000000000ULL) != detail::vli_less<N - 1, N>::apply(data, arg.data);
				}
				inline bool operator<=(vli<N> const & arg) const {
					return *this < arg || *this == arg;
				}

// sign
				inline bool sign() const {
					return (data.back() & 0x8000000000000000ULL) >> 62;
				}

// +
				inline vli<N> operator+(vli<N> arg) const {
					arg += *this;
					return arg;
				}
				inline vli<N> & operator+=(vli<N> const & arg) {
					if (data.sign() == arg.sign())
						detail::vli_add_eq<1, N>::apply(data, arg.data);
					else
						detail::vli_sub_eq<1, N>::apply(data, arg.data);
					return *this;
				}

// -
				inline vli<N> operator-(vli<N> const & arg) const {
					vli<N> tmp = *this;
					if (data.sign() == arg.sign())
						detail::vli_sub_eq<1, N>::apply(tmp.data, arg.data);
					else
						detail::vli_add_eq<1, N>::apply(tmp.data, arg.data);
					return *tmp;
				}
				inline vli<N> & operator-=(vli<N> const & arg) {
					if (data.sign() == arg.sign())
						detail::vli_sub_eq<1, N>::apply(data, arg.data);
					else
						detail::vli_add_eq<1, N>::apply(data, arg.data);
					return *this;
				}

// *
				inline vli<N> operator*(vli<N> arg) const {
					arg *= *this;
					return arg;
				}
				inline vli<N> & operator*=(vli<N> const & arg) {
					boost::array<boost::uint64_t, N> tmp;
					detail::vli_mul_eq<0, 0, N, N + 1, 1>::apply(tmp, data, arg.data);
					tmp.back() |= (tmp.back() ^ arg.data.back()) & 0x8000000000000000ULL;
					std::swap(data, tmp);
					return *this;
				}
				
// str()
				std::string str() const {
					std::string res;
					vli<N> value(*this);
					if (sign()) {
						res = "-";
						value *= -1;
					}
					if (value == 0)
						res += "0";
					else {
						vli<N> tmp;
						std::size_t digit = 1;
						for (vli<N> next(1); (next *= 10) < value; ++digit, tmp = next);
						// TODO!
						res += "<not Impl>";
					}
					return res;
				}

			private:

				boost::array<boost::uint64_t, N> data;
		};
		
		template<std::size_t N> std::ostream & operator<<(std::ostream & os, vli<N> const & arg) {
			return os << arg.str();
		}
	}
}

#endif