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
#include <sstream>
#include <iostream>

#ifndef ALPS_GRAPH_VLI
#define ALPS_GRAPH_VLI

namespace alps {
	namespace graph {
		namespace detail {
// = uint
			template<std::size_t P, std::size_t N> struct vli_set {
				static inline void apply(boost::array<boost::uint64_t, N> & lhs, boost::uint64_t rhs, boost::uint64_t fill) {
					lhs[P] = fill;
					vli_set<P + 1, N>::apply(lhs, rhs, fill);
				}
			};
			template<std::size_t N> struct vli_set<0, N> {
				static inline void apply(boost::array<boost::uint64_t, N> & lhs, boost::uint64_t rhs, boost::uint64_t fill) {
					lhs.front() = rhs;
					vli_set<1, N>::apply(lhs, rhs, fill);
				}
			};
			template<std::size_t N> struct vli_set<N, N> {
				static inline void apply(boost::array<boost::uint64_t, N> & lhs, boost::uint64_t rhs, boost::uint64_t fill) {
					lhs.back() = fill;
				}
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
					return lhs[P] < rhs[P] || lhs[P] == rhs[P] && vli_less<P - 1, N>::apply(lhs, rhs);
				}
			};
			template<std::size_t N> struct vli_less<0, N> {
				static inline bool apply(boost::array<boost::uint64_t, N> const & lhs, boost::array<boost::uint64_t, N> const & rhs) {
					return lhs.front() < rhs.front();
				}
			};
// +=
			template<std::size_t P, std::size_t N> struct vli_add_eq {
				static inline void apply(boost::array<boost::uint64_t, N> & lhs, boost::array<boost::uint64_t, N> const & rhs, boost::uint64_t carry = 0UL) {
					boost::uint64_t lb =  (lhs[P - 1] & 0x00000000FFFFFFFFULL)        +  (rhs[P - 1] & 0x00000000FFFFFFFFULL)        + carry;
					boost::uint64_t hb = ((lhs[P - 1] & 0xFFFFFFFF00000000ULL) >> 32) + ((rhs[P - 1] & 0xFFFFFFFF00000000ULL) >> 32) + ((lb & 0x0000000100000000ULL) >> 32);
					carry = (hb & 0x0000000100000000ULL) >> 32;
					lhs[P - 1] = (lb & 0x00000000FFFFFFFFULL) | (hb << 32);
					vli_add_eq<P + 1, N>::apply(lhs, rhs, carry);
				}
			};
			template<std::size_t N> struct vli_add_eq<N, N> {
				static inline void apply(boost::array<boost::uint64_t, N> & lhs, boost::array<boost::uint64_t, N> const & rhs, boost::uint64_t carry = 0UL) {
					lhs.back() += rhs.back() + carry;
				}
			};
// -=
			template<std::size_t P, std::size_t N> struct vli_sub_eq {
				static inline void apply(boost::array<boost::uint64_t, N> & lhs, boost::array<boost::uint64_t, N> const & rhs, boost::uint64_t borrow = 0UL) {
					boost::uint64_t lb =  (lhs[P - 1] & 0x00000000FFFFFFFFULL)        -  (rhs[P - 1] & 0x00000000FFFFFFFFULL)        - borrow;
					boost::uint64_t hb = ((lhs[P - 1] & 0xFFFFFFFF00000000ULL) >> 32) - ((rhs[P - 1] & 0xFFFFFFFF00000000ULL) >> 32) - ((lb & 0x0000000100000000ULL) >> 32);
					borrow = (hb & 0x0000000100000000ULL) >> 32;
					lhs[P - 1] = (lb & 0x00000000FFFFFFFFULL) | (hb << 32);
					vli_sub_eq<P + 1, N>::apply(lhs, rhs, borrow);
				}
			};
			template<std::size_t N> struct vli_sub_eq<N, N> {
				static inline void apply(boost::array<boost::uint64_t, N> & lhs, boost::array<boost::uint64_t, N> const & rhs, boost::uint64_t borrow = 0UL) {
					lhs.back() -= rhs.back() + borrow;
				}
			};
// *=
			template<std::size_t N, std::size_t L> struct vli_mul_eq_carry {
				static inline void apply(boost::array<boost::uint64_t, N> & lhs, boost::uint64_t carry) {
					lhs[L] += carry;
				}
			};
			template<std::size_t N> struct vli_mul_eq_carry<N, N> {
				static inline void apply(boost::array<boost::uint64_t, N> &, boost::uint64_t) {}
			};
			template<std::size_t P, std::size_t Q, std::size_t N, std::size_t> struct vli_mul_eq_calc {
				static inline void apply(boost::array<boost::uint64_t, N> & lhs, boost::array<boost::uint64_t, N> const & arg1, boost::array<boost::uint64_t, N> const & arg2) {
					boost::uint64_t b00 =  (arg1[P] & 0x00000000FFFFFFFFULL)        *  (arg2[Q] & 0x00000000FFFFFFFFULL);
					boost::uint64_t b01 =  (arg1[P] & 0x00000000FFFFFFFFULL)        * ((arg2[Q] & 0xFFFFFFFF00000000ULL) >> 32);
					boost::uint64_t b10 = ((arg1[P] & 0xFFFFFFFF00000000ULL) >> 32) *  (arg2[Q] & 0x00000000FFFFFFFFULL);
					boost::uint64_t b11 = ((arg1[P] & 0xFFFFFFFF00000000ULL) >> 32) * ((arg2[Q] & 0xFFFFFFFF00000000ULL) >> 32);

					boost::uint64_t lb0 = (lhs[P + Q] & 0x00000000FFFFFFFFULL) + (b00 & 0x00000000FFFFFFFFULL);
					boost::uint64_t hb0 = ((lhs[P + Q] & 0xFFFFFFFF00000000ULL) >> 32) + ((b00 & 0xFFFFFFFF00000000ULL) >> 32) 
									    + (b01 & 0x00000000FFFFFFFFULL) + (b10 & 0x00000000FFFFFFFFULL) + ((lb0 & 0xFFFFFFFF00000000ULL) >> 32);

					boost::uint64_t lb1 = (lhs[P + Q + 1] & 0x00000000FFFFFFFFULL) + ((b01 & 0xFFFFFFFF00000000ULL) >> 32) 
										+ ((b10 & 0xFFFFFFFF00000000ULL) >> 32) + (b11 & 0x00000000FFFFFFFFULL) + ((hb0 & 0xFFFFFFFF00000000ULL) >> 32);
					boost::uint64_t hb1 = ((lhs[P + Q + 1] & 0xFFFFFFFF00000000ULL) >> 32) + ((b11 & 0xFFFFFFFF00000000ULL) >> 32) + ((lb1 & 0xFFFFFFFF00000000ULL) >> 32);

					lhs[P + Q] = (lb0 & 0x00000000FFFFFFFFULL) | (hb0 << 32);
					lhs[P + Q + 1] = (lb1 & 0x00000000FFFFFFFFULL) | (hb1 << 32);

					vli_mul_eq_carry<N, P + Q + 2>::apply(lhs, (hb1 & 0xFFFFFFFF00000000ULL) >> 32);
				}
			};
			template<std::size_t P, std::size_t Q, std::size_t N> struct vli_mul_eq_calc<P, Q, N, N> {
				static inline void apply(boost::array<boost::uint64_t, N> & lhs, boost::array<boost::uint64_t, N> const & arg1, boost::array<boost::uint64_t, N> const & arg2) {
					lhs[P + Q] += arg1[P] * arg2[Q];
				}
			};
			template<std::size_t P, std::size_t Q, std::size_t N, std::size_t L, std::size_t> struct vli_mul_eq_inc_q {
				static inline void apply(boost::array<boost::uint64_t, N> & lhs, boost::array<boost::uint64_t, N> const & arg1, boost::array<boost::uint64_t, N> const & arg2) {
					vli_mul_eq_inc_q<P, Q + 1, N, L, P + Q + 1>::apply(lhs, arg1, arg2);
				}
			};
			template<std::size_t P, std::size_t Q, std::size_t N, std::size_t L> struct vli_mul_eq_inc_q<P, Q, N, L, L> {
				static inline void apply(boost::array<boost::uint64_t, N> & lhs, boost::array<boost::uint64_t, N> const & arg1, boost::array<boost::uint64_t, N> const & arg2) {
					vli_mul_eq_calc<P, Q, N, P + Q + 1>::apply(lhs, arg1, arg2);
				}
			};
			template<std::size_t P, std::size_t Q, std::size_t N, std::size_t L> struct vli_mul_eq_inc_q<P, Q, N, L, N> {
				static inline void apply(boost::array<boost::uint64_t, N> &, boost::array<boost::uint64_t, N> const &, boost::array<boost::uint64_t, N> const &) {}
			};
			template<std::size_t P, std::size_t N, std::size_t L, std::size_t> struct vli_mul_eq_inc_p {
				static inline void apply(boost::array<boost::uint64_t, N> & lhs, boost::array<boost::uint64_t, N> const & arg1, boost::array<boost::uint64_t, N> const & arg2) {
					vli_mul_eq_inc_q<P, 0, N, L - 1, P>::apply(lhs, arg1, arg2);
					vli_mul_eq_inc_p<P + 1, N, L, P>::apply(lhs, arg1, arg2);
				}
			};
			template<std::size_t P, std::size_t N, std::size_t L> struct vli_mul_eq_inc_p<P, N, L, L> {
				static inline void apply(boost::array<boost::uint64_t, N> &, boost::array<boost::uint64_t, N> const &, boost::array<boost::uint64_t, N> const &) {}
			};
			template<std::size_t N, std::size_t L = 1> struct vli_mul_eq {
				static inline void apply(boost::array<boost::uint64_t, N - 1> & lhs, boost::array<boost::uint64_t, N - 1> const & arg1, boost::array<boost::uint64_t, N - 1> const & arg2) {
					vli_mul_eq_inc_p<0, N - 1, L, 0>::apply(lhs, arg1, arg2);
					vli_mul_eq<N, L + 1>::apply(lhs, arg1, arg2);
				}
			};
			template<std::size_t N> struct vli_mul_eq<N, N> {
				static inline void apply(boost::array<boost::uint64_t, N - 1> &, boost::array<boost::uint64_t, N - 1> const &, boost::array<boost::uint64_t, N - 1> const &) {}
			};
		}
// VLI
		template<std::size_t N> class vli {
			public:
// Constructor
				inline vli() {
					detail::vli_set<0, N>::apply(data, 0ULL, 0ULL);
				}
				inline vli(boost::int64_t arg) {
					detail::vli_set<0, N>::apply(data, static_cast<boost::uint64_t>(arg), arg < 0 ? 0xFFFFFFFFFFFFFFFFULL : 0ULL);
				}
				inline vli(vli<N> const & arg) {
					data = arg.data;
				}
// Assign
				inline vli<N> & operator=(vli<N> const & arg) {
					data = arg.data;
				}
// Size
                static const std::size_t static_size = N;
                static inline std::size_t size(){
                    return static_size;
                }
// []
				inline boost::uint64_t & operator [](std::size_t index) {
					return data[index];
				}
				inline boost::uint64_t const & operator [](std::size_t index) const {
					return data[index];
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
					return data.back() & 0x8000000000000000ULL;
				}
// +
				inline vli<N> & operator+=(vli<N> const & arg) {
					detail::vli_add_eq<1, N>::apply(data, arg.data);
					return *this;
				}
// -
				inline vli<N> operator-() const {
					return vli<N>(*this) *= -1;
				}
				inline vli<N> & operator-=(vli<N> const & arg) {
					detail::vli_sub_eq<1, N>::apply(data, arg.data);
					return *this;
				}
// *
				inline vli<N> & operator*=(vli<N> const & arg) {
					boost::array<boost::uint64_t, N> tmp;
					detail::vli_set<0, N>::apply(tmp, 0ULL, 0ULL);
					detail::vli_mul_eq<N + 1>::apply(tmp, data, arg.data);
					std::swap(data, tmp);
					return *this;
				}
// str()
				std::string str() const {
					std::ostringstream buffer;
					vli<N> value(sign() ? -*this : *this);
					if (sign())
						buffer << "-";
					if (value == 0)
						buffer << 0;
					else {
						std::size_t digits = 1;
						for (vli<N> next(1); (next *= 10) <= value; ++digits);
						do {
							vli<N> tmp1(1);
							for (std::size_t i = 1; i < digits; ++i, tmp1 *= 10);
							vli<N> tmp2(tmp1);
							std::size_t d;
							for (d = 0; tmp2 <= value; ++d, tmp2 += tmp1);
							value -= (tmp2 -= tmp1);
							buffer << d;
						} while(--digits);
					}
					return buffer.str();
				}
			private:
// raw data
				boost::array<boost::uint64_t, N> data;
		};
// +
		template<std::size_t N> inline vli<N> operator+(vli<N> arg1, vli<N> const & arg2) {
			return arg1 += arg2;
		}
		template<std::size_t N> inline vli<N> operator+(vli<N> arg1, boost::int64_t arg2) {
			return arg1 += vli<N>(arg2);
		}
		template<std::size_t N> inline vli<N> operator+(boost::int64_t arg1, vli<N> const & arg2) {
			return vli<N>(arg1) += arg2;
		}
// -
		template<std::size_t N> inline vli<N> operator-(vli<N> arg1, vli<N> const & arg2) {
			return arg1 -= arg2;
		}
		template<std::size_t N> inline vli<N> operator-(vli<N> arg1, boost::int64_t arg2) {
			return arg1 -= vli<N>(arg2);
		}
		template<std::size_t N> inline vli<N> operator-(boost::int64_t arg1, vli<N> const & arg2) {
			return vli<N>(arg1) -= arg2;
		}
// *
		template<std::size_t N> inline vli<N> operator*(vli<N> arg1, vli<N> const & arg2) {
			return arg1 *= arg2;
		}
		template<std::size_t N> inline vli<N> operator*(vli<N> arg1, boost::int64_t arg2) {
			return arg1 *= vli<N>(arg2);
		}
		template<std::size_t N> inline vli<N> operator*(boost::int64_t arg1, vli<N> const & arg2) {
			return vli<N>(arg1) *= arg2;
		}
// os <<
		template<std::size_t N> std::ostream & operator<<(std::ostream & os, vli<N> const & arg) {
			return os << arg.str();
		}
	}
}

#endif
