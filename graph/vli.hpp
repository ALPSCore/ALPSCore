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
#include <cassert>

#ifndef ALPS_GRAPH_VLI
#define ALPS_GRAPH_VLI

namespace alps {
	namespace graph {
		namespace detail {
// raw type
#if defined(USE_VLI_32)
			template<std::size_t N> struct vli_raw : public boost::array<boost::uint32_t, N> {};
			inline boost::uint64_t to64(boost::uint32_t data) {
				return static_cast<boost::uint64_t>(data);
			}
#else
			template<std::size_t N> struct vli_raw : public boost::array<boost::uint64_t, N> {};
#endif
// = uint
			template<std::size_t P, std::size_t N> struct vli_set {
				static inline void apply(vli_raw<N> & lhs, boost::uint64_t rhs) {
					lhs[P] = rhs;
					vli_set<P + 1, N>::apply(lhs, rhs);
				}
			};
			template<std::size_t N> struct vli_set<0, N> {
				static inline void apply(vli_raw<N> & lhs, boost::uint64_t rhs) {
#if defined(USE_VLI_32)
					*reinterpret_cast<boost::uint64_t *>(lhs.c_array()) = rhs;
					vli_set<2, N>::apply(lhs, ((rhs & 0x8000000000000000ULL) >> 63) * 0xFFFFFFFFFFFFFFFFULL);
#else
					lhs.front() = rhs;
					vli_set<1, N>::apply(lhs, ((rhs & 0x8000000000000000ULL) >> 63) * 0xFFFFFFFFFFFFFFFFULL);
#endif
				}
			};
			template<std::size_t N> struct vli_set<N, N> {
				static inline void apply(vli_raw<N> & lhs, boost::uint64_t rhs) {}
			};
#if defined(USE_VLI_32)
			template<> struct vli_set<0, 1> {
				static inline void apply(vli_raw<1> & lhs, boost::uint64_t rhs) {
					lhs.front() = rhs;
				}
			};
#endif
			
// !=
			template<std::size_t P, std::size_t N> struct vli_neq {
				static inline bool apply(vli_raw<N> const & lhs, vli_raw<N> const & rhs) {
					return lhs[P] != rhs[P] || vli_neq<P - 1, N>::apply(lhs, rhs);
				}
			};
			template<std::size_t N> struct vli_neq<0, N> {
				static inline bool apply(vli_raw<N> const & lhs, vli_raw<N> const & rhs) {
					return lhs.front() != rhs.front();
				}
			};
// <
			template<std::size_t P, std::size_t N> struct vli_less {
				static inline bool apply(vli_raw<N> const & lhs, vli_raw<N> const & rhs) {
					return lhs[P] < rhs[P] || (lhs[P] == rhs[P] && vli_less<P - 1, N>::apply(lhs, rhs));
				}
			};
			template<std::size_t N> struct vli_less<0, N> {
				static inline bool apply(vli_raw<N> const & lhs, vli_raw<N> const & rhs) {
					return lhs.front() < rhs.front();
				}
			};
// +=
			template<std::size_t P, std::size_t N> struct vli_add_eq {
				static inline void apply(vli_raw<N> & lhs, vli_raw<N> const & rhs, typename vli_raw<N>::value_type carry = 0ULL) {
#if defined(USE_VLI_32)
					boost::uint64_t tmp = to64(lhs[P - 1]) + to64(rhs[P - 1]) + carry;
					carry = (tmp >> 32) & 0x00000001ULL;
					lhs[P - 1] = tmp;
#else
/*
					boost::uint64_t lb = ((lhs[P - 1] + rhs[P - 1]) & 0x00000000FFFFFFFFULL) + carry;
					boost::uint64_t hb = ((lhs[P - 1] + rhs[P - 1]) >> 32) + ((lb >> 32) & 0x0000000000000001ULL);
					carry = (hb >> 32) & 0x0000000000000001ULL;
					lhs[P - 1] = (lb & 0x00000000FFFFFFFFULL) | (hb << 32);
*/
					boost::uint64_t lb =  (lhs[P - 1] & 0x00000000FFFFFFFFULL)        +  (rhs[P - 1] & 0x00000000FFFFFFFFULL)        + carry;
					boost::uint64_t hb = ((lhs[P - 1] & 0xFFFFFFFF00000000ULL) >> 32) + ((rhs[P - 1] & 0xFFFFFFFF00000000ULL) >> 32) + ((lb >> 32) & 0x0000000000000001ULL);
					carry = (hb >> 32) & 0x0000000000000001ULL;
					lhs[P - 1] = (lb & 0x00000000FFFFFFFFULL) | (hb << 32);
#endif
					vli_add_eq<P + 1, N>::apply(lhs, rhs, carry);
				}
			};
			template<std::size_t N> struct vli_add_eq<N, N> {
				static inline void apply(vli_raw<N> & lhs, vli_raw<N> const & rhs, typename vli_raw<N>::value_type carry = 0ULL) {
					lhs.back() += rhs.back() + carry;
				}
			};
// invert
			template<std::size_t P, std::size_t N> struct vli_invert {
				static inline void apply(vli_raw<N> & lhs, vli_raw<N> const & rhs) {
					lhs[P - 1] = ~rhs[P - 1];
					vli_invert<P + 1, N>::apply(lhs, rhs);
				}
			};
			template<std::size_t N> struct vli_invert<N, N> {
				static inline void apply(vli_raw<N> & lhs, vli_raw<N> const & rhs) {
					lhs.back() = ~rhs.back();
				}
			};
// -=
			template<std::size_t P, std::size_t N> struct vli_sub_eq {
				static inline void apply(vli_raw<N> & lhs, vli_raw<N> const & rhs, typename vli_raw<N>::value_type borrow = 0ULL) {
#if defined(USE_VLI_32)
					boost::uint64_t tmp = to64(lhs[P - 1]) - to64(rhs[P - 1]) - borrow;
					borrow = (tmp >> 32) & 0x00000001ULL;
					lhs[P - 1] = tmp;
#else
/*
					boost::uint64_t lb = (lhs[P - 1] & 0x00000000FFFFFFFFULL) - (rhs[P - 1] & 0x00000000FFFFFFFFULL) - borrow;
					boost::uint64_t hb = (lhs[P - 1] >> 32                  ) - (rhs[P - 1] >> 32                  ) - ((lb >> 32) & 0x0000000000000001ULL);
					borrow = (hb >> 32) & 0x0000000000000001ULL;
					lhs[P - 1] = (lb & 0x00000000FFFFFFFFULL) | (hb << 32);
*/
					boost::uint64_t lb =  (lhs[P - 1] & 0x00000000FFFFFFFFULL)        -  (rhs[P - 1] & 0x00000000FFFFFFFFULL)        - borrow;
					boost::uint64_t hb = ((lhs[P - 1] & 0xFFFFFFFF00000000ULL) >> 32) - ((rhs[P - 1] & 0xFFFFFFFF00000000ULL) >> 32) - ((lb >> 32) & 0x0000000000000001ULL);
					borrow = (hb >> 32) & 0x0000000000000001ULL;
					lhs[P - 1] = (lb & 0x00000000FFFFFFFFULL) | (hb << 32);
#endif
					vli_sub_eq<P + 1, N>::apply(lhs, rhs, borrow);
				}
			};
			template<std::size_t N> struct vli_sub_eq<N, N> {
				static inline void apply(vli_raw<N> & lhs, vli_raw<N> const & rhs, typename vli_raw<N>::value_type borrow = 0ULL) {
					lhs.back() -= rhs.back() + borrow;
				}
			};
// *=
			template<std::size_t N, std::size_t L> struct vli_mul_carry {
				static inline void apply(vli_raw<N> & lhs, typename vli_raw<N>::value_type carry) {
					lhs[L] += carry;
				}
			};
			template<std::size_t N> struct vli_mul_carry<N, N> {
				static inline void apply(vli_raw<N> &, typename vli_raw<N>::value_type) {}
			};
#if defined(USE_VLI_32)
			template<std::size_t N, std::size_t L> struct vli_mul_hb1 {
				static inline void apply(vli_raw<N> & lhs, boost::uint64_t value) {
					lhs[L] = (value += to64(lhs[L]));
					vli_mul_carry<N, L + 1>::apply(lhs, value >> 32);
				}
			};
			template<std::size_t N> struct vli_mul_hb1<N, N> {
				static inline void apply(vli_raw<N> &, boost::uint64_t) {}
			};
#endif
			template<std::size_t P, std::size_t Q, std::size_t N, std::size_t> struct vli_mul_calc {
				static inline void apply(vli_raw<N> & lhs, vli_raw<N> const & arg1, vli_raw<N> const & arg2) {
#if defined(USE_VLI_32)
					boost::uint64_t b00 = to64(arg1[P    ]) * to64(arg2[Q    ]);
					boost::uint64_t b01 = to64(arg1[P    ]) * to64(arg2[Q + 1]);
					boost::uint64_t b10 = to64(arg1[P + 1]) * to64(arg2[Q    ]);

					boost::uint64_t lb0 = to64(lhs[P + Q    ]) +  (b00 & 0x00000000FFFFFFFFULL);
					boost::uint64_t hb0 = to64(lhs[P + Q + 1]) + (b00 >> 32) + (b01 & 0x00000000FFFFFFFFULL) + (b10 & 0x00000000FFFFFFFFULL) + (lb0 >> 32);

					lhs[P + Q    ] = lb0;
					lhs[P + Q + 1] = hb0;
					
					vli_mul_hb1<N, P + Q + 2>::apply(lhs, (b01 >> 32) + (b10 >> 32) + (hb0 >> 32));
#else
/*
					boost::uint64_t b00 = (arg1[P] & 0x00000000FFFFFFFFULL) * (arg2[Q] & 0x00000000FFFFFFFFULL);
					boost::uint64_t b01 = (arg1[P] & 0x00000000FFFFFFFFULL) * (arg2[Q] >> 32                  );
					boost::uint64_t b10 = (arg1[P] >> 32                  ) * (arg2[Q] & 0x00000000FFFFFFFFULL);
					boost::uint64_t b11 = (arg1[P] >> 32                  ) * (arg2[Q] >> 32                  );

					boost::uint64_t lb0 = (lhs[P + Q] & 0x00000000FFFFFFFFULL) + (b00 & 0x00000000FFFFFFFFULL);
					boost::uint64_t hb0 = (lhs[P + Q] >> 32) + (b00 >> 32) + (b01 & 0x00000000FFFFFFFFULL) + (b10 & 0x00000000FFFFFFFFULL) + (lb0 >> 32);

					boost::uint64_t lb1 = (lhs[P + Q + 1] & 0x00000000FFFFFFFFULL) + (b01 >> 32) + (b10 >> 32) + (b11 & 0x00000000FFFFFFFFULL) + (hb0 >> 32);
					boost::uint64_t hb1 = (lhs[P + Q + 1] >> 32) + (b11 >> 32) + (lb1 >> 32);

					lhs[P + Q    ] = (lb0 & 0x00000000FFFFFFFFULL) | (hb0 << 32);
					lhs[P + Q + 1] = (lb1 & 0x00000000FFFFFFFFULL) | (hb1 << 32);

					vli_mul_carry<N, P + Q + 2>::apply(lhs, hb1 >> 32);
*/
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

					lhs[P + Q    ] = (lb0 & 0x00000000FFFFFFFFULL) | (hb0 << 32);
					lhs[P + Q + 1] = (lb1 & 0x00000000FFFFFFFFULL) | (hb1 << 32);

					vli_mul_carry<N, P + Q + 2>::apply(lhs, (hb1 & 0xFFFFFFFF00000000ULL) >> 32);
#endif
				}
			};
			template<std::size_t P, std::size_t Q, std::size_t N> struct vli_mul_calc<P, Q, N, N> {
				static inline void apply(vli_raw<N> & lhs, vli_raw<N> const & arg1, vli_raw<N> const & arg2) {
					lhs[P + Q] += arg1[P] * arg2[Q];
				}
			};
			template<std::size_t P, std::size_t Q, std::size_t N, std::size_t L, std::size_t> struct vli_mul_inc_q {
				static inline void apply(vli_raw<N> & lhs, vli_raw<N> const & arg1, vli_raw<N> const & arg2) {
					vli_mul_inc_q<P, Q + 1, N, L, P + Q + 1>::apply(lhs, arg1, arg2);
				}
			};
			template<std::size_t P, std::size_t Q, std::size_t N, std::size_t L> struct vli_mul_inc_q<P, Q, N, L, L> {
				static inline void apply(vli_raw<N> & lhs, vli_raw<N> const & arg1, vli_raw<N> const & arg2) {
					vli_mul_calc<P, Q, N, P + Q + 1>::apply(lhs, arg1, arg2);
				}
			};
			template<std::size_t P, std::size_t Q, std::size_t N, std::size_t L> struct vli_mul_inc_q<P, Q, N, L, N> {
				static inline void apply(vli_raw<N> &, vli_raw<N> const &, vli_raw<N> const &) {}
			};
			template<std::size_t P, std::size_t N, std::size_t L, std::size_t> struct vli_mul_inc_p {
				static inline void apply(vli_raw<N> & lhs, vli_raw<N> const & arg1, vli_raw<N> const & arg2) {
					vli_mul_inc_q<P, 0, N, L - 1, P>::apply(lhs, arg1, arg2);
					vli_mul_inc_p<P + 1, N, L, P>::apply(lhs, arg1, arg2);
				}
			};
			template<std::size_t P, std::size_t N, std::size_t L> struct vli_mul_inc_p<P, N, L, L> {
				static inline void apply(vli_raw<N> &, vli_raw<N> const &, vli_raw<N> const &) {}
			};
			template<std::size_t N, std::size_t L = 1> struct vli_mul {
				static inline void apply(vli_raw<N - 1> & lhs, vli_raw<N - 1> const & arg1, vli_raw<N - 1> const & arg2) {
					vli_mul_inc_p<0, N - 1, L, 0>::apply(lhs, arg1, arg2);
					vli_mul<N, L + 1>::apply(lhs, arg1, arg2);
				}
			};
			template<std::size_t N> struct vli_mul<N, N> {
				static inline void apply(vli_raw<N - 1> &, vli_raw<N - 1> const &, vli_raw<N - 1> const &) {}
			};
		}
// VLI
		template<std::size_t B> class vli {
			public:
// Size
#if defined(USE_VLI_32)
                static const std::size_t static_size = B >> 5;
#else
                static const std::size_t static_size = B >> 6;
#endif
                static inline std::size_t size(){
                    return static_size;
                }
// Constructor
				inline vli() {
					detail::vli_set<0, static_size>::apply(data, 0ULL);
				}
				inline vli(boost::int64_t arg) {
					detail::vli_set<0, static_size>::apply(data, static_cast<boost::uint64_t>(arg));
				}
				inline vli(vli<B> const & arg) {
					data = arg.data;
				}
// Assign
				inline vli<B> & operator=(vli<B> const & arg) {
					data = arg.data;
                    return *this;
				}
// []
				inline typename detail::vli_raw<static_size>::value_type & operator [](std::size_t index) {
					return data[index];
				}
				inline typename detail::vli_raw<static_size>::value_type const & operator [](std::size_t index) const {
					return data[index];
				}
// ==
				inline bool operator==(vli<B> const & arg) const {
					return !(*this != arg);
				}
				inline bool operator!=(vli<B> const & arg) const {
					return detail::vli_neq<static_size - 1, static_size>::apply(data, arg.data);
				}
// <
				inline bool operator>(vli<B> const & arg) const {
					return arg < *this;
				}
				inline bool operator>=(vli<B> const & arg) const {
					return arg <= *this;
				}
				inline bool operator<(vli<B> const & arg) const {
#if defined(USE_VLI_32)
					return !!((data.back() ^ arg.data.back()) & 0x80000000) != detail::vli_less<static_size - 1, static_size>::apply(data, arg.data);
#else
					return !!((data.back() ^ arg.data.back()) & 0x8000000000000000ULL) != detail::vli_less<static_size - 1, static_size>::apply(data, arg.data);
#endif
				}
				inline bool operator<=(vli<B> const & arg) const {
					return *this < arg || *this == arg;
				}
// sign
				inline bool sign() const {
#if defined(USE_VLI_32)
					return data.back() & 0x80000000;
#else
					return data.back() & 0x8000000000000000ULL;
#endif
				}
// +
				inline vli<B> & operator+=(vli<B> const & arg) {
					detail::vli_add_eq<1, static_size>::apply(data, arg.data);
					return *this;
				}
// -
				inline vli<B> operator-() const {
					vli<B> tmp;
					detail::vli_invert<1, static_size>::apply(tmp.data, data);
					return tmp += 1;
				}
				inline vli<B> & operator-=(vli<B> const & arg) {
					detail::vli_sub_eq<1, static_size>::apply(data, arg.data);
					return *this;
				}
// *
				inline vli<B> operator*(vli<B> const & arg) const {
					vli<B> tmp;
					detail::vli_mul<static_size + 1>::apply(tmp.data, data, arg.data);
					return tmp;
				}
				inline vli<B> & operator*=(vli<B> const & arg) {
					detail::vli_raw<static_size> tmp;
					detail::vli_set<0, static_size>::apply(tmp, 0ULL);
					detail::vli_mul<static_size + 1>::apply(tmp, data, arg.data);
					std::swap(data, tmp);
					return *this;
				}
// str()
				std::string str() const {
					std::ostringstream buffer;
					vli<B> value(sign() ? -*this : *this);
					if (sign())
						buffer << "-";
					if (value == 0)
						buffer << 0;
					else {
						std::size_t digits = 1;
						for (vli<B> next(1); (next *= 10) <= value; ++digits);
						do {
							vli<B> tmp1(1);
							for (std::size_t i = 1; i < digits; ++i, tmp1 *= 10);
							vli<B> tmp2(tmp1);
							std::size_t d;
							for (d = 0; tmp2 <= value; ++d, tmp2 += tmp1)
								assert(d < 10);
							value -= (tmp2 -= tmp1);
							buffer << d;
						} while(--digits);
					}
					return buffer.str();
				}
			private:
// raw data
				detail::vli_raw<static_size> data;
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
		template<std::size_t N> inline vli<N> operator*(vli<N> arg1, boost::int64_t arg2) {
			return arg1 * vli<N>(arg2);
		}
		template<std::size_t N> inline vli<N> operator*(boost::int64_t arg1, vli<N> const & arg2) {
			return vli<N>(arg1) * arg2;
		}
// os <<
		template<std::size_t N> std::ostream & operator<<(std::ostream & os, vli<N> const & arg) {
			return os << arg.str();
		}
	}
}

#endif
