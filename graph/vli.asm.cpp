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

#include <boost/cstdint.hpp>

namespace alps {
	namespace graph {
		namespace detail {

			void vli256_add(boost::uint64_t * A, boost::uint64_t const * B) {
				asm (
					"movq  %[lhs], %%r8						\n"
					"movq  0x08%[lhs], %%r9					\n"
					"movq  0x10%[lhs], %%rax				\n"
					"movq  0x18%[lhs], %%rcx				\n"
					"addq  %[arg], %%r8						\n"
					"adcq  0x08%[arg], %%r9					\n"
					"adcq  0x10%[arg], %%rax				\n"
					"adcq  0x18%[arg], %%rcx				\n"
					"movq  %%r8, %[lhs]						\n"
					"movq  %%r9, 0x08%[lhs]					\n"
					"movq  %%rax, 0x10%[lhs]				\n"
					"movq  %%rcx, 0x18%[lhs]				\n"

					: [lhs] "+m" (A[0])
					: [arg] "m" (B[0])
					: "rax", "rcx", "r8", "r9", "memory"
				);
			}

			void vli256_mul(boost::uint64_t * A, boost::uint64_t const * B) {
				asm (
						"movq  0x18%[lhs], %%r9				\n"
						"movq  %[lhs2], %%rax				\n"
						"movq  %[lhs2], %%rcx				\n"
						// 3 * 0
						"imulq %[arg0], %%r9				\n"
						// 2 * 0
						"mulq  %[arg0]						\n"
						"movq  %%rax, %[lhs2]				\n"
						"addq  %%rdx, %%r9					\n"
						// 2 * 1
						"imulq %[arg1], %%rcx				\n"
						"addq  %%rcx, %%r9					\n"
						"movq  %[lhs1], %%r8				\n"
						"movq  %%r8, %%rax					\n"
						// 1 * 0
						"mulq  %[arg0]						\n"
						"movq  %%rax, %[lhs1]				\n"
						"addq  %%rdx, %[lhs2]				\n"
						"adcq  $0, %%r9						\n"
						"movq  %%r8, %%rax					\n"
						// 1 * 1
						"mulq  %[arg1]						\n"
						"addq  %%rax, %[lhs2]				\n"
						"adcq  %%rdx, %%r9					\n"
						// 1 * 2
						"imulq %[arg2], %%r8				\n"
						"addq  %%r8, %%r9					\n"
						"movq  %[lhs], %%r8					\n"
						"movq  %%r8, %%rax					\n"
						// 0 * 0
						"mulq  %[arg0]						\n"
						"movq  %%rax, %[lhs]				\n"
						"addq  %%rdx, %[lhs1]				\n"
						"adcq  $0, %[lhs2]					\n"
						"movq  %%r8, %%rax					\n"
						// 0 * 1
						"mulq  %[arg1]						\n"
						"addq  %%rax, %[lhs1]				\n"
						"adcq  %%rdx, %[lhs2]				\n"
						"adcq  $0, %[lhs2]					\n"
						"movq  %%r8, %%rax					\n"
						// 0 * 2
						"mulq  %[arg2]						\n"
						"addq  %%rax, %[lhs2]				\n"
						"adcq  %%rdx, %%r9					\n"
						// 0 * 3
						"imulq %[arg3], %%r8				\n"
						"addq  %%r8, %%r9					\n"
						"movq  %%r9, 0x18%[lhs]				\n"
					: [lhs] "+m" (A[0])
					, [lhs1] "+r" (A[1])
					, [lhs2] "+r" (A[2])
					: [arg0] "r" (B[0])
					, [arg1] "r" (B[1])
					, [arg2] "r" (B[2])
					, [arg3] "r" (B[3])
					: "rax", "rcx", "rdx", "r8", "r9", "memory"
				);
			}

		}
	}
}
