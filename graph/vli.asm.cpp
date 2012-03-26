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

// atm vli only works with GCC
#if defined(__GNUG__) && !defined(__ICC) && !defined(__FCC_VERSION)

#include <boost/cstdint.hpp>

namespace alps {
	namespace graph {
		namespace detail {

			void vli256_add(boost::uint64_t * /*%%rdi*/, boost::uint64_t const * /*%%rsi*/) {
				asm (
					"movq  (%%rdi), %%r8					\n"
					"movq  0x08(%%rdi), %%r9				\n"
					"movq  0x10(%%rdi), %%rax				\n"
					"movq  0x18(%%rdi), %%rcx				\n"
					"addq  (%%rsi), %%r8					\n"
					"adcq  0x08(%%rsi), %%r9				\n"
					"adcq  0x10(%%rsi), %%rax				\n"
					"adcq  0x18(%%rsi), %%rcx				\n"
					"movq  %%r8, (%%rdi)					\n"
					"movq  %%r9, 0x08(%%rdi)				\n"
					"movq  %%rax, 0x10(%%rdi)				\n"
					"movq  %%rcx, 0x18(%%rdi)				\n"
					: : : "rax", "rcx", "rdi", "rsi", "r8", "r9", "memory"
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

			void vli256_madd(boost::uint64_t * /*c:=rdi*/, boost::uint64_t const * /*A:=rsi*/, boost::uint64_t const * /*B:=rdx(->rcx)*/) {
				asm (
					"movq  %%rdx, %%rcx						\n"	// B -> rcx
					// check if pushq/popq is faster
					"movq  %%r10, -0x08(%%rsp)				\n" // r10 -> stack
					"movq  %%r12, -0x10(%%rsp)				\n" // r12 -> stack
					"movq  %%r13, -0x18(%%rsp)				\n" // r13 -> stack
					"movq  %%r14, -0x20(%%rsp)				\n" // r14 -> stack
					"movq  %%r15, -0x28(%%rsp)				\n" // r15 -> stack
					"movq  %%rbx, -0x30(%%rsp)				\n" // rbx -> stack

					"movq  0x00(%%rsi), %%r10				\n" // A[0] -> r10
					"movq  0x10(%%rcx), %%rbx				\n" // B[2] -> rbx
					"movq  0x18(%%rcx), %%r8				\n" // B[3] -> rcx
					"movq  %%rbx, %%rax						\n" // rbx -> rax (B[2])
					"mulq  %%r10							\n" // A[0] * B[2] -> rax, rdx
					"imulq %%r10, %%r8						\n" // A[0] * B[3] -> r8

					"movq  0x00(%%rdi), %%r12				\n" // C[0] -> r12
					"movq  0x08(%%rdi), %%r13				\n" // C[1] -> r13
					"movq  0x10(%%rdi), %%r14				\n" // C[2] -> r14
					"movq  0x18(%%rdi), %%r15				\n" // C[3] -> r15

					"addq  %%rax, %%r14						\n" // C[2] += A[0] * B[2](0-63)
					"adcq  %%rdx, %%r15						\n" // C[3] += A[0] * B[2](64-127) + CB

					"addq  %%r8, %%r15						\n" // C[3] += A[0] * B[3]

					"movq  0x08(%%rsi), %%r8				\n" // A[1] -> r8
					"movq  0x08(%%rcx), %%r9				\n" // B[1] -> r9
					"movq  %%r8, %%rax						\n" // r8 -> rax (A[1])
					"mulq  %%r9								\n" // A[1] * B[1] -> rax, rdx

					"addq  %%rax, %%r14						\n" // C[2] += A[1] * B[1](0-63)
					"adcq  %%rdx, %%r15						\n" // C[3] += A[1] * B[1](64-127) + CB

					"imulq %%r8, %%rbx						\n" // A[1] * B[2] -> rbx
					"addq  %%rbx, %%r15						\n" // C[3] += A[1] * B[2]

					"movq  0x08(%%rcx), %%rax				\n" // B[1] -> rax
					"movq  0x10(%%rsi), %%rbx				\n" // A[2] -> rbx
					"imulq %%rbx, %%r9						\n" // A[2] * B[1] -> r9

					"mulq  %%r10							\n" // A[0] * B[1] -> rax, rdx
					"addq  %%rax, %%r13						\n" // C[1] += A[0] * B[1](0-31)
					"adcq  %%rdx, %%r14						\n" // C[2] += A[0] * B[1](64-127) + CB
					"adcq  %%r9, %%r15						\n" // C[3] += A[2] * B[1] + CB
					
					"movq  0x00(%%rcx), %%r9				\n" // B[0] -> r9
					"movq  %%r9, %%rax						\n" // r9 -> rax (B[0])
					"mulq  %%r10							\n" // A[0] * B[0] -> rax, rdx
					"addq  %%rax, %%r12						\n" // C[0] += A[0] * B[0](0-63)
					"adcq  %%rdx, %%r13						\n" // C[1] += A[0] * B[0](64-127) + CB
					"adcq  $0, %%r14						\n" // C[2] += 0 + CB
					"adcq  $0, %%r15						\n" // C[3] += 0 + CB
					
					"movq  %%r9, %%r10						\n" // r9 -> r10 (B[0])
					"imulq  0x18(%%rsi), %%r10				\n" // A[3] * B[0] -> r10

					"movq  %%r9, %%rax						\n" // r9 -> rax (B[0])
					"mulq  %%r8								\n" // A[1] * B[0] -> rax, rdx
					"addq  %%rax, %%r13						\n" // C[1] += A[1] * B[0](0-63)
					"adcq  %%rdx, %%r14						\n" // C[2] += A[1] * B[0](64-127) + CB
					"adcq  %%r10, %%r15						\n" // C[3] += A[3] * B[0] + CB

					"movq  %%r9, %%rax						\n" // r9 -> rax (B[0])
					"mulq  %%rbx							\n" // A[2] * B[0] -> rax, rdx
					"addq  %%rax, %%r14						\n" // C[2] += A[2] * B[0](0-31)
					"adcq  %%rdx, %%r15						\n" // C[3] += A[2] * B[0](64-127) + CB
					
					"movq  %%r12, (%%rdi)					\n" // r12 -> C[0]
					"movq  %%r13, 0x08(%%rdi)				\n" // r13 -> C[1]
					"movq  %%r14, 0x10(%%rdi)				\n" // r14 -> C[2]
					"movq  %%r15, 0x18(%%rdi)				\n" // r14 -> C[3]

					"movq -0x30(%%rsp), %%rbx				\n" // stack -> rbx
					"movq -0x28(%%rsp), %%r15				\n" // stack -> r15
					"movq -0x20(%%rsp), %%r14				\n" // stack -> r14
					"movq -0x18(%%rsp), %%r13				\n" // stack -> r13
					"movq -0x10(%%rsp), %%r12				\n" // stack -> r12
					"movq -0x08(%%rsp), %%r10				\n" // stack -> r10
					: : :
				);
			}

		}
	}
}

#endif
