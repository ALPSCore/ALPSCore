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


#include <iostream>
#include <iomanip>


#include <alps/graph/vli.hpp>


int main() {
	{
		alps::graph::vli<256> b, a, c, d;
		for (std::size_t i = 0; i < 4; ++i)
			for (std::size_t j = 0; j < 4; ++j) {
				a[0] = 0; a[1] = 0; a[2] = 0; a[3] = 0;
				b[0] = 0; b[1] = 0; b[2] = 0; b[3] = 0;
				d[0] = 0; d[1] = 0; d[2] = 0; d[3] = 0;
				a[i] = 16;
				b[j] = 16;
				std::cout << "a: " << std::setbase(16) << a[0] << " " << a[1] << " " << a[2] << " " << a[3];
				std::cout << ", b: " << std::setbase(16) << b[0] << " " << b[1] << " " << b[2] << " " << b[3];
				c = a * b;
				d.madd(a, b);
				std::cout << ", a*b: " << std::setbase(16) << c[0] << " " << c[1] << " " << c[2] << " " << c[3];
				std::cout << ", a+=a*b: " << std::setbase(16) << d[0] << " " << d[1] << " " << d[2] << " " << d[3] << std::endl;
			}
	}
	{
		alps::graph::vli<256> a(10), b(4);
		std::cout << (a < b ? "true" : "false") << " " << (a == b ? "true" : "false") << " " << (a <= b ? "true" : "false") << " " << std::endl;
	}
	{
		alps::graph::vli<256> a(1LL), b(-1LL);
		for (std::size_t i = 0; i < 75; ++i)
			std::cout << i << ": " << (a *= 10LL) << " " << (b *= 10LL) << std::endl;
		
		alps::graph::vli<256> f(1000000000000000000);
		std::cout << f << std::endl;
		std::cout << (f += 18947077082LL) << std::endl;
		std::cout << (f *= 2LL) << std::endl;
	}
	{
		alps::graph::vli<256> a(1141308421510392410LL), b(750164176105114121LL);
		alps::graph::vli<256> ba = b - a;
		a -= b;
		a = a * (-1);
		std::cout << a << " " << ba << std::endl;
	}
	{
		alps::graph::vli<256> a(1), b(8264), c(-1), d(-862), e(100005), f(72677163250000LL);
		std::cout << a << " " << b << " " << c << " " << d << " " << e << " " << (f += 18947077082LL) << std::endl;
	}
	{
		alps::graph::vli<256> a(0);
		a -= 854116085LL;
		std::cout << a << std::endl;
	}
};
