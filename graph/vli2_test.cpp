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

#include <alps/graph/vli.hpp>

#include <iostream>

int main() {
/*
	{
		alps::graph::vli<384> a(10), b(4);
		std::cout << (a < b ? "true" : "false") << " " << (a == b ? "true" : "false") << " " << (a <= b ? "true" : "false") << " " << std::endl;
	}
*/
	{
		alps::graph::vli<128> f(72677163250000LL);
		std::cout << f << std::endl;
//		std::cout << (f += 18947077082LL) << std::endl;
	}
/*
	{
		alps::graph::vli<128> a(114130849121512190392410), b(7501641761016051614121);
		alps::graph::vli<128> ba = b - a;
		a -= b;
		a = a * (-1);
		std::cout << a << " " << ba << std::endl;
	}
	{
		alps::graph::vli<384> a(1), b(8264), c(-1), d(-862), e(100005), f(72677163250000LL);
		std::cout << a << " " << b << " " << c << " " << d << " " << e << " " << (f += 18947077082LL) << std::endl;
	}
	{
		alps::graph::vli<256> a(0);
		a -= 854116085LL;
		std::cout << a << std::endl;
	}
	*/
};
