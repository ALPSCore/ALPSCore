/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2011 - 2013 by Lukas Gamper <gamperl@gmail.com>                   *
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

#include <alps/ngs/config.hpp>

#ifndef ALPS_NGS_USE_NEW_ALEA
#error "This test only works with new alea library"
#endif

#include <alps/ngs/accumulator/accumulator.hpp>

#include <cassert>

#include <iostream>


int main() {

	alps::accumulator::accumulator_set measurements;
	measurements << alps::accumulator::RealObservable("obs1")
				 << alps::accumulator::RealObservable("obs2");

	for (int i = 1; i < 1000; ++i) {
		measurements["obs1"] << 1.;
		assert(measurements["obs1"].mean<double>() == 1.);
		measurements["obs2"] << i;
		assert(measurements["obs2"].mean<double>() == double(i + 1) / 2.);
	}

	alps::accumulator::result_set results(measurements);
	assert(results["obs1"].mean<double>() == 1.);
	assert(results["obs2"].mean<double>() == 500.);

    return EXIT_SUCCESS;
}
