/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2011 by Lukas Gamper <gamperl@gmail.com>                   *
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

#include <alps/ngs/hash.hpp>
#include <alps/ngs/stacktrace.hpp>

#include <cmath>
#include <stdio.h>
#include <iostream>
#include <stdexcept>

// TODO: check if matrix is linear independant (mathematica)
// tODO: improve hash
// after 2^32 change matrix
// if many independant generators are needed, take different matrecs. Differen meens diferent eigenvalues
class rng {
    public:
        rng(boost::uint64_t seed = 42)
            : state(seed)
        {
            if (seed == 0)
                throw std::runtime_error("Seed 0 is not valid" + ALPS_STACKTRACE);
        }
        boost::uint64_t operator()() {
            using alps::hash_value;
            return state = hash_value(state);
        }
    private:
        boost::uint64_t state;
};

int main() {
    rng gen(42);
/*    boost::uint64_t i = 0, last = 0, next = gen();
    for (; last != next && i < boost::uint64_t(-1); last = next, next = gen(), ++i)
        if ((i & 0xFFFFFFULL) == 0ULL && i > 0) {
            using std::log;
            std::cout << log(i) / log(2) << std::endl;
        }
    if (last != next)
        std::cout << "pass!" << std::endl;
    else
        std::cout << "fail: " << i << " " << last << std::endl;
/*/
//    FILE * pFile = fopen("rng.bin", "wb");
    for (std::size_t i = 0; i < 100000; ++i) {
        boost::uint64_t value = gen();
        std::cout << value << std::endl;
//        fwrite(&value, sizeof(boost::uint64_t), 1, pFile);
    }
//    fclose(pFile);
/*
    hash<double> hasher;
    std::size_t h = hasher(1.);
    std::cout << h << " ";
    hash_combine(h, 4);
    std::cout << h << std::endl;
*/
    return 0;
}
