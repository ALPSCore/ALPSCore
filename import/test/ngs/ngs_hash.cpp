/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

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
