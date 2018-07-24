/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#include <alps/alea/batch.hpp>

#include "gtest/gtest.h"
#include "dataset.hpp"

#include <iterator>
#include <iostream>

using namespace alps::alea;

void test_galois(size_t n, size_t steps)
{
    std::vector< std::pair<size_t, size_t> > intv(n);

    internal::galois_hopper hop(n);
    for (size_t i = 0; i != steps; ++i, ++hop) {
        if (hop.merge_mode()) {
            EXPECT_LT(hop.merge_into(), n);

            std::pair<size_t, size_t> curr = intv[hop.current()];
            std::pair<size_t, size_t> oldnext = intv[hop.merge_into()];

            std::cerr << (i % (n/2) == 0 ? "\n" : "; ")
                      << "[" << curr.first << "," << curr.second << ") -> "
                      << "[" << oldnext.first << "," << oldnext.second << ")";

            EXPECT_EQ(curr.second, oldnext.first);
            EXPECT_LT(curr.first, oldnext.second);

            intv[hop.merge_into()] = std::make_pair(curr.first, oldnext.second);
        }

        EXPECT_LT(hop.current(), n);
        intv[hop.current()] = std::make_pair(i, i + 1);
    }
    std::cerr << std::endl << std::endl;
}

TEST(galois_hop, testwork0) { test_galois(4, 4); }
TEST(galois_hop, testwork1) { test_galois(6, 100); }
TEST(galois_hop, testwork2) { test_galois(8, 100); }
TEST(galois_hop, testwork3) { test_galois(24, 119); }


class galois_case
    : public ::testing::Test
{
public:
    galois_case() : acc_(0) { }

    virtual void SetUp()
    {
        acc_ = batch_acc<double>(2, 8);
        std::vector<double> curr(2);
        for (size_t i = 0; i != twogauss_count; ++i) {
            std::copy(twogauss_data[i], twogauss_data[i+1], curr.begin());
            acc_ << curr;
        }
    }

    batch_acc<double> acc_;
};

TEST_F(galois_case, correct_order)
{
    EXPECT_EQ(acc_.count(), twogauss_count);

    for (size_t i = 0; i != acc_.store().num_batches(); ++i) {
        size_t offset = acc_.offset()[i];
        size_t size = acc_.store().count()[i];
        EXPECT_LE(offset + size, twogauss_count);

        std::vector<double> expect(2, 0.0);
        for (size_t j = offset; j != offset + size; ++j) {
            expect[0] += twogauss_data[j][0];
            expect[1] += twogauss_data[j][1];
        }

        EXPECT_NEAR(acc_.store().batch()(0, i), expect[0], 1e-5);
        EXPECT_NEAR(acc_.store().batch()(1, i), expect[1], 1e-5);
    }
}

// int main(int argc, char **argv)
// {
//     ::testing::InitGoogleTest(&argc, argv);
//     return RUN_ALL_TESTS();
// }
