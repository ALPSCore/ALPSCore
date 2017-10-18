#include <alps/alea/mean.hpp>
#include <alps/alea/variance.hpp>
#include <alps/alea/covariance.hpp>
#include <alps/alea/batch.hpp>

#include "gtest/gtest.h"
#include "dataset.hpp"

#include <iostream>

template <typename Acc>
class twogauss_case
    : public ::testing::Test
{
public:
    typedef typename alps::alea::traits<Acc>::value_type value_type;

    twogauss_case()
        : acc_(2)
    {
        std::vector<value_type> curr(2);
        for (size_t i = 0; i != twogauss_count; ++i) {
            std::copy(twogauss_data[i], twogauss_data[i+1], curr.begin());
            acc_ << curr;
        }
    }

    void test_mean()
    {
        std::vector<value_type> obs_mean = acc_.mean();
        EXPECT_NEAR(obs_mean[0], twogauss_mean[0], 1e-6);
        EXPECT_NEAR(obs_mean[1], twogauss_mean[1], 1e-6);
    }

private:
    Acc acc_;
};

typedef ::testing::Types<
    alps::alea::mean_acc<double>,
    alps::alea::var_acc<double>,
    alps::alea::cov_acc<double>,
    alps::alea::batch_acc<double>
    > test_types;

TYPED_TEST_CASE(twogauss_case, test_types);

TYPED_TEST(twogauss_case, test_mean) { this->test_mean(); }

// int main(int argc, char **argv)
// {
//     ::testing::InitGoogleTest(&argc, argv);
//     return RUN_ALL_TESTS();
// }
