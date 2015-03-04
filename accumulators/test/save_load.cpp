/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <stdexcept>

#include "alps/config.hpp"
#include "alps/accumulators.hpp"
#include "alps/hdf5.hpp"
#include "gtest/gtest.h"
// Test for saving/restoring accumulators and results to/from archives.

// using Google Test Fixture
template <typename A>
class AccumulatorTest : public ::testing::Test {
  public:
    typedef typename alps::accumulators::value_type<typename A::accumulator_type>::type value_type;

    unsigned int nsamples;
    std::string h5name;

    AccumulatorTest() : nsamples(0) {}

    // Add measurements to an existing file, or start a new one if the name is given
    void add_samples(const unsigned int howmany, const char* fname=0)
    {
        alps::accumulators::accumulator_set measurements;

        if (fname) {
            // Initialize new name
            h5name = fname;
            nsamples = 0;
            // and create an accumulator
            measurements<<A("one_half");
        } else {
            // Or read measurements
            if (h5name.empty())
                throw std::logic_error("Incorrect test usage: call add_samples(n,fname) first.");
        
            alps::hdf5::archive ar(h5name,"r");
            ar["measurements"] >> measurements;
        }

        // Generate more samples
        for(int count=0; count<howmany; ++count){
            measurements["one_half"] << 0.5;
        }
        nsamples+=howmany;

        // Save the samples
        alps::hdf5::archive ar(h5name,"w");
        ar["measurements"] << measurements;
    }

    void test_samples()
    {
        if (h5name.empty())
            throw std::logic_error("Incorrect test usage: call add_samples(n,fname) first.");
        
        alps::accumulators::accumulator_set measurements;
        alps::hdf5::archive ar(h5name,"r");
        ar["measurements"] >> measurements;

        alps::accumulators::result_set results(measurements);
        const alps::accumulators::result_wrapper& res=results["one_half"];
        value_type xmean=res.mean<value_type>();
        
        EXPECT_EQ(value_type(0.5), xmean);
        EXPECT_EQ(nsamples, res.count());
    }
};

typedef ::testing::Types<
    alps::accumulators::MeanAccumulator<double>,
    alps::accumulators::NoBinningAccumulator<double>,
    alps::accumulators::LogBinningAccumulator<double>,
    alps::accumulators::FullBinningAccumulator<double> > MyTypes;

TYPED_TEST_CASE(AccumulatorTest, MyTypes);

// Saving and loading only
TYPED_TEST(AccumulatorTest,SaveLoad)
{
    this->add_samples(1000, "saveload.h5");
    this->test_samples();
}

// Saving, adding and loading
TYPED_TEST(AccumulatorTest,SaveAddLoad)
{
    this->add_samples(1000, "saveload.h5");
    this->add_samples(500);
    this->test_samples();
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
