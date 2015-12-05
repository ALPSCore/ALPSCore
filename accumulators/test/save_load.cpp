/*
 * Copyright (C) 1998-2015 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <stdexcept>

#include "alps/config.hpp"
#include "alps/accumulators.hpp"
#include "alps/utilities/temporary_filename.hpp"
#include "alps/hdf5.hpp"
#include "gtest/gtest.h"
// Test for saving/restoring accumulators and results to/from archives.
double prec=1e-12;

template <typename T>
::testing::AssertionResult AreAllElementsNear(T a, T b, float delta) {
    if (std::abs(a - b) < delta)
        return ::testing::AssertionSuccess();
    else 
        return ::testing::AssertionFailure() << "doubles differ by more than " << delta;
}


template <typename T>
::testing::AssertionResult AreAllElementsNear(const std::vector<T>& a, const std::vector<T>& b, float delta) {
  double res=0.0;
  for (int i=0; i<a.size(); ++i) res+=std::sqrt((a[i] - b[i]) * (a[i] - b[i])) / double(a.size());
  if (res < prec)
    return ::testing::AssertionSuccess();
  else
    return ::testing::AssertionFailure() << "Vectors differ by more than " << delta;
}

// Service functions to generate scalar or vector data points
template <typename T, typename U>
inline T get_datum(U val, T*)
{
  return val;
}

template <typename T, typename U>
inline std::vector<T> get_datum(U val, std::vector<T>*)
{
  return std::vector<T>(10, val);
}

// using Google Test Fixture
template <typename A>
class AccumulatorTest : public ::testing::Test {
  public:
    typedef typename alps::accumulators::value_type<typename A::accumulator_type>::type value_type;
    typedef A accumulator_type;

    int nsamples;
    const std::string h5name;

    AccumulatorTest() : nsamples(0), h5name(alps::temporary_filename("save_load")+".h5")
    { }
  
    // Add (constant) data to an accumulator
    void add_samples(alps::accumulators::accumulator_set& measurements,
                     const int howmany,
                     const double v)
    {
        for(int count=0; count<howmany; ++count){
            measurements["my_acc"] << get_datum(v, (value_type*)0);
        }
        nsamples += howmany;
    }

    // Compare the results only
    void test_results(const alps::accumulators::result_set& results,
                      const double v)
    {
        if (!results.has("my_acc")) {
            // No results implies there were no data saved
            EXPECT_EQ(0, nsamples) << "Data generated, but no results are present";
            return;
        }
        EXPECT_NE(0,nsamples) << "No data generated, but results are present";

        const alps::accumulators::result_wrapper& res=results["my_acc"];
        value_type xmean=res.mean<value_type>();
        
        EXPECT_EQ(nsamples, res.count());
//        EXPECT_EQ(get_datum(v, (value_type*)0), xmean);
        EXPECT_TRUE(AreAllElementsNear(get_datum(v, (value_type*)0), xmean, prec));
    }

    // Compare the expected and actual accumulator data
    void test_samples(alps::accumulators::accumulator_set& measurements,
                      const double v)
    {
        alps::accumulators::result_set results(measurements);
        test_results(results,v);
    }

    // Save measurements/results/whatever
    template <typename T>
    void save(const T& measurements) const
    {
        alps::hdf5::archive ar(h5name,"w");
        ar["mydata"] << measurements;
    }        
  
    // Load measurements/results/whatever
    template <typename T>
    void load(T& measurements) const
    {
        alps::hdf5::archive ar(h5name,"w");
        ar["mydata"] >> measurements;
    }        
};

typedef std::vector<double> doublevec;
typedef std::vector<float> floatvec;

typedef ::testing::Types<
    alps::accumulators::MeanAccumulator<double>,
    alps::accumulators::NoBinningAccumulator<double>,
    alps::accumulators::LogBinningAccumulator<double>,
    alps::accumulators::FullBinningAccumulator<double>,

    alps::accumulators::MeanAccumulator<float>,
    alps::accumulators::NoBinningAccumulator<float>,
    alps::accumulators::LogBinningAccumulator<float>,
    alps::accumulators::FullBinningAccumulator<float>,

    alps::accumulators::MeanAccumulator<floatvec>,
    alps::accumulators::NoBinningAccumulator<floatvec>,
    alps::accumulators::LogBinningAccumulator<floatvec>,
    alps::accumulators::FullBinningAccumulator<floatvec>,

    alps::accumulators::MeanAccumulator<doublevec>,
    alps::accumulators::NoBinningAccumulator<doublevec>,
    alps::accumulators::LogBinningAccumulator<doublevec>,
    alps::accumulators::FullBinningAccumulator<doublevec>
    > MyTypes;


TYPED_TEST_CASE(AccumulatorTest, MyTypes);


// Saving and loading only
TYPED_TEST(AccumulatorTest,SaveLoad)
{
    alps::accumulators::accumulator_set measurements;
    typedef typename TestFixture::accumulator_type accumulator_type;
    measurements << accumulator_type("my_acc");
    
    this->add_samples(measurements, 1000, 0.5);
    this->save(measurements);

    alps::accumulators::accumulator_set new_measurements;
    this->load(new_measurements);

    this->test_samples(new_measurements, 0.5);
}

// Saving and loading results
TYPED_TEST(AccumulatorTest,SaveLoadResults)
{
    alps::accumulators::accumulator_set measurements;
    typedef typename TestFixture::accumulator_type accumulator_type;
    measurements << accumulator_type("my_acc");
    
    this->add_samples(measurements, 1000, 0.5);
    alps::accumulators::result_set results(measurements);
    
    this->save(results);

    alps::accumulators::result_set new_results;
    this->load(new_results);

    this->test_results(new_results, 0.5);
}

// Saving and loading a single sample, which is different in float and double representation,
// to be sure that the correct type is restored.
TYPED_TEST(AccumulatorTest,SaveLoadTypecheck)
{
    alps::accumulators::accumulator_set measurements;
    typedef typename TestFixture::accumulator_type accumulator_type;
    measurements << accumulator_type("my_acc");
    
    this->add_samples(measurements, 1, 0.3);
    this->save(measurements);

    alps::accumulators::accumulator_set new_measurements;
    this->load(new_measurements);

    this->test_samples(new_measurements, 0.3);
}

// Saving and loading results, checking that the type is restored correctly
TYPED_TEST(AccumulatorTest,SaveLoadResultsTypecheck)
{
    alps::accumulators::accumulator_set measurements;
    typedef typename TestFixture::accumulator_type accumulator_type;
    measurements << accumulator_type("my_acc");
    
    this->add_samples(measurements, 1, 0.3);
    alps::accumulators::result_set results(measurements);
    
    this->save(results);

    alps::accumulators::result_set new_results;
    this->load(new_results);

    this->test_results(new_results, 0.3);
}

// Saving and loading an empty accumulator
TYPED_TEST(AccumulatorTest,SaveLoadEmpty)
{
    alps::accumulators::accumulator_set measurements;
    typedef typename TestFixture::accumulator_type accumulator_type;
    measurements << accumulator_type("my_acc");
    
    this->add_samples(measurements, 0, 0.3);
    this->save(measurements);

    alps::accumulators::accumulator_set new_measurements;
    this->load(new_measurements);

    this->test_samples(new_measurements, 0.3);
}

// Saving and loading an empty result
TYPED_TEST(AccumulatorTest,SaveLoadEmptyResults)
{
    alps::accumulators::accumulator_set measurements;
    typedef typename TestFixture::accumulator_type accumulator_type;
    measurements << accumulator_type("my_acc");
    
    this->add_samples(measurements, 0, 0.5);
    alps::accumulators::result_set results(measurements);
    
    this->save(results);

    alps::accumulators::result_set new_results;
    this->load(new_results);

    this->test_results(new_results, 0.5);
}

// Saving empty accumulator
TYPED_TEST(AccumulatorTest,SaveEmpty)
{
    alps::accumulators::accumulator_set measurements;
    typedef typename TestFixture::accumulator_type accumulator_type;
    measurements << accumulator_type("my_acc");
    
    this->add_samples(measurements, 0, 0.3);
    this->save(measurements);
}

// Saving an empty result
TYPED_TEST(AccumulatorTest,SaveEmptyResults)
{
    alps::accumulators::accumulator_set measurements;
    typedef typename TestFixture::accumulator_type accumulator_type;
    measurements << accumulator_type("my_acc");
    
    this->add_samples(measurements, 0, 0.5);
    alps::accumulators::result_set results(measurements);
    
    this->save(results);
}

// Saving, loading and adding
TYPED_TEST(AccumulatorTest,SaveLoadAdd)
{
    alps::accumulators::accumulator_set measurements;
    typedef typename TestFixture::accumulator_type accumulator_type;
    measurements << accumulator_type("my_acc");
    
    this->add_samples(measurements, 1000, 0.5);
    this->save(measurements);

    alps::accumulators::accumulator_set new_measurements;
    this->load(new_measurements);
    this->add_samples(new_measurements, 500, 0.5);
    
    this->test_samples(new_measurements, 0.5);
}
