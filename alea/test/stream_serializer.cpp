#include <alps/alea/plugin/stream_serializer.hpp>

#include "gtest/gtest.h"
#include "dataset.hpp"

#include <vector>
#include <queue>
#include <cstdint>

// Mock class that emulates Boost/HPX serialization archive
class MockArchive {

public:

    MockArchive() {}

    // Store simple types
    MockArchive & operator<<(double x) { store_fundamental(x); return *this; }
    MockArchive & operator<<(long x) { store_fundamental(x); return *this; }
    MockArchive & operator<<(unsigned long x) { store_fundamental(x); return *this; }
    MockArchive & operator<<(std::complex<double> x)
    {
        *this << x.real() << x.imag();
        return *this;
    }

    // Store complex_op<T>
    template<typename T>
    MockArchive & operator<<(const alps::alea::complex_op<T> &x)
    {
        *this << x.rere() << x.reim() << x.imre() << x.imim();
        return *this;
    }

    // Store ALEA results
    template <typename T>
    typename std::enable_if<alps::alea::is_alea_result<T>::value, MockArchive &>::type
    operator<<(const T &result)
    {
        save(*this, result, 0);
        return *this;
    }

    // Extract simple types
    MockArchive & operator>>(double &x) { extract_fundamental(x); return *this; }
    MockArchive & operator>>(long &x) { extract_fundamental(x); return *this; }
    MockArchive & operator>>(unsigned long &x) { extract_fundamental(x); return *this; }
    MockArchive & operator>>(std::complex<double> &x) {
      double r, i;
      extract_fundamental(r);
      extract_fundamental(i);
      x = std::complex<double>(r, i);
    }

    // Extract complex_op<T>
    template<typename T>
    MockArchive & operator>>(alps::alea::complex_op<T> &x)
    {
        *this >> x.rere() >> x.reim() >> x.imre() >> x.imim();
        return *this;
    }

    // Extract ALEA results
    template <typename T>
    typename std::enable_if<alps::alea::is_alea_result<T>::value, MockArchive &>::type
    operator>>(T &result)
    {
        load(*this, result, 0);
        return *this;
    }

private:

    template<typename T>
    void store_fundamental(T x) {
        std::int8_t* p = reinterpret_cast<std::int8_t*>(&x);
        for(int n = 0; n < sizeof(T); ++n)
            buf.push(*(p + n));
    }

    template<typename T>
    void extract_fundamental(T &x) {
        std::int8_t* p = reinterpret_cast<std::int8_t*>(&x);
        for(int n = 0; n < sizeof(T); ++n) {
            *(p + n) = buf.front();
            buf.pop();
        }
    }

    // FIFO container with raw byte representation of stored values
    std::queue<std::int8_t> buf;
};

template <typename Acc>
class twogauss_serialize_case
    : public ::testing::Test
{
public:
    typedef typename alps::alea::traits<Acc>::value_type value_type;
    typedef typename alps::alea::traits<Acc>::result_type result_type;

    twogauss_serialize_case() { }

    void test_result()
    {
        Acc in_acc(2);
        for (size_t i = 0; i != twogauss_count; ++i)
            in_acc << std::vector<value_type>{twogauss_data[i][0], twogauss_data[i][1]};
        auto in = in_acc.result();

        MockArchive archive;
        archive << in; // serialize

        Acc out_acc(2);
        auto out = out_acc.result();

        archive >> out; // deserialize

        EXPECT_EQ(in, out);
    }
};

using namespace alps::alea;

typedef ::testing::Types<
        mean_acc<double>
      , mean_acc<std::complex<double> >
      , var_acc<double>
      , var_acc<std::complex<double> >
      , var_acc<std::complex<double>, elliptic_var>
      , cov_acc<double>
      , cov_acc<std::complex<double> >
      , cov_acc<std::complex<double>, elliptic_var>
      , autocorr_acc<double>
      , autocorr_acc<std::complex<double> >
      , batch_acc<double>
      , batch_acc<std::complex<double> >
    > stream_serializable;

TYPED_TEST_CASE(twogauss_serialize_case, stream_serializable);
TYPED_TEST(twogauss_serialize_case, test_result) { this->test_result(); }
