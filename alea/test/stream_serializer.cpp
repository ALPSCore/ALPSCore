/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#include <alps/alea/plugin/stream_serializer.hpp>

#include "gtest/gtest.h"
#include "dataset.hpp"

#include <vector>
#include <queue>

// Mock class that emulates Boost/HPX serialization archive
class mock_archive {

public:

    mock_archive() {}

    // Store simple types
    mock_archive & operator<<(double x) { store_fundamental(x); return *this; }
    mock_archive & operator<<(long x) { store_fundamental(x); return *this; }
    mock_archive & operator<<(unsigned long x) { store_fundamental(x); return *this; }
    mock_archive & operator<<(std::complex<double> x)
    {
        *this << x.real() << x.imag();
        return *this;
    }

    // Store complex_op<T>
    template<typename T>
    mock_archive & operator<<(const alps::alea::complex_op<T> &x)
    {
        *this << x.rere() << x.reim() << x.imre() << x.imim();
        return *this;
    }

    // Store ALEA results
    template <typename T>
    typename std::enable_if<alps::alea::is_alea_result<T>::value, mock_archive &>::type
    operator<<(const T &result)
    {
        save(*this, result, 0);
        return *this;
    }

    // Extract simple types
    mock_archive & operator>>(double &x) { extract_fundamental(x); return *this; }
    mock_archive & operator>>(long &x) { extract_fundamental(x); return *this; }
    mock_archive & operator>>(unsigned long &x) { extract_fundamental(x); return *this; }
    mock_archive & operator>>(std::complex<double> &x) {
      double r, i;
      extract_fundamental(r);
      extract_fundamental(i);
      x = std::complex<double>(r, i);
      return *this;
    }

    // Extract complex_op<T>
    template<typename T>
    mock_archive & operator>>(alps::alea::complex_op<T> &x)
    {
        *this >> x.rere() >> x.reim() >> x.imre() >> x.imim();
        return *this;
    }

    // Extract ALEA results
    template <typename T>
    typename std::enable_if<alps::alea::is_alea_result<T>::value, mock_archive &>::type
    operator>>(T &result)
    {
        load(*this, result, 0);
        return *this;
    }

private:

    template<typename T>
    void store_fundamental(T x) {
        unsigned char* p = reinterpret_cast<unsigned char*>(&x);
        for(size_t n = 0; n < sizeof(T); ++n)
            buf.push(*(p + n));
    }

    template<typename T>
    void extract_fundamental(T &x) {
        unsigned char* p = reinterpret_cast<unsigned char*>(&x);
        for(size_t n = 0; n < sizeof(T); ++n) {
            *(p + n) = buf.front();
            buf.pop();
        }
    }

    // FIFO container with raw byte representation of stored values
    std::queue<unsigned char> buf;
};

TEST(twogauss_serialize_case, mock_archive) {
    mock_archive archive;

    archive << (double)3.14
            << (long)-123456
            << (unsigned long)7890
            << std::complex<double>(0.5,0.75)
            << alps::alea::complex_op<double>(1, 2, 3, 4);

    double x = 0;
    archive >> x;
    EXPECT_EQ(3.14, x);
    long l = 0;
    archive >> l;
    EXPECT_EQ(-123456, l);
    unsigned long ul = 0;
    archive >> ul;
    EXPECT_EQ(7890U, ul);
    std::complex<double> c = 0;
    archive >> c;
    EXPECT_EQ(std::complex<double>(0.5,0.75), c);
    alps::alea::complex_op<double> co(0, 0, 0, 0);
    archive >> co;
    EXPECT_EQ(alps::alea::complex_op<double>(1, 2, 3, 4), co);
}

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
        std::cerr << alps::alea::PRINT_VERBOSE << "\nin\n" << in;

        mock_archive archive;
        archive << in; // serialize

        Acc out_acc(2);
        auto out = out_acc.result();

        archive >> out; // deserialize

        std::cerr << alps::alea::PRINT_VERBOSE << "\nout\n" << out << "\n";
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
