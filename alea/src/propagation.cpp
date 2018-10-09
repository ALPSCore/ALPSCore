/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#include <alps/alea/propagation.hpp>

#include <iostream>

namespace alps { namespace alea {

template <typename T>
typename eigen<T>::matrix jacobian(const transformer<T> &f, column<T> x, double dx)
{
    size_t in_size = f.in_size();
    size_t out_size = f.out_size();

    typename eigen<T>::matrix result(out_size, in_size);
    for (size_t j = 0; j != in_size; ++j) {
        x(j) += dx;
        result.col(j) = f(x);
        x(j) -= dx;
    }
    result.colwise() -= f(x);
    result.array() /= dx;
    return result;
}

template eigen<double>::matrix jacobian(
            const transformer<double> &, column<double>, double);
template eigen< std::complex<double> >::matrix jacobian(
            const transformer<std::complex<double> > &, column<std::complex<double> >,
            double);


template <typename T>
batch_data<T> jackknife(const batch_data<T> &in, const transformer<T> &tf)
{
    // compute batch sums
    if (tf.in_size() != in.size())
        throw size_mismatch();

    batch_data<T> res(tf.out_size(), in.num_batches());
    column<T> sum_batch = in.batch().rowwise().sum();
    ptrdiff_t sum_count = in.count().sum();

    // compute leave-one-out statistics and transforms
    column<T> leaveout(in.size());
    for (size_t i = 0; i != in.num_batches(); ++i) {
        leaveout = (sum_batch - in.batch().col(i))
                                    / (sum_count - in.count()(i));
        res.batch().col(i) = tf(leaveout);
    }

    res.count() = in.count();

    // Since sum_count and res.count().array() are unsigned values,
    // (res.count().array() - sum_count) would be an array with huge positive elements.
    res.batch().array().rowwise() *=
                        -(sum_count - res.count().array()).template cast<T>();

    // compute transform of mean
    sum_batch /= sum_count;
    column<T> mean_result = tf(sum_batch);
    res.batch().colwise() += mean_result * sum_count;

    return res;
}

template batch_data<double> jackknife(const batch_data<double> &in,
                                      const transformer<double> &tf);
template batch_data<std::complex<double> > jackknife(
                                const batch_data<std::complex<double> > &in,
                                const transformer<std::complex<double> > &tf);

}}

