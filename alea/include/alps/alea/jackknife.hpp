/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <alps/alea/core.hpp>
#include <alps/alea/util.hpp>
#include <alps/alea/batch.hpp>

// Forward declarations

namespace alps { namespace alea {
}}

// Actual declarations

namespace alps { namespace alea {

template <typename T>
class jackknife_result
    : public batch_data<T>
{
public:
    template <typename InT>
    jackknife_result(const batch_data<InT> &in, transform<InT, T> &tf)
        : batch_data<T>(in.num_batches(), tf.out_size(in.size()))
    {
        // compute batch sums
        column<InT> sum_batch = in.batch_value().rowwise().sum;
        ptrdiff_t sum_count = in.batch_count().sum();

        // compute leave-one-out statistics and transforms
        column<InT> leaveout(in.size());
        for (size_t i = 0; i != in.num_batches(); ++i) {
            leaveout = (sum_batch - in.batch_value().col(i))
                                        / (sum_count - in.batch_count()(i));
            tf(sink<const InT>(leaveout.data(), leaveout.rows()),
            sink<T>(this->batch_value().col(i).data(), this->batch_value().rows()));
        }
        this->batch_value().colwise() *=
                this->batch_count().template cast<ptrdiff_t>().array() - sum_count;

        // compute transform of mean
        column<T> mean_result(this->batch_value().rows());
        sum_batch /= sum_count;
        tf(sink<const InT>(sum_batch.data(), sum_batch.rows()),
           sink<T>(mean_result.data(), mean_result.rows()));
        this->batch_value().rowwise() += mean_result * sum_count;

        this->batch_count() = in.batch_count();
    }
};

extern template class jackknife_result<double>;

}}
