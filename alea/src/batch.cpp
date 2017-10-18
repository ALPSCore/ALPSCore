#include <alps/alea/batch.hpp>

#include <numeric>
#include <iostream>

namespace alps { namespace alea {

galois_hopper::galois_hopper(size_t size)
    : size_(size)
{
    if (size_ % 2 != 0)
        throw std::runtime_error("Number of batches must be even to allow "
                                 "for rebatching");
    reset();
}

void galois_hopper::reset(bool merge_mode)
{
    if (merge_mode) {
        level_ = 1;
        factor_ = 2;
        skip_ = 1;
    } else {
        level_ = 0;
        factor_ = 1;
        skip_ = 0;
    }
    cycle_ = 0;
    level_pos_ = 0;
    current_ = 0;
}

galois_hopper galois_hopper::operator++(int)
{
    galois_hopper save = *this;
    ++(*this);
    return save;
}

galois_hopper &galois_hopper::operator++()
{
    if (level_ == 0)
        advance_fill();
    else
        advance_galois();
    return *this;
}

void galois_hopper::advance_fill()
{
    assert(level_ == 0);
    ++current_;
    ++level_pos_;

    // we have filled all the elements, switch to Galois mode
    if (current_ == size_) {
        reset(true);
        ++cycle_;
    }
}

void galois_hopper::advance_galois()
{
    assert(level_ != 0);
    ++level_pos_;
    if (level_pos_ == size_/2) {
        ++level_;
        level_pos_ = 0;
        factor_ *= 2.;
        skip_ *= 2;
    }
    current_ = (current_ + 2 * skip_) % (size_ + 1);
    assert(current_ != size_);

    // We have completed the cycle. Make sure skip does not overflow
    if (current_ == 0 && merge_into() == 1) {
        assert(level_pos_ == 0);
        ++cycle_;
    }
}


template <typename T>
batch_data<T>::batch_data(size_t size, size_t num_batches)
    : batch_(size, num_batches)
    , count_(num_batches)
{
    reset();
}

template <typename T>
void batch_data<T>::reset()
{
    batch_.fill(0);
    count_.fill(0);
}

template class batch_data<double>;
template class batch_data<std::complex<double> >;


template <typename T>
batch_acc<T>::batch_acc(size_t size, size_t num_batches, size_t base_size)
    : base_size_(base_size)
    , cursor_(num_batches)
    , data_(size, num_batches)
    , offset_(num_batches)
{
    if (num_batches % 2 != 0) {
        throw std::runtime_error("Number of batches must be even to allow "
                                 "for rebatching.");
    }
}

template <typename T>
void batch_acc<T>::reset()
{
    data_.reset();
    cursor_.reset();

    for (size_t i = 0; i != data_.num_batches(); ++i)
        offset_[i] = i * base_size_;
}

template <typename T>
batch_acc<T> &batch_acc<T>::operator<<(computed<T> &source)
{
    // batch is full, move the cursor.
    // Doing this before the addition ensures no empty batches.
    if (data_.count()(cursor_.current()) == current_batch_size())
        next_batch();

    // Since Eigen matrix are column-major, we can just pass the pointer
    source.add_to(sink<T>(data_.batch().col(cursor_.current()).data(), size()));
    data_.count()(cursor_.current()) += 1;

    return *this;
}

template <typename T>
void batch_acc<T>::next_batch()
{
    ++cursor_;
    if (cursor_.merge_mode()) {
        // merge counts
        data_.count()(cursor_.merge_into()) += data_.count()(cursor_.current());
        data_.count()(cursor_.current()) = 0;

        // merge batches
        data_.batch().col(cursor_.merge_into()) +=
                                        data_.batch().col(cursor_.current());
        data_.batch().col(cursor_.current()).fill(0);

        // merge offsets
        offset_(cursor_.merge_into()) = std::min(offset_(cursor_.merge_into()),
                                                 offset_(cursor_.current()));
        offset_(cursor_.current()) = count();
    }
}

template <typename T>
void batch_acc<T>::get_mean(sink<T> out) const
{
    typename eigen<T>::col_map out_map(out.data(), out.size());
    out_map += data_.batch().rowwise().sum() / count();
}

template <typename T>
void batch_acc<T>::get_var(sink<error_type> out) const
{
    typename eigen<error_type>::col_map out_map(out.data(), out.size());
    //out_map += batch_.rowwise().sum() / count();
}

template class batch_acc<double>;
//template class batch_acc<std::complex<double> >;

}}
