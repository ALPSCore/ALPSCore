#include <alps/alea/batch.hpp>

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


void galois_hopper::advance()
{
    if (level_ == 0)
        advance_fill();
    else
        advance_galois();
}

void galois_hopper::advance_fill()
{
    assert(level_ == 0);
    ++current_;
    ++level_pos_;

    // we have filled all the elements, switch to Galois mode
    if (current_ == size_)
        reset(true);
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
        skip_ = 1;
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

template <typename T>
void batch_data<T>::get_mean(sink<T> out) const
{
    typename eigen<T>::col_map out_map(out.data(), out.size());
    out_map += batch_.rowwise().sum() / count();
}

template <typename T>
void batch_data<T>::get_var(sink<error_type> out) const
{
    typename eigen<error_type>::col_map out_map(out.data(), out.size());
    //out_map += batch_.rowwise().sum() / count();
}

template class batch_data<double>;
template class batch_data<std::complex<double> >;


template <typename T>
batch_acc<T>::batch_acc(size_t size, size_t num_batches, size_t base_size)
    : batch_data<T>(size, num_batches)
    , base_size_(base_size)
    , cursor_(num_batches)
{
    if (num_batches % 2 != 0) {
        throw std::runtime_error("Number of batches must be even to allow "
                                 "for rebatching.");
    }
}

template <typename T>
void batch_acc<T>::reset()
{
    batch_data<T>::reset();
    cursor_.reset();
}

template <typename T>
batch_acc<T> &batch_acc<T>::operator<<(computed<T> &source)
{
    // Since Eigen matrix are column-major, we can just pass the pointer
    source.add_to(sink<T>(this->batch_value().col(cursor_.current()).data(),
                          this->size()));
    this->batch_count()(cursor_.current()) += 1;

    // batch is full, move the cursor
    if (this->batch_count()(cursor_.current()) == current_batch_size())
        next_batch();

    return *this;
}

template <typename T>
void batch_acc<T>::next_batch()
{
    cursor_.advance();
    if (cursor_.merge_mode()) {
        // merge counts
        this->batch_count()(cursor_.merge_into()) += this->batch_count()(cursor_.current());
        this->batch_count()(cursor_.current()) = 0;

        // merge batches
        this->batch_value().col(cursor_.merge_into()) += this->batch_value().col(cursor_.current());
        this->batch_value().col(cursor_.current()).fill(0);
    }
}

template class batch_acc<double>;
//template class batch_acc<std::complex<double> >;

}}
