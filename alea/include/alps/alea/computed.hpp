/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <cstddef>
#include <cassert>
#include <stdexcept>
#include <complex>

#include <vector>
#include <array>
#include <Eigen/Dense>

#include <alps/alea/core.hpp>
#include <alps/alea/util.hpp>

// Forward declarations

namespace alps { namespace alea {
    template <typename T> class value_adapter;
    template <typename T> class vector_adapter;
    template <typename T, size_t N> class array_adapter;
    template <typename T, typename Derived> class eigen_adapter;
}}

// Actual declarations

namespace alps { namespace alea {

template <typename Derived>
eigen_adapter<typename Derived::Scalar, Derived> make_adapter(
                                    const Eigen::DenseBase<Derived> &in)
{
    return eigen_adapter<typename Derived::Scalar, Derived>(in);
}

template <typename T>
vector_adapter<T> make_adapter(const std::vector<T> &v)
{
    return vector_adapter<T>(v);
}

template<typename T, size_t N>
array_adapter<T,N> make_array_adapter(const std::array<T, N> &a){
  return array_adapter<T,N>(a);
}

template <typename T>
class value_adapter
    : public computed<T>
{
public:
    typedef T value_type;

public:
    value_adapter(T in) : in_(in) { }

    size_t size() const { return 1; }

    void add_to(view<T> out) const
    {
        if (out.size() != 1)
            throw size_mismatch();
        out.data()[0] += in_;
    }

    ~value_adapter() { }

private:
    T in_;
};

inline value_adapter<long> make_adapter(size_t v)  // FIXME
{
    return value_adapter<long>(v);
}

inline value_adapter<long> make_adapter(long v)
{
    return value_adapter<long>(v);
}

inline value_adapter<double> make_adapter(double v)
{
    return value_adapter<double>(v);
}



template <typename T>
class vector_adapter
    : public computed<T>
{
public:
    typedef T value_type;

public:
    vector_adapter(const std::vector<T> &in) : in_(in) { }

    size_t size() const { return in_.size(); }

    void add_to(view<T> out) const
    {
        if (out.size() != in_.size())
            throw size_mismatch();
        for (size_t i = 0; i != in_.size(); ++i)
            out.data()[i] += in_[i];
    }

    ~vector_adapter() { }

private:
    const std::vector<T> &in_;
};

template <typename T, size_t N> class array_adapter : public computed<T> {
public:
  typedef T value_type;

public:
  array_adapter(const std::array<T, N> &in) : in_(in) {}

  size_t size() const { return in_.size(); }

  void add_to(view<T> out) const {
    if (out.size() != in_.size())
      throw size_mismatch();
    for (size_t i = 0; i != in_.size(); ++i)
      out.data()[i] += in_[i];
  }

  ~array_adapter() {}

private:
  const std::array<T, N> &in_;
};

template <typename T, typename Derived>
class eigen_adapter
    : public computed<T>
{
public:
    typedef T value_type;

public:
    eigen_adapter(const Eigen::DenseBase<Derived> &in)
        : in_(in)
    {
        EIGEN_STATIC_ASSERT_VECTOR_ONLY(Eigen::DenseBase<Derived>);
        static_assert(std::is_same<T, typename Derived::Scalar>::value,
                      "Type mismatch -- use explicit cast");
    }

    size_t size() const { return in_.size(); }

    void add_to(view<T> out) const
    {
        if (out.size() != (size_t)in_.rows())
            throw size_mismatch();

        typename eigen<T>::col_map out_map(out.data(), out.size());
        out_map += in_;
    }

    ~eigen_adapter() { }

private:
    const Eigen::DenseBase<Derived> &in_;
};

/**
 * Proxy object for computed results.
 */
template <typename T, typename Parent>
class computed_cmember
    : public computed<T>
{
public:
    typedef T value_type;
    typedef void (Parent::*adder_type)(view<T>) const;

public:
    computed_cmember(const Parent &parent, adder_type adder, size_t size)
        : parent_(parent)
        , adder_(adder)
        , size_(size)
    { }

    size_t size() const { return size_; }

    void add_to(view<T> out) const { (parent_.*adder_)(out); }

    void fast_add_to(view<T> out) { (parent_.*adder_)(out); }

    const Parent &parent() const { return parent_; }

    const adder_type &adder() const { return adder_; }

    ~computed_cmember() { }

private:
    const Parent &parent_;
    adder_type adder_;
    size_t size_;
};

/** Add scalar value to accumulator */
template<typename AccType>
typename std::enable_if<is_alea_acc<AccType>::value, AccType&>::type
operator<<(AccType& acc, const typename AccType::value_type& v){
  return acc << value_adapter<typename AccType::value_type>(v);
}

/** Add Eigen vector-valued expression to accumulator */
template<typename AccType, typename Derived>
typename std::enable_if<is_alea_acc<AccType>::value, AccType&>::type
operator<<(AccType& acc, const Eigen::DenseBase<Derived>& v){
  return acc << eigen_adapter<typename AccType::value_type, Derived>(v);
}

/** Add `std::vector` to accumulator */
template<typename AccType>
typename std::enable_if<is_alea_acc<AccType>::value, AccType&>::type
operator<<(AccType& acc, const std::vector<typename AccType::value_type>& v){
  return acc << vector_adapter<typename AccType::value_type>(v);
}

/** Add `std::array` to accumulator */
template<typename AccType, size_t N>
typename std::enable_if<is_alea_acc<AccType>::value, AccType&>::type
operator<<(AccType& acc, const std::array<typename AccType::value_type, N>& v){
  return acc << array_adapter<typename AccType::value_type, N>(v);
}

}}
