/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <cstddef>
#include <cassert>
#include <stdexcept>
#include <complex>

#include <vector>
#include <Eigen/Dense>

#include <alps/alea/core.hpp>
#include <alps/alea/util.hpp>

namespace alps { namespace alea {

// Forward declarations

struct size_mismatch;
struct unsupported_operation;
template <typename T> class sink;
template <typename T> struct computed;
template <typename T, typename E> class computed_adapter;

// Actual declarations

/** Promote type to computed result */
template <typename T, typename Estimator>
class computed_adapter;

template <typename T>
class computed_adapter<T, T>
    : public computed<T>
{
public:
    typedef T value_type;

public:
    computed_adapter(T in) : in_(in) { }

    size_t size() const { return 1; }

    void add_to( sink<T> out)
    {
        if (out.size() != 1)
            throw size_mismatch();
        out.data()[0] += in_;
    }

    ~computed_adapter() { }

private:
    T in_;
};

template <typename T>
class computed_adapter< T, std::vector<T> >
    : public computed<T>
{
public:
    typedef T value_type;

public:
    computed_adapter(const std::vector<T> &in) : in_(in) { }

    size_t size() const { return in_.size(); }

    void add_to( sink<T> out)
    {
        if (out.size() != in_.size())
            throw size_mismatch();
        for (size_t i = 0; i != in_.size(); ++i)
            out.data()[i] += in_[i];
    }

    ~computed_adapter() { }

private:
    const std::vector<T> &in_;
};


template <typename T>
class computed_adapter< T, column<T> >
    : public computed<T>
{
public:
    typedef T value_type;

public:
    computed_adapter(const column<T> &in) : in_(in) { }

    size_t size() const { return in_.size(); }

    void add_to( sink<T> out)
    {
        if (out.size() != (size_t)in_.rows())
            throw size_mismatch();
        for (size_t i = 0; i != (size_t)in_.rows(); ++i)
            out.data()[i] += in_(i);
    }

    ~computed_adapter() { }

private:
    const column<T> &in_;
};

/**
 * Proxy object for computed results.
 */
template <typename T, typename Parent>
class computed_member
    : public computed<T>
{
public:
    typedef T value_type;
    typedef void (Parent::*adder_type)(sink<T>);

public:
    computed_member(Parent &parent, adder_type adder, size_t size)
        : parent_(parent)
        , adder_(adder)
        , size_(size)
    { }

    size_t size() const { return size_; }

    void add_to(sink<T> out) { (parent_.*adder_)(out); }

    const Parent &parent() const { return parent_; }

    const adder_type &adder() const { return adder_; }

    ~computed_member() { }

private:
    Parent &parent_;
    adder_type adder_;
    size_t size_;
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
    typedef void (Parent::*adder_type)(sink<T>) const;

public:
    computed_cmember(const Parent &parent, adder_type adder, size_t size)
        : parent_(parent)
        , adder_(adder)
        , size_(size)
    { }

    size_t size() const { return size_; }

    void add_to(sink<T> out) { (parent_.*adder_)(out); }

    void fast_add_to(sink<T> out) { (parent_.*adder_)(out); }

    const Parent &parent() const { return parent_; }

    const adder_type &adder() const { return adder_; }

    ~computed_cmember() { }

private:
    const Parent &parent_;
    adder_type adder_;
    size_t size_;
};

}}
