/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <alps/alea/core.hpp>
#include <iosfwd>

namespace alps { namespace alea { namespace util {

struct null_serializer
    : public serializer
{
public:
    null_serializer() { }

    void enter(const std::string &) override { }

    void exit() override { }

    void write(const std::string &, ndview<const double>) override { }

    void write(const std::string &, ndview<const std::complex<double>>) override { }

    void write(const std::string &, ndview<const complex_op<double>>) override { }

    void write(const std::string &, ndview<const long>) override { }

    void write(const std::string &, ndview<const unsigned long>) override { }

    null_serializer *clone() override { return new null_serializer(*this); }

    ~null_serializer() { }
};

class debug_serializer
    : public serializer
{
public:
    debug_serializer(std::ostream &stream)
        : str_(stream)
        , inner_(nullptr)
    {
        static null_serializer null_instance;
        inner_ = &null_instance;
    }

    debug_serializer(std::ostream &stream, serializer &inner)
        : str_(stream)
        , inner_(&inner)
    { }

    void enter(const std::string &group) override
    {
        str_ << "debug_serializer: entering group '" << group << "'\n";
        inner_->enter(group);
    }

    void exit() override
    {
        str_ << "debug_serializer: exitting group.\n";
        inner_->exit();
    }

    void write(const std::string &key, ndview<const double> value) override {
        do_write(key, value);
    }

    void write(const std::string &key, ndview<const std::complex<double>> value) override {
        do_write(key, value);
    }

    void write(const std::string &key, ndview<const complex_op<double>> value) override {
        do_write(key, value);
    }

    void write(const std::string &key, ndview<const long> value) override {
        do_write(key, value);
    }

    void write(const std::string &key, ndview<const unsigned long> value) override {
        do_write(key, value);
    }

protected:
    template <typename T>
    void do_write(const std::string &key, ndview<const T> value)
    {
        str_ << "debug_serializer: writing data set '" << key
             << "' of shape {";
        for (size_t i = 0; i != value.ndim(); ++i)
            str_ << value.shape()[i] << (i == 0 ? "" : ",");
        str_ << "}\n";
        inner_->write(key, value);
    }

private:
    std::ostream &str_;
    serializer *inner_;
};

}}}
