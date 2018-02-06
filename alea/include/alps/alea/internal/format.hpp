/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <cassert>
#include <ostream>
#include <memory>

namespace alps { namespace alea { namespace internal {

/**
 * Undoes all formatting flag changes on stream when it goes out of scope.
 *
 * Allows RAII-type use of a stream inside of a function while making sure
 * that no format changes are visible outside of it:
 *
 *     void dump(std::ostream &out) {
 *         alps::alea::internal::format_sentry sentry(out);
 *         // flag changes here
 *         // ...
 *     } // restores flags on exit
 *
 */
class format_sentry
{
public:
    explicit format_sentry(std::ostream &str)
        : stream_(str)
        , saved_(NULL)
    {
        // save all flags to auxiliary ios_base instance
        saved_.copyfmt(stream_);
    }

    ~format_sentry()
    {
        // restore flags
        stream_.copyfmt(saved_);
    }

    std::ios::fmtflags saved_flags() const { return saved_.flags(); }

    std::ostream &stream() { return stream_; }

    const std::ostream &stream() const { return stream_; }

private:
    // make it non-copyable and -movable
    format_sentry(const format_sentry &) = delete;
    format_sentry &operator=(const format_sentry &) = delete;

    std::ostream &stream_;
    std::ios saved_;
};

/**
 * Allows the addition of custom formatting state to I/O stream objects.
 *
 * This class allows to store one instance of type `T` with each `std::ios_base`
 * instance via the `get()` method.  This in turn can be used to implement
 * user-defined I/O manipulators that modify these format flags and are then
 * used by stream operations with user-defined objects.
 */
template <typename T>
class format_registry
{
public:
    /** Returns reference to format object for stream, creating it if needed. */
    static T &get(std::ios_base &stream, T init = T())
    {
        T *&format = get_format(stream);

        // if format is not there, we should add it
        if (format == nullptr) {
            format = new T(init);     // must be copy constructible
            stream.register_callback(callback, get_xindex());
        }
        return *format;
    }

    /** Returns value of format object for stream or fallback if not present. */
    static T value(const std::ios_base &stream, T fallback = T())
    {
        // HACK: pword() has no const variant, so we have to do this
        T *&format = get_format(const_cast<std::ios_base &>(stream));

        if (format == nullptr)
            return fallback;
        else
            return format;
    }

    /** Removes format object from stream */
    static void remove(std::ios_base &stream)
    {
        callback(std::ios::erase_event, stream, get_xindex());
    }

protected:
    /** Manages lifecycle using the std::ios_base call-back mechanism */
    static void callback(std::ios::event event, std::ios_base &stream,
                         int xindex)
    {
        if (xindex != get_xindex()) { assert(false); }
        T *&self = get_format(stream);

        switch (event) {
        case std::ios::erase_event:
            // this will be called in destructor, so we MUST NOT throw
            // exceptions
            if (self != nullptr) {
                delete self;
                self = nullptr;  // to be sure
            }
            break;

        case std::ios::copyfmt_event:
            // copyfmt first triggers an erase_event, which we use to clean up
            // the memory, then does a shallow copy, and then triggers this
            // event.  This means that self here will point to the original
            // instance.
            self = new T(*self);
            break;

        case std::ios::imbue_event:
            // FIXME
            break;
        }
    }

    static T *& get_format(std::ios_base &stream)
    {
        return reinterpret_cast<T *&>(stream.pword(get_xindex()));
    }

    static int get_xindex()
    {
        // guaranteed to be distinct since for different iomanip's since their
        // T will be different.
        static int xindex = std::ios_base::xalloc();
        return xindex;
    }

private:
    format_registry() = delete;
};

}}}
