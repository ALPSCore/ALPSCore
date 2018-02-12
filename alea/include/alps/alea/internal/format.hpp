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
    class format_sentry;
    template <typename T> class format_registry;
}}}

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
        , saved_(nullptr)
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
 * Allows the use of custom formatting state to I/O stream objects.
 *
 * Retrieves a reference of the object of copy-constructable type `T`
 * associated with stream.  If no such object exists, it will first associate
 * `initial_value` with the stream.
 *
 * Example use:
 *
 *     #include <iostream>
 *     enum verbosity { TERSE, VERBOSE, DEBUG };
 *     class my_type;
 *     // ...
 *     std::ostream &operator<<(std::ostream &str, verbosity verb) {
 *         get_format(std::cerr, TERSE) = verb;
 *     }
 *     // ...
 *     std::ostream &operator<<(std::ostream &str, const my_type &obj) {
 *         if (get_format(std::cerr, TERSE) == VERBOSE) {
 *             // ...
 *         }
 *     }
 *
 * @see format_registry
 */
template <typename T>
T &get_format(std::ios_base &stream, T initial_value = T())
{
    return format_registry<T>::get(stream, initial_value);
}

/**
 * Allows the addition of custom formatting state to I/O stream objects.
 *
 * This class allows to store one instance of type `T` with each `std::ios_base`
 * instance via the `get()` method.  This in turn can be used to implement
 * user-defined I/O manipulators that modify these format flags and are then
 * used by stream operations with user-defined objects.
 *
 * @see get_format()
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
            delete self;
            self = nullptr;
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
        // T will be different. xalloc is threadsafe from C++14 onwards, however
        // C++11 guarantees that static initialization is threadsafe, so we're
        // fine.
        static int xindex = std::ios_base::xalloc();
        return xindex;
    }

private:
    format_registry() = delete;
};


// explicitly disallow some built-in types

template <> class format_registry<char> { };
template <> class format_registry<bool> { };
template <> class format_registry<short> { };
template <> class format_registry<unsigned short> { };
template <> class format_registry<int> { };
template <> class format_registry<unsigned int> { };
template <> class format_registry<long> { };
template <> class format_registry<unsigned long> { };
template <> class format_registry<float> { };
template <> class format_registry<double> { };

}}}
