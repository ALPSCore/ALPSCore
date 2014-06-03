/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_NGS_MUTEX_HPP
#define ALPS_NGS_MUTEX_HPP

#include <alps/ngs/thread_exceptions.hpp>

#include <boost/shared_ptr.hpp>

#ifndef ALPS_NGS_SINGLE_THREAD
    #include <boost/thread/mutex.hpp>
#endif

namespace alps {
    namespace detail {

        struct mutex_base {
            virtual ~mutex_base() {}

            virtual void lock() = 0;
            virtual bool try_lock() = 0;
            virtual void unlock() = 0;

        };

        template <typename T> struct mutex_derived : public mutex_base, public T {

            void lock() { T::lock(); }
            bool try_lock() { return T::try_lock(); }
            void unlock() { T::unlock(); }

        };

        class noop_mutex {

            public:

                noop_mutex()
                    : has_lock(false)
                {}

                void lock() {
                    if (has_lock)
                        throw boost::lock_error();
                    has_lock = true;
                }

                bool try_lock() {
                    if (has_lock)
                        return false;
                    else {
                        lock();
                        return true;
                    }
                }

                void unlock() {
                    if (!has_lock)
                        throw boost::lock_error();
                    has_lock = false;
                }

            private:

                bool has_lock;
        };
    }

    class mutex {

        public:

            template<typename T> mutex(T * arg)
                : has_lock(false)
                , impl(arg)
            {}

            void lock() {
                impl->lock();
                has_lock = true;
            }

            bool locked() {
                return has_lock;
            }

            bool try_lock() {
                bool success = impl->try_lock();
                if (success)
                    has_lock = true;
                return success;
            }

            void unlock() {
                impl->unlock();
                has_lock = false;
            }

        private:

            bool has_lock;
            boost::shared_ptr<detail::mutex_base> impl;
    };
    
    typedef detail::mutex_derived<detail::noop_mutex> noop_lockable;
    
    #ifndef ALPS_NGS_SINGLE_THREAD
        typedef detail::mutex_derived<boost::mutex> native_lockable;
    #endif
}

#endif
