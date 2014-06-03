/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#if !defined(ALPS_NGS_SCHEDULER_CONTROLTHREADSIM_NG_HPP) && !defined(ALPS_NGS_SINGLE_THREAD)
#define ALPS_NGS_SCHEDULER_CONTROLTHREADSIM_NG_HPP

#include <alps/ngs/api.hpp>

#include <boost/thread.hpp>

namespace alps {

    template<typename Impl> class controlthreadsim_ng : public Impl {
        public:
            controlthreadsim_ng(typename alps::parameters_type<Impl>::type const & p, std::size_t seed_offset = 0)
                : Impl(p, seed_offset)
                , m_status(Impl::initialized)
            {
                // TODO: this is ugly, but virtual functions do not work in constructor of base class
                Impl::data_mutex = mutex(new native_lockable());
                Impl::result_mutex = mutex(new native_lockable());
            }

            double fraction_completed() const {
                typename Impl::lock_guard data_lock(Impl::get_data_lock());
                return Impl::fraction_completed();
            }

            typename Impl::status_type status() const {
                return m_status;
            }
        
        protected:

            template<typename T> class atomic {
                public:

                    atomic() {}
                    atomic(T const & v): value(v) {}
                    atomic(atomic<T> const & v): value(v.value) {}

                    atomic<T> & operator=(T const & v) {
                        boost::lock_guard<boost::mutex> lock(atomic_mutex);
                        value = v;
                        return *this;
                    }

                    operator T() const {
                        boost::lock_guard<boost::mutex> lock(atomic_mutex);
                        return value;
                    }

                private:

                    T volatile value;
                    boost::mutex mutable atomic_mutex;
            };

            void on_unlock() {}

            void set_status(typename Impl::status_type status) {
                m_status = status;
            }

        private:

            atomic<typename Impl::status_type> m_status;
    };

}

#endif
