/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_DETAIL_PARAMS_IMPL_BASE_HPP
#define ALPS_DETAIL_PARAMS_IMPL_BASE_HPP

#warning this file is deprecated

#include <alps/config.hpp>

#include <string>

namespace alps {

    namespace detail {

        class params_impl_base {

            public:
                
                virtual ~params_impl_base() {};
            
                virtual std::size_t size() const = 0;

                virtual std::vector<std::string> keys() const = 0;

                virtual param operator[](std::string const &) = 0;

                virtual param const operator[](std::string const &) const = 0;

                virtual bool defined(std::string const &) const = 0;

                virtual void save(hdf5::archive &) const = 0;

                virtual void load(hdf5::archive &) = 0;
                
                virtual params_impl_base * clone() = 0;

                #ifdef ALPS_HAVE_MPI
                    virtual void broadcast(int root) = 0;
                #endif

        };

    }
}

#endif
