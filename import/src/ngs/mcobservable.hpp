/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_NGS_MCOBSERVABLE_HPP
#define ALPS_NGS_MCOBSERVABLE_HPP

#include <alps/hdf5/archive.hpp>
#include <alps/ngs/config.hpp>

#include <alps/alea/observable_fwd.hpp>

#include <map>
#include <iostream>

namespace alps {

    class ALPS_DECL mcobservable {

        public:

            mcobservable();
            mcobservable(Observable const * obs);
            mcobservable(mcobservable const & rhs);

            virtual ~mcobservable();

            mcobservable & operator=(mcobservable rhs);

            Observable * get_impl();

            Observable const * get_impl() const;

            std::string const & name() const;

            template<typename T> mcobservable & operator<<(T const & value);

            void save(hdf5::archive & ar) const;
            void load(hdf5::archive & ar);

            void merge(mcobservable const &);

            void output(std::ostream & os) const;

        private:

            Observable * impl_;
            static std::map<Observable *, std::size_t> ref_cnt_;

    };

    std::ostream & operator<<(std::ostream & os, mcobservable const & obs);

}

#endif
