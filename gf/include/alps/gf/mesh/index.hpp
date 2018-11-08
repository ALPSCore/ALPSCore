/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#ifdef ALPS_HAVE_MPI
#include <alps/gf/mpi_bcast.hpp>
#endif

namespace alps {namespace gf {

    /// A generic index
    template <typename X>
    class generic_index : boost::additive2<generic_index<X>, int,
        boost::unit_steppable< generic_index<X>,
        boost::totally_ordered2< generic_index<X>, int> > >
  {
    private:
    int index_;
    public:
    explicit generic_index(int i): index_(i) {}
    generic_index() : index_(0) {}
    void operator=(int i) { index_=i; }

    generic_index& operator++() { index_++; return *this; }
    generic_index& operator--() { index_--; return *this; }

    generic_index& operator+=(int i) { index_+=i; return *this; }
    generic_index& operator-=(int i) { index_-=i; return *this; }

    bool operator<(int x) const { return index_ <x; }
    bool operator>(int x) const { return index_ >x; }
    bool operator==(int x) const { return index_==x; }

    int operator()() const { return index_; }

#ifdef ALPS_HAVE_MPI
    void broadcast(const alps::mpi::communicator& comm, int root) {
            alps::mpi::broadcast(comm, index_, root);
        }
#endif
  };
}}
