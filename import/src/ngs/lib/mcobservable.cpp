/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/hdf5/archive.hpp>
#include <alps/ngs/mcobservable.hpp>

#include <alps/alea/observable.h>

#include <vector>
#include <valarray>
#include <iostream>

namespace alps {

    mcobservable::mcobservable()
        : impl_(NULL) 
    {}

    mcobservable::mcobservable(Observable const * obs) {
        ref_cnt_[impl_ = obs->clone()] = 1;
    }

    mcobservable::mcobservable(mcobservable const & rhs) {
        ++ref_cnt_[impl_ = rhs.impl_];
    }

    mcobservable::~mcobservable() {
        if (impl_ && !--ref_cnt_[impl_])
            delete impl_;
    }

    mcobservable & mcobservable::operator=(mcobservable rhs) {
        if (impl_ && !--ref_cnt_[impl_])
            delete impl_;
        ++ref_cnt_[impl_ = rhs.impl_];
        return *this;
    }

    Observable * mcobservable::get_impl() {
        return impl_;
    }

    Observable const * mcobservable::get_impl() const {
        return impl_;
    }

    std::string const & mcobservable::name() const {
        return impl_->name();
    }

     template<> ALPS_DECL mcobservable & mcobservable::operator<< <double>(double const & value) {
        (*impl_) << value;
        return *this;
    }

     template<> ALPS_DECL mcobservable & mcobservable::operator<< <std::vector<double> >(std::vector<double>  const & value) {
        std::valarray<double> varr(value.size());
        std::copy(value.begin(), value.end(), &varr[0]);
        (*impl_) << varr;
        return *this;
    }

     template<> ALPS_DECL mcobservable & mcobservable::operator<< <std::valarray<double> >(std::valarray<double>  const & value) {
        (*impl_) << value;
        return *this;
    }

    void mcobservable::save(hdf5::archive & ar) const {
        impl_->save(ar);
    }

    void mcobservable::load(hdf5::archive & ar) {
        impl_->load(ar);
    }

    void mcobservable::merge(mcobservable const & obs) {
        if (!impl_->can_merge()) {
            Observable* unmergeable = impl_;
            ++ref_cnt_[impl_ = unmergeable->convert_mergeable()];
            if (!--ref_cnt_[unmergeable])
                delete unmergeable;
        }
        impl_->merge(*obs.get_impl());
    }

    void mcobservable::output(std::ostream & os) const {
        os << *(impl_);
    }

    std::map<Observable *, std::size_t> mcobservable::ref_cnt_;

    std::ostream & operator<<(std::ostream & os, mcobservable const & obs) {
        obs.output(os);
        return os;
    }

}
