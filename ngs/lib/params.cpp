/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2011 by Lukas Gamper <gamperl@gmail.com>                   *
 *                                                                                 *
 * This software is part of the ALPS libraries, published under the ALPS           *
 * Library License; you can use, redistribute it and/or modify it under            *
 * the terms of the license, either version 1 or (at your option) any later        *
 * version.                                                                        *
 *                                                                                 *
 * You should have received a copy of the ALPS Library License along with          *
 * the ALPS Libraries; see the file LICENSE.txt. If not, the license is also       *
 * available from http://alps.comp-phys.org/.                                      *
 *                                                                                 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        *
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT       *
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE       *
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,     *
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER     *
 * DEALINGS IN THE SOFTWARE.                                                       *
 *                                                                                 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <alps/ngs/params.hpp>

#include <alps/ngs/lib/params_impl_map.ipp>
#include <alps/ngs/lib/params_impl_ordred.ipp>

#ifdef ALPS_HAVE_PYTHON
    #include <alps/ngs/lib/params_impl_dict.ipp>
#endif

namespace alps {

    params::params(params const & arg)
        : impl_(arg.impl_->clone())
    {}

    params::params(hdf5::archive & arg)
        : impl_(new detail::params_impl_map(arg))
    {}

    params::params(std::string const & arg)
        : impl_(new detail::params_impl_ordred(arg))
    {}

    #ifdef ALPS_HAVE_PYTHON
        params::params(boost::python::object const & arg)
            : impl_(new detail::params_impl_dict(arg))
        {}
    #endif

    params::~params() {}
    
    std::size_t params::size() const {
        return impl_->size();
    }

    std::vector<std::string> params::keys() const {
        return impl_->keys();
    }

    param params::operator[](std::string const & key) {
        return (*impl_)[key];
    }

    param const params::operator[](std::string const & key) const {
        return (*impl_)[key];
    }

    bool params::defined(std::string const & key) const {
        return impl_->defined(key);
    }

    void params::save(hdf5::archive & ar) const {
        return impl_->save(ar);
    }

    void params::load(hdf5::archive & ar) {
        return impl_->load(ar);
    }

    #ifdef ALPS_HAVE_PYTHON

        detail::params_impl_base * params::get_impl() {
            return impl_.get();
        }

        detail::params_impl_base const * params::get_impl() const {
            return impl_.get();
        }

    #endif

}
