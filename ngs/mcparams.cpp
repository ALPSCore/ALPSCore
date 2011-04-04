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

#include <alps/ngs/macros.hpp>
#include <alps/ngs/mchdf5.hpp>
#include <alps/ngs/mcparams.hpp>

namespace alps {

    namespace detail {
        struct mcparamvalue_saver: public boost::static_visitor<> {

            mcparamvalue_saver(hdf5::archive & a, std::string const & p)
                : ar(a), path(p) 
            {}

            template<typename T> void operator()(T & v) const {
                ar << make_pvp(path, v);
            }

            hdf5::archive & ar;
            std::string const & path;
        };
    }

    #define ALPS_NGS_MCPARAMS_CAST_OPERATOR_IMPL(T)                                                                                                        \
        mcparamvalue::operator T () const {                                                                                                                \
            detail::mcparamvalue_reader< T > visitor;                                                                                                      \
            boost::apply_visitor(visitor, *this);                                                                                                          \
            return visitor.value;                                                                                                                          \
        }
    ALPS_NGS_MCPARAMS_CAST_OPERATOR_IMPL(short)
    ALPS_NGS_MCPARAMS_CAST_OPERATOR_IMPL(unsigned short)
    ALPS_NGS_MCPARAMS_CAST_OPERATOR_IMPL(int)
    ALPS_NGS_MCPARAMS_CAST_OPERATOR_IMPL(unsigned int)
    ALPS_NGS_MCPARAMS_CAST_OPERATOR_IMPL(long)
    ALPS_NGS_MCPARAMS_CAST_OPERATOR_IMPL(unsigned long)
    ALPS_NGS_MCPARAMS_CAST_OPERATOR_IMPL(long long)
    ALPS_NGS_MCPARAMS_CAST_OPERATOR_IMPL(unsigned long long)
    ALPS_NGS_MCPARAMS_CAST_OPERATOR_IMPL(float)
    ALPS_NGS_MCPARAMS_CAST_OPERATOR_IMPL(double)
    ALPS_NGS_MCPARAMS_CAST_OPERATOR_IMPL(long double)
    ALPS_NGS_MCPARAMS_CAST_OPERATOR_IMPL(bool)
    ALPS_NGS_MCPARAMS_CAST_OPERATOR_IMPL(std::string)
    #undef ALPS_NGS_MCPARAMS_CAST_OPERATOR_IMPL

    mcparams::mcparams(std::string const & input_file) {
        hdf5::archive ar(input_file);
        ar >> make_pvp("/parameters", *this);
    }

    mcparamvalue & mcparams::operator[](std::string const & k) {
        return std::map<std::string, mcparamvalue>::operator[](k);
    }

    mcparamvalue const & mcparams::operator[](std::string const & k) const {
        if (find(k) == end())
            ALPS_NGS_THROW_INVALID_ARGUMENT("unknown argument: "  + k);
        return find(k)->second;
    }

    mcparamvalue mcparams::value_or_default(std::string const & k, mcparamvalue const & v) const {
        if (find(k) == end())
            return mcparamvalue(v);
        return find(k)->second;
    }

    bool mcparams::defined(std::string const & k) const {
        return find(k) != end();
    }

    void mcparams::save(hdf5::archive & ar) const {
        for (const_iterator it = begin(); it != end(); ++it)
            boost::apply_visitor(detail::mcparamvalue_saver(ar, it->first), it->second);
    }

    void mcparams::load(hdf5::archive & ar) {
        std::vector<std::string> list = ar.list_children(ar.get_context());
        for (std::vector<std::string>::const_iterator it = list.begin(); it != list.end(); ++it) {
            std::string v;
            ar >> make_pvp(*it, v);
            insert(std::make_pair(*it, v));
        }
    }

}
