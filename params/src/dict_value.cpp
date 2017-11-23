/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file dict_value.cpp
    Contains implementation of some alps::params_ns::dict_value members */

#include <alps/params/dict_value.hpp>
#include <alps/params/hdf5_variant.hpp>

#ifdef ALPS_HAVE_MPI
#include <alps/params/mpi_variant.hpp>
#endif

namespace alps {
    namespace params_ns {
        
        // int dict_value::compare(const dict_value& rhs) const
        // {
        //     if (this->empty() || rhs.empty()) throw exception::uninitialized_value(name_+"<=>"+rhs.name_,"Attempt to compare uninitialized value");
                
        //     try {
        //         return boost::apply_visitor(detail::visitor::comparator2(), val_, rhs.val_);
        //     } catch (exception::exception_base& exc) {
        //         exc.set_name(name_+"<=>"+rhs.name_);
        //         throw;
        //     } 
        // }

        void dict_value::save(alps::hdf5::archive& ar) const {
            if (this->empty()) return;
            alps::hdf5::write_variant<detail::dict_all_types>(ar, val_);
        }
            
        void dict_value::load(alps::hdf5::archive& ar) {
            const std::string context=ar.get_context();
            std::string::size_type slash_pos=context.find_last_of("/");
            if (slash_pos==std::string::npos) slash_pos=0; else ++slash_pos;
            name_=context.substr(slash_pos);
            val_=alps::hdf5::read_variant<detail::dict_all_types>(ar);
        }

        namespace {
            struct typestring_visitor : public boost::static_visitor<std::string> {
                template <typename T>
                std::string operator()(const T& val) const {
                    std::string ret=detail::type_info<T>::pretty_name();
                    return ret;
                }
            };
            
        }
            
        std::ostream& print(std::ostream& s, const dict_value& dv, bool terse) {
            if (dv.empty()) {
                s << "[NONE]";
                if (!terse) s << " (type: None)";
            } else {
                s << dv.val_;
                if (!terse) s << " (type: " << boost::apply_visitor(typestring_visitor(), dv.val_) << ")";
            }
            if (!terse) s << " (name='" << dv.name_ << "')";
            return s;
        }

#ifdef ALPS_HAVE_MPI
        void dict_value::broadcast(const alps::mpi::communicator& comm, int root)
        {
            using alps::mpi::broadcast;
            broadcast(comm, name_, root);
            broadcast<detail::dict_all_types>(comm, val_, root);
        }
#endif

        
    } // params_ns::
} // alps::
