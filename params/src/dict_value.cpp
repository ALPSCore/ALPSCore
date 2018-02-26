/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
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

        namespace detail {
            namespace visitor {

                /// Visitor to compare 2 value of dict_value type
                /** @note Values are comparable iff they are of the same type (FIXME!) */
                class comparator2 : public boost::static_visitor<int> {
                    template <typename A, typename B>
                    static bool cmp_(const A& a, const B& b) { return (a==b)? 0 : (a<b)? -1:1; }
                    
                    public:
                    /// Called by apply_visitor for bound values of different types
                    template <typename LHS_T, typename RHS_T>
                    int operator()(const LHS_T& lhs, const RHS_T& rhs) const {
                        std::string lhs_name=detail::type_info<LHS_T>::pretty_name();
                        std::string rhs_name=detail::type_info<RHS_T>::pretty_name();
                        throw exception::type_mismatch("","Attempt to compare dictionary values containing "
                                                       "incompatible types "+
                                                       lhs_name + "<=>" + rhs_name);
                    }
                    
                    /// Called by apply_visitor for bound values of the same type
                    template <typename LHS_RHS_T>
                    int operator()(const LHS_RHS_T& lhs, const LHS_RHS_T& rhs) const {
                        return cmp_(lhs,rhs);
                    }

                    /// Called by apply_visitor for bound values both having None type
                    int operator()(const dict_value::None& lhs, const dict_value::None& rhs) const {
                        return 1;
                    }
                        

                    // FIXME:TODO:
                    // Same types: compare directly
                    // Integral types: compare using signs (extract it to a separate namespace/class)
                    // FP types: compare directly
                    // Everything else: throw
                };

                /// Visitor to test for exact equality (name and value)
                class equals2 : public boost::static_visitor<bool> {
                    public:
                    /// Called when bound values have the same type
                    template <typename LHS_RHS_T>
                    bool operator()(const LHS_RHS_T& lhs, const LHS_RHS_T& rhs) const {
                        return lhs==rhs;
                    }

                    /// Called when bound types are different
                    template <typename LHS_T, typename RHS_T>
                    bool operator()(const LHS_T& lhs, const RHS_T& rhs) const{
                        return false;
                    }

                    /// Called when LHS is None
                    template <typename RHS_T>
                    bool operator()(const dict_value::None&, const RHS_T&) const {
                        return false;
                    }
                    
                    /// Called when RHS is None
                    template <typename LHS_T>
                    bool operator()(const LHS_T&, const dict_value::None&) const {
                        return false;
                    }
                    
                    /// Called when both are None
                    bool operator()(const dict_value::None&, const dict_value::None&) const {
                        return true;
                    }
                };
                
            } // visitor::
            
        } // detail::
        
        int dict_value::compare(const dict_value& rhs) const
        {
            if (this->empty() || rhs.empty()) throw exception::uninitialized_value(name_+"<=>"+rhs.name_,"Attempt to compare uninitialized value");
                
            try {
                return boost::apply_visitor(detail::visitor::comparator2(), val_, rhs.val_);
            } catch (exception::exception_base& exc) {
                exc.set_name(name_+"<=>"+rhs.name_);
                throw;
            } 
        }

        bool dict_value::equals(const dict_value& rhs) const
        {
            return boost::apply_visitor(detail::visitor::equals2(), val_, rhs.val_);
        }

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
            
            // Printing of a vector
            // FIXME!!! Consolidate with other definitions and move to alps::utilities
            template <typename T>
            inline std::ostream& operator<<(std::ostream& strm, const std::vector<T>& vec)
            {
                typedef std::vector<T> vtype;
                typedef typename vtype::const_iterator itype;

                strm << "[";
                itype it=vec.begin();
                const itype end=vec.end();

                if (end!=it) {
                    strm << *it;
                    for (++it; end!=it; ++it) {
                        strm << ", " << *it;
                    }
                }
                strm << "]";

                return strm;
            }

            struct print_visitor : public boost::static_visitor<std::ostream&> {
                std::ostream& os_;

                print_visitor(std::ostream& os) : os_(os) {}
                
                template <typename T>
                std::ostream& operator()(const T& val) const {
                    return os_ << val;
                }

                std::ostream& operator()(const dict_value::None&) const {
                    throw std::logic_error("print_visitor: This is not expected to be called");
                }
            };

        }
            
        std::ostream& print(std::ostream& s, const dict_value& dv, bool terse) {
            if (dv.empty()) {
                s << "[NONE]";
                if (!terse) s << " (type: None)";
            } else {
                // s << dv.val_;
                boost::apply_visitor(print_visitor(s), dv.val_);
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
