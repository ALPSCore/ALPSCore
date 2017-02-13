/*
 * Copyright (C) 1998-2016 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file option_type.hpp Defines type(s) used to hold parameters and populate `alps::params` container. */

/*  Requirements for the `option_type`:

    1. It can hold a value of a scalar type or a value of a vector type, each can be in the "undefined" state.

    2. If "undefined", it cannot be assigned to anything.

    3. If holds a value of some type, it can be assigned to the same or a "larger" type
       (as defined by `detail::is_convertible<FROM,TO>`).

    4. If holds "undefined", any value can be assigned to it, and it will acquire the new type.

    5. If holds a value of some type, only the same or a smaller typed value can be assigned to it
       (as defined by `detail::is_convertible<FROM,TO>`), and the assignment will not change the type.

    6. The parameter must hold its name (and possibly typename?), for error reporting purposes
*/

#ifndef ALPS_PARAMS_OPTION_TYPE_INCLUDED
#define ALPS_PARAMS_OPTION_TYPE_INCLUDED

#include <iostream>
#include <stdexcept>

#include <boost/variant.hpp>
#include <boost/utility.hpp> /* for enable_if */
#include <boost/preprocessor/seq.hpp>

#include "alps/utilities/short_print.hpp" // for streaming
#include "alps/hdf5/archive.hpp"          // archive support
#include "alps/hdf5/vector.hpp"           // vector archiving support
#include "alps/hdf5/map.hpp"              // map archiving support

#include "alps/params/param_types.hpp" // Sequences of supported types
#include "alps/params/param_types_ranking.hpp" // for detail::is_convertible<F,T>

namespace alps {
    namespace params_ns {
        
        class option_type {

            // friend class option_description_type;  // to interface with boost::program_options

            public: // FIXME: not everything is public

            /// "Empty value" type
            typedef detail::None None;

            variant_all_type val_; ///< Value of the option

            std::string name_; ///< The option name (FIXME: make it "functionally const")

            // std::string type_name_; ///< The option type

            /// Exception to be thrown by visitor class: type mismatch
            struct visitor_type_mismatch: public std::runtime_error {
                visitor_type_mismatch(const std::string& a_what)
                    : std::runtime_error(a_what) {}
            };
            
            /// Exception to be thrown by visitor class: empty value used
            struct visitor_none_used: public std::runtime_error {
                visitor_none_used(const std::string& a_what)
                    : std::runtime_error(a_what) {}
            };
            
            /// General exception (base class)
            class exception_base : public std::runtime_error {
                const char* const name_; ///< name of the option that caused the error
            public:
                exception_base(const std::string& a_name, const std::string& a_reason)
                    : std::runtime_error("Option '"+a_name+"': "+a_reason),
                      name_(a_name.c_str())
                {}

                std::string name() const { return name_; }
            };
            
            /// Exception for mismatching types assignment
            struct type_mismatch : public exception_base {
                type_mismatch(const std::string& a_name, const std::string& a_reason)
                    : exception_base(a_name, a_reason) {};
            };

            /// Exception for using uninitialized option value
            struct uninitialized_value : public exception_base {
                uninitialized_value (const std::string& a_name, const std::string& a_reason)
                    : exception_base(a_name, a_reason) {};
            };
            
            /// Visitor to assign a value of type RHS_T to a variant containing type optional<LHS_T>
            template <typename RHS_T>
            struct setter_visitor: public boost::static_visitor<>
            {
                const RHS_T& rhs; ///< The rhs value to be assigned

                /// Constructor saves the value to be assigned
                setter_visitor(const RHS_T& a_rhs): rhs(a_rhs) {}

                /// Called when the bound type optional<LHS_T> holds RHS_T
                void apply(boost::optional<RHS_T>& lhs) const
                {
                    lhs=rhs;
                }

                /// Called when the contained type LHS_T and rhs type RHS_T are distinct, convertible types
                template <typename LHS_T>
                void apply(boost::optional<LHS_T>& lhs,
                           typename boost::enable_if< detail::is_convertible<RHS_T,LHS_T> >::type* =0) const
                {
                    lhs=rhs;
                }

                /// Called when the contained type LHS_T and rhs type RHS_T are distinct, non-convertible types
                template <typename LHS_T>
                void apply(boost::optional<LHS_T>& lhs,
                           typename boost::disable_if< detail::is_convertible<RHS_T,LHS_T> >::type* =0) const
                {
                    throw visitor_type_mismatch(
                        std::string("Attempt to assign a value of type \"")
                        + detail::type_id<RHS_T>().pretty_name()
                        + "\" to the option_type object containing type \""
                        + detail::type_id<LHS_T>().pretty_name()+"\"");
                }

                // /// Called when the bound type RHS_T is None (should never happen, option_type::operator=() must take care of this)
                // void apply(None& lhs) const
                // {
                //     throw std::logic_error("Should not happen: setting an option_type object containing None");
                // }

                /// Called when the bound type in the variant is None (should never happen, option_type::operator=() must take care of this)
                void operator()(const None& ) const
                {
                    throw std::logic_error("Should not happen: setting an option_type object containing None");
                }

                /// Called by apply_visitor()
                template <typename LHS_T>
                void operator()(boost::optional<LHS_T>& lhs) const
                {
                    apply(lhs);
                }
            };

            /// Checks if the type it contains is None
            bool isNone() const
            {
                return val_.which()==0;  // NOTE:Caution -- relies on None being the first type!
            }

            /// Visitor to check if the contained value is empty (unassigned)
            struct isempty_visitor: public boost::static_visitor<bool>
            {
                /// Called by apply_visitor()
                template <typename T>
                bool operator()(const boost::optional<T>& val) const
                {
                    return !val;
                }
                /// Called by apply_visitor()
                bool operator()(const None&) const
                {
                    return true;
                }
            };
            
            /// Checks if the option is empty (does not have an assigned value)
            bool isEmpty() const
            {
                return boost::apply_visitor(isempty_visitor(),val_);
            }
          
            /// Assignment operator: assigns a value of type T
            template <typename T>
            void operator=(const T& rhs)
            {
                if (isNone()) { 
                    val_=boost::optional<T>(rhs);
                    return;
                }

                try {
                    boost::apply_visitor(setter_visitor<T>(rhs), val_);
                } catch (visitor_type_mismatch& exc) {
                    throw type_mismatch(name_, exc.what());
                }
            }

            /// Assignment operator specialization: assigns a value of type `char*`
            void operator=(const char* rhs)
            {
                *this=std::string(rhs);
            }

            // /// Assignment operator specialization: assigns a value of type `unsigned int`
            // void operator=(unsigned int rhs)
            // {
            //     *this=int(rhs); // FIXME?? Is it good --- to abandon signedness?
            // }

            //  /// Assignment operator specialization: assigns a value of type `unsigned long`
            // void operator=(unsigned long rhs)
            // {
            //     *this=long(rhs); // FIXME?? Is it good --- to abandon signedness?
            // }

            /// Set the contained value to "empty" of the given type
            template <typename T>
            void reset()
            {
                val_=boost::optional<T>();
            }

            /// Set the constained value to the given value and type
            template <typename T>
            void reset(const T& v)
            {
                val_=boost::optional<T>(v);
            }

            /// Visitor to get a value (with conversion): returns type LHS_T, converts from the bound type optional<RHS_T>
            template <typename LHS_T>
            struct getter_visitor: public boost::static_visitor<LHS_T> {

                /// Simplest case: the values are of the same type
                LHS_T apply(const LHS_T& val) const {
                    return val; // no conversion 
                }
    
                /// Types are convertible (Both are scalar types)
                template <typename RHS_T>
                LHS_T apply(const RHS_T& val,
                            typename boost::enable_if< detail::is_convertible<RHS_T,LHS_T> >::type* =0) const {
                    return val; // invokes implicit conversion 
                }

                /// Types are not convertible 
                template <typename RHS_T>
                LHS_T apply(const RHS_T& val,
                            typename boost::disable_if< detail::is_convertible<RHS_T,LHS_T> >::type* =0) const {
                    throw visitor_type_mismatch(
                        std::string("Attempt to assign an option_type object containing a value of type \"")
                        + detail::type_id<RHS_T>().pretty_name()
                        + "\" to a value of an incompatible type \""
                        + detail::type_id<LHS_T>().pretty_name()+"\"");
                }

                /// Extracting None type --- always fails
                LHS_T operator ()(const None&) const {
                    throw visitor_none_used("Attempt to use uninitialized option value");
                }

                /// Called by apply_visitor()
                template <typename RHS_T>
                LHS_T operator()(const boost::optional<RHS_T>& val) const {
                    if (!val) throw visitor_none_used("Attempt to use uninitialized option value");
                    return apply(*val);
                }
            };

            /// Conversion operator to a generic type T (invoked by implicit conversion)
            template <typename T>
            operator T() const
            {
                try {
                    return boost::apply_visitor(getter_visitor<T>(), val_);
                } catch (visitor_type_mismatch& exc) {
                    throw type_mismatch(name_,exc.what());
                } catch (visitor_none_used& exc) {
                    throw uninitialized_value(name_, exc.what());
                }
            }

            /// Explicit conversion to a generic type T
            template <typename T>
            T as() const
            {
                return *this;
            }


            /// Visitor to check if the bound type U is convertible to type T
            template <typename T>
            struct typecheck_visitor : public boost::static_visitor<bool> {
              public: // FIXME: not everything has to be public!
                /// Called by apply_visitor() for a bound type U
                template <typename U>
                bool operator()(const boost::optional<U>& val) const
                {
                    if (!val) return false; // empty value is not convertible (FIXME?)
                    return apply(*val);
                }

                /// Called by apply_visitor() for a bound type None
                bool operator()(const None& val) const
                {
                    throw std::logic_error("Checking convertibility of type None --- should not be needed?"); // FIXME???
                }

                /// The bound type U is the same as requested type T:
                bool apply(const T&) { return true; }

                /// The bound type U {is / is not} convertible to T:
                template <typename U>
                bool apply(const U&) const
                {
                    return detail::is_convertible<U,T>::value;
                }
            };

            /// Check if the bound type is convertible to the type T
            template <typename T>
            bool is_convertible() const
            {
                typecheck_visitor<T> visitor;
                return boost::apply_visitor(visitor, this->val_);
            }
                    
            
            /// Visitor to call alps::utilities::short_print on the type, hidden in boost::variant
            struct ostream_visitor : public boost::static_visitor<> {
            public:
                ostream_visitor(std::ostream & arg) : os(arg) {}

                void operator()(const None&) const {
                        os << "[undefined]"; // FIXME: should it ever appear?
                }
                    
                template <typename U>
                void operator()(const boost::optional<U>& v) const {
                    if (!v) {
                        os << "[empty]"; // FIXME: should it ever appear?
                    } else {
                        os << short_print(*v);
                    }
                }
            private:
                std::ostream & os;
            };

            /// Output an option
            friend std::ostream& operator<< (std::ostream& out, option_type const& x) 
            {
                ostream_visitor visitor(out);
                boost::apply_visitor(visitor, x.val_);
                return out;
            } 
                
            /// Visitor to archive an option with a proper type
            struct save_visitor : public boost::static_visitor<> {
                hdf5::archive& ar_;
                const std::string& name_;

                save_visitor(hdf5::archive& ar, const std::string& name) : ar_(ar), name_(name) {}

                /// sends value of the bound type U to an archive
                template <typename U>
                void operator()(const U& val) const
                {
                    if (!!val) ar_[name_] << *val;
                }

                /// specialization for U==None: skips the value
                void operator()(const None&) const
                { }

                /// specialization for trigger_tag: throws (FIXME???)
                void operator()(const boost::optional<detail::trigger_tag>&) const
                {
                    throw std::logic_error("Attempt to archive trigger_tag type --- should not be needed??");
                }
            };

            /// Class for reading the option from archive
            // FIXME: simplified version of detail::option_description_type::reader,
            //        can it be made more general? and used there?
            class reader {
                alps::hdf5::archive& ar_;
                const std::string& name_;
                public:
                reader(alps::hdf5::archive& ar, const std::string& name)
                    : ar_(ar), name_(name)
                { }

                template <typename T>
                bool can_read(const T* dummy)
                {
                    bool ok=ar_.is_datatype<T>(name_);
                    ok &= ar_.is_scalar(name_);
                    std::cout << "DEBUG: can_read<T>=" << typeid(*dummy).name()
                              << " name=" << name_
                              << " ok=" << ok
                              << std::endl;
                    return ok;
                }

                template <typename T>
                bool can_read(const std::vector<T>* dummy)
                {
                    bool ok=ar_.is_datatype<T>(name_);
                    ok &= !ar_.is_scalar(name_);
                    std::cout << "DEBUG: can_read<T>=" << typeid(*dummy).name()
                              << " name=" << name_
                              << " ok=" << ok
                              << std::endl;
                    return ok;
                }

                template <typename T>
                option_type read(const T* dummy)
                {
                    std::cout << "DEBUG: read<T>=" << typeid(*dummy).name()
                              << " name=" << name_
                              << std::endl;
                    T val;
                    ar_[name_] >> val;
                    option_type opt(name_);
                    opt.reset(val);
                    return opt;
                }

            };


            /// Outputs the option to an archive
            void save(hdf5::archive& ar) const
            {
                save_visitor visitor(ar,this->name_);
                boost::apply_visitor(visitor, this->val_);
            }

            /// Constructs the option from archive (factory method)
            static option_type get_loaded(hdf5::archive& ar, const std::string& name)
            {
                reader rd(ar,name);
                    
                // macro: try reading, return if ok
#define ALPS_LOCAL_TRY_LOAD(_r_,_d_,_type_)                             \
                if (rd.can_read((_type_*)0)) return rd.read((_type_*)0);

                // try reading for each defined type
                BOOST_PP_SEQ_FOR_EACH(ALPS_LOCAL_TRY_LOAD, X, ALPS_PARAMS_DETAIL_VTYPES_SEQ ALPS_PARAMS_DETAIL_STYPES_SEQ);
#undef ALPS_LOCAL_TRY_LOAD
                    
                throw std::runtime_error("No matching payload type in the archive "
                                         "for `option_type` for "
                                         "name='" + name + "'");
            }
                
            // /// Outputs the option to an archive
            // void load(hdf5::archive& ar)
            // {
            //     throw std::logic_error("alps::params::option_type::load() is not implemented yet");
            // }

          
            /// Constructor preserving the option name
            option_type(const std::string& a_name):
                name_(a_name) {}

            /// A fake constructor to create uninitialized object for serialization --- DO NOT USE IT!!!
            // FIXME: can i avoid it?
            option_type()
                : name_("**UNINITIALIZED**") {}

        };

        /// Equality operator for option_type
        template <typename T> inline bool operator==(const option_type& lhs, const T& rhs) { return (lhs.as<T>() == rhs); }
        /// Equality operator for option_type and a char-string
        inline bool operator==(const option_type& lhs, const char* rhs) { return (lhs.as<std::string>() == rhs); }

        /// Less-then operator for option_type
        template <typename T> inline bool operator<(const option_type& lhs, const T& rhs) { return (lhs.as<T>() < rhs); }
        /// Less-then operator for option_type and a char-string
        inline bool operator<(const option_type& lhs, const char* rhs) { return (lhs.as<std::string>() < rhs); }

        /// Less-then operator for option_type
        template <typename T> inline bool operator<(const T& lhs, const option_type& rhs) { return (lhs < rhs.as<T>()); }
        /// Less-then operator for option_type and a char-string
        inline bool operator<(const char* lhs, const option_type& rhs) { return (lhs < rhs.as<std::string>()); }

        /// Greater-then operator for option_type
        template <typename T> inline bool operator>(const T& lhs, const option_type& rhs) { return (lhs > rhs.as<T>()); }
        /// Greater-then operator for option_type and a char-string
        inline bool operator>(const char* lhs, const option_type& rhs) { return (lhs > rhs.as<std::string>()); }

        /// Greater-then operator for option_type
        template <typename T> inline bool operator>(const option_type& lhs, const T& rhs) { return (lhs.as<T>() > rhs); }
        /// Greater-then operator for option_type and a char-string
        inline bool operator>(const option_type& lhs, const char* rhs) { return (lhs.as<std::string>() > rhs); }
            
        /// Equality operator for option_type
        template <typename T> inline bool operator==(const T& lhs, const option_type& rhs) { return (rhs == lhs); }

        /// Greater-equal operator for option_type
        template <typename T> inline bool operator>=(const option_type& lhs, const T& rhs) { return !(lhs < rhs); }
        
        /// Greater-equal operator for option_type
        template <typename T> inline bool operator>=(const T& lhs, const option_type& rhs) { return !(lhs < rhs); }
        
        /// Less-equal operator for option_type
        template <typename T> inline bool operator<=(const option_type& lhs, const T& rhs) { return !(lhs > rhs); }
        
        /// Less-equal operator for option_type
        template <typename T> inline bool operator<=(const T& lhs, const option_type& rhs) { return !(lhs > rhs); }
        
        /// Not-equal operator for option_type
        template <typename T> inline bool operator!=(const T& lhs, const option_type& rhs) { return !(lhs == rhs); }

        /// Not-equal operator for option_type
        template <typename T> inline bool operator!=(const option_type& lhs, const T& rhs) { return !(lhs == rhs); }

        
        /// Class "map of options" (needed to ensure that option is always initialized by the name)
        class options_map_type : public std::map<std::string, option_type> {
        public:
            /// Access to a constant object
            const mapped_type& operator[](const key_type& k) const
            {
                const_iterator it=find(k);
                if (it == end() || it->second.isEmpty() || it->second.isNone() ) {
                    throw option_type::uninitialized_value(k, "Attempt to access non-existing key '"+k+"'");
                }
                return it->second;
            }

            /// Access to the map with intention to assign an element
            mapped_type& operator[](const key_type& k)
            {
                iterator it=find(k);
                if (it==end()) {
                    // it's a new element, we have to construct it here
                    value_type newpair(k, option_type(k));
                    // ...and copy it to the map, returning the ref to the inserted element
                    it=insert(end(),newpair);
                }
                // return reference to the existing or the newly-created element
                return it->second;
            }

            // void save(hdf5::archive& ar) const
            // {
            //     // throw std::logic_error("options_map_type::save() not implemented yet");
            //     typedef std::map<std::string, option_type> super_type;
            //     ar["alps::params::options_map_type"] << static_cast<const super_type&>(*this);
            // }

            // void load(hdf5::archive& ar)
            // {
            //     // throw std::logic_error("options_map_type::load() not implemented yet");
            //     typedef std::map<std::string, option_type> super_type;
            //     ar["alps::params::options_map_type"] >> static_cast<super_type&>(*this);
            // }

        };

        namespace detail {

            /// Checks if an option is "missing"
            inline bool is_option_missing(const option_type& opt) {
                return opt.isEmpty() || opt.isNone();
            }

            /// Tag type to indicate vector/list parameter (FIXME: make sure it works for output too)
            template <typename T>
            struct vector_tag {};

            /// Type to indicate string parameter and hold default value (for our own string validator)
            class string_container {
                std::string contains_;
              public:
                string_container(const std::string& s): contains_(s) {}
                operator std::string() const { return contains_; } 
            };
            

            /// Service class calling boost::program_options::add_options(), to work around lack of function template specializations
            /// T is the option type, U is the tag type used to treat parsing of strings and vectors/lists specially
            template <typename T, typename U=T>
            struct do_define {
                /// Add option with a default value
                static void add_option(boost::program_options::options_description& a_opt_descr,
                                       const std::string& optname, T defval, const std::string& a_descr)
                {
                    a_opt_descr.add_options()(optname.c_str(),
                                              boost::program_options::value<U>()->default_value(defval),
                                              a_descr.c_str());
                }

                /// Add option with a default value with known string representation
                static void add_option(boost::program_options::options_description& a_opt_descr,
                                       const std::string& optname, T defval, const std::string& defval_str,
                                       const std::string& a_descr)
                {
                    a_opt_descr.add_options()(optname.c_str(),
                                              boost::program_options::value<U>()->default_value(defval,defval_str),
                                              a_descr.c_str());
                }

                /// Add option with no default value
                static void add_option(boost::program_options::options_description& a_opt_descr,
                                       const std::string& optname, const std::string& a_descr)
                {
                    a_opt_descr.add_options()(optname.c_str(),
                                              boost::program_options::value<U>(),
                                              a_descr.c_str());
                }
            };

            /// Specialization of the service do_define class to define a vector/list option 
            template <typename T>
            struct do_define< std::vector<T> > {
                /// Add option with no default value
                static void add_option(boost::program_options::options_description& a_opt_descr,
                                       const std::string& optname, const std::string& a_descr)
                {
                    do_define< std::vector<T>, vector_tag<T> >::add_option(a_opt_descr, optname, a_descr);
                }

                /// Add option with default value: should never be called (a plug to keep boost::variant happy)
                static void add_option(boost::program_options::options_description& a_opt_descr,
                                       const std::string& optname, const std::vector<T>& defval, const std::string& a_descr)
                {
                    throw std::logic_error("Should not happen: setting default value for vector/list parameter");
                }
            };
          
            /// Specialization of the service do_define class to define a "trigger" (parameterless) option
            template <>
            struct do_define<trigger_tag> {
                /// Add option with no default value
                static void add_option(boost::program_options::options_description& a_opt_descr,
                                       const std::string& optname, const std::string& a_descr)
                {
                    a_opt_descr.add_options()(optname.c_str(),
                                              a_descr.c_str());
                }
            };

            /// Specialization of the service do_define class for std::string parameter
            template <>
            struct do_define<std::string> {
                /// Add option with a default value
                static void add_option(boost::program_options::options_description& a_opt_descr,
                                       const std::string& optname, const std::string& defval, const std::string& a_descr)
                {
                    do_define<std::string, string_container>::add_option(a_opt_descr, optname, defval, defval, a_descr); // defval is passed as both default val and its string representation
                }

                /// Add option with no default value
                static void add_option(boost::program_options::options_description& a_opt_descr,
                                       const std::string& optname, const std::string& a_descr)
                {
                    do_define<std::string, string_container>::add_option(a_opt_descr, optname, a_descr);
                }
            };

            

        } // detail

            
        
    } // params_ns
} // alps

#endif // ALPS_PARAMS_OPTION_TYPE_INCLUDED
