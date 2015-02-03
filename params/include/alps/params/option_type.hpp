/** Designing a suitable type to hold parameters */

/** Requirements:

    1. Can hold `None`, a scalar type, a vector type.

    2. `None` cannot be assigned to anything.

    3. Vector type can be assigned only to the same type.

    4. Scalar type can be assigned to any compatible (by implicit cast) type.

    5. Anything can be assigned to `None`.

    6. Only the same type can be assigned to any vector or scalar type.

    7. Models of scalar type: `int`, `double`, `bool`.

    8. Models of vector type: `std::string`, `std::vector<T>`

    9. The parameter must hold its name (and possibly typename?), for error reporting purposes

*/

#ifndef ALPS_PARAMS_OPTION_TYPE_INCLUDED

#include <iostream>
#include <stdexcept>

#include "boost/variant.hpp"
#include "boost/utility.hpp" // for enable_if
#include "boost/type_traits.hpp" // for is_convertible

#include "alps/params/param_types.hpp" // Sequences of supported types

namespace alps {
    namespace params_ns {
        
        class option_type {

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
            
            /// Exception to be thrown by visitor class: None value used
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
            
            /// Visitor to assign a value of type T to a variant containing type U
            template <typename T>
            struct setter_visitor: public boost::static_visitor<>
            {
                const T& rhs; ///< The rhs value to be assigned

                /// Constructor save the value to be assigned
                setter_visitor(const T& a_rhs): rhs(a_rhs) {}

                /// Called when the bound type U is the same as T
                void apply(T& lhs) const
                {
                    lhs=rhs;
                }

                /// Called when the bound type U and rhs type T are distinct types
                template <typename U>
                void apply(U& lhs) const
                {
                    std::string msg="Attempt to assign type T=";
                    msg += typeid(T).name();
                    msg += " to the option_type object containing type U=";
                    msg += typeid(U).name();
                    throw visitor_type_mismatch(msg);
                }

                /// Called when the bound type U is None (should never happen, option_type::operator=() must take care of this)
                void apply(None& lhs) const
                {
                    throw std::logic_error("Should not happen: setting an option_type object containing None");
                }

                /// Called by apply_visitor()
                template <typename U>
                void operator()(U& lhs) const
                {
                    apply(lhs);
                }
            };
      
            /// Assignment operator: assigns a value of type T
            template <typename T>
            void operator=(const T& rhs)
            {
                if (val_.which()==0) { // NOTE:Caution -- relies on None being the first type!
                    val_=rhs;
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

            /// Visitor to get a value (with conversion): returns type T, converts from the bound type U
            template <typename T>
            struct getter_visitor: public boost::static_visitor<T> {

                /// Simplest case: the values are of the same type
                T apply(const T& val) const {
                    return val; // no conversion 
                }

    
                /// Types are convertible (Both are scalar types)
                template <typename U>
                // T apply(const U& val, typename boost::enable_if< detail::both_scalar<T,U>, bool>::type =true) const {
                T apply(const U& val, typename boost::enable_if< boost::is_convertible<U,T>, bool>::type =true) const {
                    return val; // invokes implicit conversion 
                }

                /// Types are not convertible (One of the types is not a scalar)
                template <typename U>
                // T apply(const U& val, typename boost::disable_if< detail::both_scalar<T,U>, bool>::type =true) const {
                T apply(const U& val, typename boost::disable_if< boost::is_convertible<U,T>, bool>::type =true) const {
                    throw visitor_type_mismatch("Attempt to assign non-convertible types");
                }

                /// Extracting None type --- always fails
                T apply(const None& val) const {
                    throw visitor_none_used("Attempt to use uninitialized option value");
                }

                /// Called by apply_visitor()
                template <typename U>
                T operator()(const U& val) const {
                    return apply(val);
                }
            };

            /// Conversion operator to a generic type T
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

            /// Assignment from boost::any containing type T
            template <typename T>
            void assign_any(const boost::any& aval)
            {
                val_=boost::any_cast<T>(aval);
            }

            /// Constructor preserving the option name
            option_type(const std::string& a_name):
                name_(a_name) {}

            // /// Default constructor of anonymous option (FIXME: can we avoid using it?)
            // option_type(): name_("") {}
        };

        /// Equality operator for option_type
        template <typename T>
        bool operator==(const option_type& lhs, T rhs)
        {
            const T& lhs_t=lhs;
            return (rhs == lhs_t);
        }

        /// Equality operator for option_type
        template <typename T>
        bool operator==(T lhs, const option_type& rhs)
        {
            return (rhs == lhs);
        }

        /// Class "map of options" (needed to ensure that option is always initialized by the name)
        class options_map_type : public std::map<std::string, option_type> {
        public:
            /// Access to a constant object
            const mapped_type& operator[](const key_type& k) const
            {
                const_iterator it=find(k);
                if (it == end()) {
                    throw std::runtime_error("Attempt to access non-existing key '"+k+"'");
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
        };
    } // params_ns
} // alps


#endif // ALPS_PARAMS_OPTION_TYPE_INCLUDED
