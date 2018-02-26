/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPSCORE_PIEACEWISE_POLYNOMIAL_HPP
#define ALPSCORE_PIEACEWISE_POLYNOMIAL_HPP

#include <complex>
#include <cmath>
#include <type_traits>
#include <vector>
#include <cassert>
#include <boost/multi_array.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/typeof/typeof.hpp>

#include <alps/hdf5/archive.hpp>
#include <alps/hdf5/complex.hpp>
#include <alps/hdf5/vector.hpp>
#include <alps/hdf5/multi_array.hpp>

#ifdef ALPS_HAVE_MPI
#include "mpi_bcast.hpp"
#endif

namespace alps {
    namespace gf {

        /// Class representing a piecewise polynomial and utilities
        template<typename>
        class piecewise_polynomial;

        namespace detail {
            /**
             *
             * @tparam T double or std::complex<double>
             * @param a  scalar
             * @param b  scalar
             * @return   conj(a) * b
             */
            template<class T>
            typename std::enable_if<boost::is_floating_point<T>::value, T>::type
            outer_product(T a, T b) {
                return a * b;
            }

            template<class T>
            std::complex<T>
            outer_product(const std::complex<T> &a, const std::complex<T> &b) {
                return std::conj(a) * b;
            }

            template<class T>
            typename std::enable_if<boost::is_floating_point<T>::value, T>::type
            conjg(T a) {
                return a;
            }

            template<class T>
            std::complex<T>
            conjg(const std::complex<T> &a) {
                return std::conj(a);
            }

            template<typename T, typename Op>
            struct pp_element_wise_op {
                void perform(const T *p1, const T *p2, T *p_r, int k1, int k2, int k_r) const {
                    Op op;
                    int k = std::min(k1, k2);
                    assert(k_r >= k);
                    for (int i=0; i<k_r+1; ++i) {
                        p_r[i] = 0.0;
                    }
                    for (int i=0; i<k+1; ++i) {
                        p_r[i] = op(p1[i], p2[i]);
                    }
                }
            };

            template<typename T>
            struct pp_plus : public pp_element_wise_op<T, std::plus<T> > {};

            template<typename T>
            struct pp_minus : public pp_element_wise_op<T, std::minus<T> > {};

///  element-Wise operations on piecewise_polynomial coefficients
            template<typename T, typename Op>
            piecewise_polynomial<T>
            do_op(const piecewise_polynomial<T> &f1, const piecewise_polynomial<T> &f2, const Op& op) {
                if (f1.section_edges_ != f2.section_edges_) {
                    throw std::runtime_error("Cannot add two numerical functions with different sections!");
                }

                const int k_new = std::max(f1.order(), f2.order());
                piecewise_polynomial<T> result(k_new, f1.section_edges());

                const int k_min = std::min(f1.order(), f2.order());

                for (int s=0; s < f1.num_sections(); ++s) {
                    op.perform(&f1.coeff_[s][0], &f2.coeff_[s][0], &result.coefficient(s,0), f1.order(), f2.order(), k_min);
                }

                return result;
            }
        }

/**
 * Class for representing a piecewise polynomial
 *   A function is represented by a polynomial in each section [x_n, x_{n+1}).
 */
        template<typename T>
        class piecewise_polynomial {
        private:
            int k_;

            typedef boost::multi_array<T, 2> coefficient_type;

            template<typename TT, typename Op>
            friend piecewise_polynomial<TT>
            detail::do_op(const piecewise_polynomial<TT> &f1, const piecewise_polynomial<TT> &f2, const Op& op);

            template<typename TT>
            friend piecewise_polynomial<TT>
            operator+(const piecewise_polynomial<TT> &f1, const piecewise_polynomial<TT> &f2);

            template<typename TT>
            friend piecewise_polynomial<TT>
            operator-(const piecewise_polynomial<TT> &f1, const piecewise_polynomial<TT> &f2);

            template<typename TT>
            friend const piecewise_polynomial<TT> operator*(TT scalar, const piecewise_polynomial<TT> &pp);

            template<typename TT>
            friend
            class piecewise_polynomial;

            /// number of sections
            int n_sections_;

            /// edges of sections. The first and last elements should be -1 and 1, respectively.
            std::vector<double> section_edges_;

            /// expansion coefficients [s,l]
            /// The polynomial is represented as
            ///   \sum_{l=0}^k a_{s,l} (x - x_s)^l,
            /// where x_s is the left end point of the s-th section.
            coefficient_type coeff_;

            bool valid_;

            void check_range(double x) const {
                if (x < section_edges_[0] || x > section_edges_[section_edges_.size() - 1]) {
                    throw std::runtime_error("Give x is out of the range.");
                }
            }

            void check_validity() const {
                if (!valid_) {
                    throw std::runtime_error("pieacewise_polynomial object is not properly constructed!");
                }
            }

            void set_validity() {
                valid_ = true;
                valid_ = valid_ && (n_sections_ >= 1);
                assert(valid_);
                valid_ = valid_ && (section_edges_.size() == n_sections_ + 1);
                assert(valid_);
                valid_ = valid_ && (coeff_.shape()[0] == n_sections_);
                assert(valid_);
                valid_ = valid_ && (coeff_.shape()[1] == k_ + 1);
                assert(valid_);
                for (int i = 0; i < n_sections_; ++i) {
                    valid_ = valid_ && (section_edges_[i] < section_edges_[i + 1]);
                }
                assert(valid_);
            }

        public:
            piecewise_polynomial() : k_(-1), n_sections_(0), valid_(false) {};

            /// Construct an object set to zero
            piecewise_polynomial(int k, const std::vector<double> &section_edges) : k_(k),
                                                                                    n_sections_(section_edges.size()-1),
                                                                                    section_edges_(section_edges),
                                                                                    coeff_(boost::extents[n_sections_][k+1]),
                                                                                    valid_(false) {
                std::fill(coeff_.origin(), coeff_.origin()+coeff_.num_elements(), 0.0);

                set_validity();
                check_validity();//this may throw
            };

            piecewise_polynomial(int n_section,
                                 const std::vector<double> &section_edges,
                                 const boost::multi_array<T, 2> &coeff) : k_(coeff.shape()[1]-1),
                                                                          n_sections_(section_edges.size() - 1),
                                                                          section_edges_(section_edges),
                                                                          coeff_(coeff), valid_(false) {
                set_validity();
                check_validity();//this may throw
            };

            /// Copy operator
            piecewise_polynomial<T>& operator=(const piecewise_polynomial<T>& other) {
                k_ = other.k_;
                n_sections_ = other.n_sections_;
                section_edges_ = other.section_edges_;
                //Should be resized before a copy
                coeff_.resize(boost::extents[other.coeff_.shape()[0]][other.coeff_.shape()[1]]);
                coeff_ = other.coeff_;
                valid_ = other.valid_;
                return *this;
            }

            /// Order of the polynomial
            int order() const {
                return k_;
            }

            /// Number of sections
            int num_sections() const {
#ifndef NDEBUG
                check_validity();
#endif
                return n_sections_;
            }

            /// Return an end point. The index i runs from 0 (smallest) to num_sections()+1 (largest).
            inline double section_edge(int i) const {
                assert(i >= 0 && i < section_edges_.size());
#ifndef NDEBUG
                check_validity();
#endif
                return section_edges_[i];
            }

            /// Return a refence to end points
            const std::vector<double> &section_edges() const {
#ifndef NDEBUG
                check_validity();
#endif
                return section_edges_;
            }

            /// Return the coefficient of $x^p$ for the given section.
            inline const T& coefficient(int i, int p) const {
                assert(i >= 0 && i < section_edges_.size());
                assert(p >= 0 && p <= k_);
#ifndef NDEBUG
                check_validity();
#endif
                return coeff_[i][p];
            }

            /// Return a reference to the coefficient of $x^p$ for the given section.
            inline T& coefficient(int i, int p) {
                assert(i >= 0 && i < section_edges_.size());
                assert(p >= 0 && p <= k_);
#ifndef NDEBUG
                check_validity();
#endif
                return coeff_[i][p];
            }

            /// Set to zero
            void set_zero() {
                std::fill(coeff_.origin(), coeff_.origin()+coeff_.num_elements(), 0.0);
            }

            /// Compute the value at x.
            inline T compute_value(double x) const {
#ifndef NDEBUG
                check_validity();
#endif
                return compute_value(x, find_section(x));
            }

            /// Compute the value at x. x must be in the given section.
            inline T compute_value(double x, int section) const {
#ifndef NDEBUG
                check_validity();
#endif
                if (x < section_edges_[section] || (x != section_edges_.back() && x >= section_edges_[section + 1])) {
                    throw std::runtime_error("The given x is not in the given section.");
                }

                const double dx = x - section_edges_[section];
                T r = 0.0, x_pow = 1.0;
                for (int p = 0; p < k_ + 1; ++p) {
                    r += coeff_[section][p] * x_pow;
                    x_pow *= dx;
                }
                return r;
            }

            /// Find the section involving the given x
            int find_section(double x) const {
#ifndef NDEBUG
                check_validity();
#endif
                if (x == section_edges_[0]) {
                    return 0;
                } else if (x == section_edges_.back()) {
                    return coeff_.size() - 1;
                }

                std::vector<double>::const_iterator it =
                        std::upper_bound(section_edges_.begin(), section_edges_.end(), x);
                --it;
                return (&(*it) - &(section_edges_[0]));
            }

            /// Compute overlap <this | other> with complex conjugate. The two objects must have the same sections.
            template<class T2>
            T overlap(const piecewise_polynomial<T2> &other) const {
                check_validity();
                if (section_edges_ != other.section_edges_) {
                    throw std::runtime_error("Computing overlap between piecewise polynomials with different section edges are not supported");
                }
                typedef BOOST_TYPEOF(static_cast<T>(1.0)*static_cast<T2>(1.0))  Tr;

                const int k = this->order();
                const int k2 = other.order();

                Tr r = 0.0;
                std::vector<double> x_min_power(k+k2+2), dx_power(k+k2+2);

                for (int s = 0; s < n_sections_; ++s) {
                    dx_power[0] = 1.0;
                    const double dx = section_edges_[s + 1] - section_edges_[s];
                    for (int p = 1; p < dx_power.size(); ++p) {
                        dx_power[p] = dx * dx_power[p - 1];
                    }

                    for (int p = 0; p < k + 1; ++p) {
                        for (int p2 = 0; p2 < k2 + 1; ++p2) {
                            r += detail::outer_product((Tr) coeff_[s][p], (Tr) other.coeff_[s][p2])
                                 * dx_power[p + p2 + 1] / (p + p2 + 1.0);
                        }
                    }
                }
                return r;
            }

            /// Compute squared norm
            double squared_norm() const {
                return static_cast<double>(this->overlap(*this));
            }

            /// Returns whether or not two objects are numerically the same.
            bool operator==(const piecewise_polynomial<T> &other) const {
                return (n_sections_ == other.n_sections_) &&
                        (section_edges_ == other.section_edges_) &&
                                (coeff_ == other.coeff_);
            }

            /// Save to a hdf5 file
            void save(alps::hdf5::archive& ar, const std::string& path) const {
                check_validity();
                ar[path+"/k"] <<  k_;
                ar[path+"/num_sections"] << num_sections();
                ar[path+"/section_edges"] << section_edges_;
                ar[path+"/coefficients"] << coeff_;
            }

            /// Load from a hdf5 file
            void load(alps::hdf5::archive& ar, const std::string& path) {
                ar[path+"/k"] >>  k_;
                ar[path+"/num_sections"] >> n_sections_;
                ar[path+"/section_edges"] >> section_edges_;
                ar[path+"/coefficients"] >> coeff_;

                set_validity();
                check_validity();
            }

            /// Save to HDF5
            void save(alps::hdf5::archive& ar) const
            {
                save(ar, ar.get_context());
            }

            /// Load from HDF5
            void load(alps::hdf5::archive& ar)
            {
                load(ar, ar.get_context());
            }

#ifdef ALPS_HAVE_MPI
            /// Broadcast
            void broadcast(const alps::mpi::communicator& comm, int root)
            {
                using alps::mpi::broadcast;

                broadcast(comm, k_, root);
                broadcast(comm, n_sections_, root);
                section_edges_.resize(n_sections_+1);
                broadcast(comm, &section_edges_[0], n_sections_+1, root);

                coeff_.resize(boost::extents[n_sections_][k_+1]);
                broadcast(comm, coeff_.origin(), (k_+1)*n_sections_, root);

                set_validity();
                check_validity();
            }
#endif

        };//class pieacewise_polynomial

/// Add piecewise_polynomial objects
        template<typename T>
        piecewise_polynomial<T>
        operator+(const piecewise_polynomial<T> &f1, const piecewise_polynomial<T> &f2) {
            return detail::do_op(f1, f2, detail::pp_plus<T>());
        }

/// Substract piecewise_polynomial objects
        template<typename T>
        piecewise_polynomial<T>
        operator-(const piecewise_polynomial<T> &f1, const piecewise_polynomial<T> &f2) {
            return detail::do_op(f1, f2, detail::pp_minus<T>());
        }

/// Multiply piecewise_polynomial by a scalar
        template<typename T>
        const piecewise_polynomial<T> operator*(T scalar, const piecewise_polynomial<T> &pp) {
            piecewise_polynomial<T> pp_copy(pp);
            std::transform(
                    pp_copy.coeff_.origin(), pp_copy.coeff_.origin() + pp_copy.coeff_.num_elements(),
                    pp_copy.coeff_.origin(), std::bind1st(std::multiplies<T>(), scalar)

            );
            return pp_copy;
        }

/// Gram-Schmidt orthonormalization
        template<typename T>
        void orthonormalize(std::vector<piecewise_polynomial<T> > &pps) {
            typedef piecewise_polynomial<T> pp_type;

            for (int l = 0; l < pps.size(); ++l) {
                pp_type pp_new(pps[l]);
                for (int l2 = 0; l2 < l; ++l2) {
                    const T overlap = pps[l2].overlap(pps[l]);
                    pp_new = pp_new - overlap * pps[l2];
                }
                double norm = pp_new.overlap(pp_new);
                pps[l] = (1.0 / std::sqrt(norm)) * pp_new;
            }
        }
    }
}

#endif //ALPSCORE_PIEACEWISE_POLYNOMIAL_HPP
