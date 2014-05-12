/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* Copyright (C) 2011-2012 by Lukas Gamper <gamperl@gmail.com>,
*                            Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Maximilian Poprawe <poprawem@ethz.ch>
*
* This software is part of the ALPS libraries, published under the ALPS
* Library License; you can use, redistribute it and/or modify it under
* the terms of the license, either version 1 or (at your option) any later
* version.
*
* You should have received a copy of the ALPS Library License along with
* the ALPS Libraries; see the file LICENSE.txt. If not, the license is also
* available from http://alps.comp-phys.org/.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
* SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
* FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/

#ifndef ALPS_NUMERIC_REGRESSION_HPP
#define ALPS_NUMERIC_REGRESSION_HPP

#include <alps/numeric/vector_functions.hpp>

//#include <boost/numeric/ublas/matrix.hpp>
//#include <boost/numeric/ublas/io.hpp>
//#include <boost/numeric/bindings/traits/ublas_matrix.hpp>
//#include <boost/numeric/bindings/traits/ublas_symmetric.hpp>
//#include <boost/numeric/bindings/traits/ublas_hermitian.hpp>
//#include <boost/numeric/bindings/traits/ublas_vector2.hpp>
//#include <boost/numeric/bindings/lapack/sysv.hpp>
//#include <boost/numeric/bindings/lapack/hesv.hpp>
#include <complex>
#include <exception>
#include <utility>

namespace alps {
namespace numeric {


template <class Iterator>
std::pair<double,double> linear_timeseries_fit (Iterator y_begin, Iterator y_end) {

  // LHS = trans(X) * X (symmetric)
  // rhs = trans(X) * y

  size_t count = 1;
  
  double LHS11(0.);
  double LHS12(0.);
  double LHS22(0.);

  double rhs1(0.);
  double rhs2(0.);

  double beta1(0.);
  double beta2(0.);

  double det(0.);

  while (y_begin != y_end) {

    LHS12 += count;
    LHS22 += count * count;

    rhs1 += *y_begin;
    rhs2 += (*y_begin) * count;
//    std::cout << "\n rhs1  " << rhs1 << "\n rhs2  " << rhs2 << "\n iter" << *y_begin; // DEBUG

    ++y_begin; ++count;
  }

  LHS11 = count;

  if ( (det = (LHS11 * LHS22 - LHS12 * LHS12) ) == 0.) boost::throw_exception(std::runtime_error("Regression Error: System is not determined!"));

  beta1 = (LHS22 * rhs1 - LHS12 * rhs2) / det;
  beta2 = (LHS11 * rhs2 - LHS12 * rhs1) / det;

//  std::cout << " LHS22  " << LHS22 << " rhs2  " << rhs2; // DEBUG

  return std::make_pair(beta1, beta2);

}

template <class Iterator>
std::pair<double,double> exponential_timeseries_fit (Iterator y_begin, Iterator y_end) {
  std::vector<double> log_values;

  for(Iterator iter = y_begin; iter != y_end; ++iter) {
    if (*iter <= 0.) {std::cout << "Warning: cannot fit negative values!\n"; break;}
    log_values.push_back(std::log(*iter));
  }

  std::pair<double, double> OUT = linear_timeseries_fit(log_values.begin(), log_values.end());
  OUT.first = std::exp(OUT.first);

  return OUT;
}


// below are several regression methods that are more general/do weighted regressions
// were not needed yet

/*

class bad_regression: public std::exception {
  virtual const char* what() const throw() {
    return "Regression error";
  }
} reg_exception;


template <class T> struct solver_helper 
{
  template <class M, class V>
  static void solve(M& m, V& v) { 
    boost::numeric::bindings::lapack::sysv('U',m,v);
  }
};

template <class T> struct solver_helper<std::complex<T> >
{
  template <class M, class V>
  static void solve(M& m, V& v) { 
    boost::numeric::bindings::lapack::hesv('U',m,v);
  }
};

template <class MATRIX, class VECTOR>
struct Solver
{
  typedef VECTOR vector_type;
  typedef typename vector_type::value_type scalar_type;
  typedef MATRIX matrix_type;
    
  void operator() (const matrix_type& mat, const vector_type& x, vector_type& y) const {
    y = x;
    matrix_type mat_ = mat;
    solver_helper<scalar_type>::solve(mat_,y);
  }
};


typedef boost::numeric::ublas::matrix<double> matrix_type; 
typedef boost::numeric::ublas::vector<double> vector_type;

vector_type weighted_least_squares (const matrix_type& X, const vector_type& y, const matrix_type& W) { // W is the sqrt of the weights, in most cases 1/sqrt(var) = 1/sd
  matrix_type weighted_X = prod(W, X);
  vector_type weighted_y = prod(W, y);
  matrix_type weighted_X_trans = trans(weighted_X);
  matrix_type LHS = prod(weighted_X_trans, weighted_X);
  vector_type rhs = prod(weighted_X_trans, weighted_y);

  vector_type beta;
  Solver<matrix_type, vector_type> lapack_solver;
  lapack_solver(LHS, rhs, beta);
  return beta;
}

vector_type weighted_linear_fit_1D (const vector_type& x, const vector_type& y, const vector_type& error) {
  matrix_type X(x.size(),2);
  for (size_t i = 0; i < x.size(); ++i) {
    X(i, 0) = 1.;
    X(i, 1) = x(i);
  }
  matrix_type W = boost::numeric::ublas::zero_matrix<double>(error.size(), error.size());
  for (size_t i = 0; i < error.size(); ++i) {
    if (error(i) == 0.) throw reg_exception;
    W(i, i) = 1. / error(i);
  }

  return weighted_least_squares (X, y, W);
}

vector_type weighted_linear_fit_1D (const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& error) {
  matrix_type X(x.size(),2);
  vector_type y_(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    X(i, 0) = 1;
    X(i, 1) = x[i];
    y_(i) = y[i];
  }
  matrix_type W = boost::numeric::ublas::zero_matrix<double>(error.size(), error.size());
  for (size_t i = 0; i < error.size(); ++i) {
    if (error[i] == 0.) throw reg_exception;
    W(i, i) = 1. / error[i];
  }

  return weighted_least_squares (X, y_, W);
}


vector_type weighted_linear_fit_1D_noconst (const vector_type& x, const vector_type& y, const vector_type& error) {
  matrix_type X(x.size(),1);
  for (size_t i = 0; i < x.size(); ++i)
    X(i, 0) = x(i);  
  matrix_type W = boost::numeric::ublas::zero_matrix<double>(error.size(), error.size());
  for (size_t i = 0; i < error.size(); ++i) {
    if (error(i) == 0.) throw reg_exception;
    W(i, i) = 1. / error(i);
  }
  return weighted_least_squares (X, y, W);
}


template <class YIterator, class ErrorIterator>
std::pair<double,double> weighted_linear_fit_1D (YIterator y_begin, YIterator y_end, ErrorIterator error_begin) {

  // trans(X) * W * X = LHS (symmetric)
  // trans(X) * W * y = rhs

  size_t count = 1;

  double weight;
  
  double LHS11(0.);
  double LHS12(0.);
  double LHS22(0.);

  double rhs1(0.);
  double rhs2(0.);

  double beta1(0.);
  double beta2(0.);

  double det(0.);
  double tmp(0.);

  while (y_begin != y_end) {
    std::cout << count << " "; //DEBUG

    if (*error_begin == 0.) throw reg_exception;
    weight = 1. / ( *error_begin * *error_begin );

    LHS11 += weight;
    tmp = count * weight;
    LHS12 += tmp;
    LHS22 += count * tmp;
    
    tmp = (*y_begin) * weight;
    rhs1 += tmp;
    rhs2 += tmp * count;

    ++y_begin; ++error_begin; ++count;
  }

  if ( (det = (LHS11 * LHS22 - LHS12 * LHS12) ) == 0.) throw reg_exception;

  beta1 = (LHS22 * rhs1 - LHS12 * rhs2) / det;
  beta2 = (LHS11 * rhs2 - LHS12 * rhs1) / det;

  return std::make_pair(beta1, beta2);

}


template <class TimeseriesType>
std::pair<double,double> weighted_linear_fit_1D (const TimeseriesType& timeseries) {
  return weighted_linear_fit_1D (timeseries.begin(), timeseries.end(), );
}


template <class TimeseriesType>
std::pair<double,double> weighted_exponential_fit_1D (const TimeseriesType& timeseries) {

  typename const_iterator_type<TimeseriesType>::type iter = timeseries.begin();

  std::vector<double> log_values;

  while (iter != timeseries.end()) {
    
    ++iter;
  }

}
*/



} // ending namespace numeric
} // ending namespace alps

#endif
