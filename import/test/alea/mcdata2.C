/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2010 by Lukas Gamper <gamperl@gmail.com>,
*                            Ping Nang Ma <pingnang@itp.phys.ethz.ch>,
*                            Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

/* $Id: nobinning.h 3520 2009-12-11 16:49:53Z gamperl $ */

#include <alps/alea/mcdata.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <iterator>
#include <algorithm>


int main(int argc, char** argv)
{
  using namespace alps::alea;

  std::cout.precision(10);

  // test: mcdata<double>

  mcdata<double> a(0.81,0.1);
  std::cout << "\na: \t" << a << "\n";

  mcdata<double> b = mcdata<double>(1.21,0.2);
  std::cout << "\nb: \t" << b << "\n";

  mcdata<double> c = mcdata<double>(4.5,0.3);
  std::cout << "\nb: \t" << c << "\n";


  // positivity, negtivity, absolute
  std::cout << "\n +a: \t" << +a << "\n";
  std::cout << "\n -a: \t" << -a << "\n";
  std::cout << "\n abs(c): \t" << abs(c) << "\n";

  // operation-assign
  mcdata<double> e = a;

  a = e;
  a += b;
  std::cout << "\na += b: \t" << a << "\n";

  a = e;
  a -= b;
  std::cout << "\na -= b: \t" << a << "\n";

  a = e;
  a *= b;
  std::cout << "\na *= b: \t" << a << "\n";

  a = e;
  a /= b;
  std::cout << "\na /= b: \t" << a << "\n";

  a = e;
  a += 2.;
  std::cout << "\na += 2.: \t" << a << "\n";

  a = e;
  a -= 2.;
  std::cout << "\na -= 2.: \t" << a << "\n";

  a = e;
  a *= 2.;
  std::cout << "\na *= 2.: \t" << a << "\n";

  a = e;
  a /= 2.;
  std::cout << "\na /= 2.: \t" << a << "\n";

  a = e;


  // operators
  std::cout << "\na + b: \t" << a + b << "\n";
  std::cout << "\na - b: \t" << a - b << "\n";
  std::cout << "\na * b: \t" << a * b << "\n";
  std::cout << "\na / b: \t" << a / b << "\n";
 
  std::cout << "\na + 2.: \t" << a + 2. << "\n";
  std::cout << "\na - 2.: \t" << a - 2. << "\n";
  std::cout << "\na * 2.: \t" << a * 2. << "\n";
  std::cout << "\na / 2.: \t" << a / 2. << "\n";

  std::cout << "\n2 + a: \t" << 2. + a << "\n";
  std::cout << "\n2 - a: \t" << 2. - a << "\n";
  std::cout << "\n2 * a: \t" << 2. * a << "\n";
  std::cout << "\n2 / a: \t" << 2. / a << "\n";  

  // (pow, sq, sqrt, cb, cbrt, exp, log) operations

  mcdata<double> res;

  res = pow(a,2.71);
  std::cout << "\na^(2.71): \t";
  std::cout << res << std::endl;

  std::cout << "\nsq(a): \t" << sq(a) << "\n";
  std::cout << "\ncb(a): \t" << cb(a) << "\n";
  std::cout << "\nsqrt(a): \t" << sqrt(a) << "\n";
  std::cout << "\ncbrt(a): \t" << cbrt(a) << "\n";
  std::cout << "\nexp(a): \t" << exp(a) << "\n";
  std::cout << "\nlog(a): \t" << log(a) << "\n";


  // (sin,...atanh) operations
  std::cout << "\nsin(a): \t" << sin(a) << "\n";
  std::cout << "\ncos(a): \t" << cos(a) << "\n";
  std::cout << "\ntan(a): \t" << tan(a) << "\n";
  std::cout << "\nasin(a): \t" << asin(a) << "\n";
  std::cout << "\nacos(a): \t" << acos(a) << "\n";
  std::cout << "\natan(a): \t" << atan(a) << "\n";
  std::cout << "\nsinh(a): \t" << sinh(a) << "\n";
  std::cout << "\ncosh(a): \t" << cosh(a) << "\n";
  std::cout << "\ntanh(a): \t" << tanh(a) << "\n";
// asinh, aconsh and atanh are not part of C++03 standard
//  std::cout << "\nasinh(a): \t" << asinh(a) << "\n";
//  std::cout << "\nacosh(b): \t" << acosh(b) << "\n";
//  std::cout << "\natanh(a): \t" << atanh(a) << "\n";


  // test: mcdata<std::vector<double> >

  mcdata<std::vector<double> > vecA(std::vector<double>(10,0.81),std::vector<double>(10,0.1));
  std::cout << "\nA: \n" << vecA << "\n"; 

  mcdata<std::vector<double> > vecB(std::vector<double>(10,1.21),std::vector<double>(10,0.2));
  std::cout << "\nB: \n" << vecB << "\n";

  mcdata<std::vector<double> > vecC(std::vector<double>(10,-4.5),std::vector<double>(10,0.3));
  std::cout << "\nC: \n" << vecC << "\n";         

  std::vector<double> vec2(10,2.);
  std::cout << "\nvec(2): \n";
  std::copy(vec2.begin(),vec2.end(),std::ostream_iterator<double>(std::cout,"\n"));


  // positivity, negtivity, absolute
  std::cout << "\n +A: \n" << +vecA << "\n";
  std::cout << "\n -A: \n" << (-vecA) << "\n";
  std::cout << "\n abs(C): \n" << abs(vecC) << "\n";


  // operation-assign
  mcdata<std::vector<double> > vecE = vecA;

  vecA =  vecE;
  vecA += vecB;
  std::cout << "\nA += B: \n" << vecA << "\n";

  vecA = vecE;
  vecA -= vecB;
  std::cout << "\nA -= B: \n" << vecA << "\n";

  vecA = vecE;
  vecA *= vecB;
  std::cout << "\nA *= B: \n" << vecA << "\n";

  vecA = vecE;
  vecA /= vecB;
  std::cout << "\nA /= B: \n" << vecA << "\n";

  vecA = vecE;
  vecA += vec2;
  std::cout << "\nvecA += vec(2.): \n" << vecA << "\n";

  vecA = vecE;
  vecA -= vec2;
  std::cout << "\nvecA -= vec(2.): \n" << vecA << "\n";

  vecA = vecE;
  vecA *= vec2;
  std::cout << "\nvecA *= vec(2.): \n" << vecA << "\n";

  vecA = vecE;
  vecA /= vec2;
  std::cout << "\nvecA /= vec(2.): \n" << vecA << "\n";

  vecA = vecE;

  // operators
  std::cout << "\nA + B: \n" << vecA + vecB << "\n";
  std::cout << "\nA - B: \n" << vecA - vecB << "\n";
  std::cout << "\nA * B: \n" << vecA * vecB << "\n";
  std::cout << "\nA / B: \n" << vecA / vecB << "\n";
 
  std::cout << "\nA + vec(2.): \n" << vecA + vec2 << "\n";
  std::cout << "\nA - vec(2.): \n" << vecA - vec2 << "\n";
  std::cout << "\nA * vec(2.): \n" << vecA * vec2 << "\n";
  std::cout << "\nA / vec(2.): \n" << vecA / vec2 << "\n";

  std::cout << "\nvec(2.) + A: \n" << vec2 + vecA << "\n";
  std::cout << "\nvec(2.) - A: \n" << vec2 - vecA << "\n";
  std::cout << "\nvec(2.) * A: \n" << vec2 * vecA << "\n";
  std::cout << "\nvec(2.) / A: \n" << vec2 / vecA << "\n";

  std::cout << "\nA + 2.: \n" << vecA + 2. << "\n";
  std::cout << "\nA - 2.: \n" << vecA - 2. << "\n";
  std::cout << "\nA * 2.: \n" << vecA * 2. << "\n";
  std::cout << "\nA / 2.: \n" << vecA / 2. << "\n";

  std::cout << "\n2. + A: \n" << 2. + vecA << "\n";
  std::cout << "\n2. - A: \n" << 2. - vecA << "\n";
  std::cout << "\n2. * A: \n" << 2. * vecA << "\n";
  std::cout << "\n2. / A: \n" << 2. / vecA << "\n";

  // (pow, sq, sqrt, cb, cbrt, exp, log) operations

  mcdata<std::vector<double> > resV;

  resV = pow(vecA,2.71);
  std::cout << "\nA^(2.71): \n";
  std::cout << resV << std::endl;

  std::cout << "\nsq(A): \t" << sq(vecA) << "\n";
  std::cout << "\ncb(A): \t" << cb(vecA) << "\n";
  std::cout << "\nsqrt(A): \t" << sqrt(vecA) << "\n";
  std::cout << "\ncbrt(A): \t" << cbrt(vecA) << "\n";
  std::cout << "\nexp(A): \t" << exp(vecA) << "\n";
  std::cout << "\nlog(A): \t" << log(vecA) << "\n";


  // (sin,...atanh) operations
  std::cout << "\nsin(A): \t" << sin(vecA) << "\n";
  std::cout << "\ncos(A): \t" << cos(vecA) << "\n";
  std::cout << "\ntan(A): \t" << tan(vecA) << "\n";
  std::cout << "\nasin(A): \t" << asin(vecA) << "\n";
  std::cout << "\nacos(A): \t" << acos(vecA) << "\n";
  std::cout << "\natan(A): \t" << atan(vecA) << "\n";
  std::cout << "\nsinh(A): \t" << sinh(vecA) << "\n";
  std::cout << "\ncosh(A): \t" << cosh(vecA) << "\n";
  std::cout << "\ntanh(A): \t" << tanh(vecA) << "\n";
// asinh, aconsh and atanh are not part of C++03 standard
//  std::cout << "\nasinh(A): \t" << asinh(vecA) << "\n";
//  std::cout << "\nacosh(B): \t" << acosh(vecB) << "\n";
//  std::cout << "\natanh(A): \t" << atanh(vecA) << "\n";


  // TODO: do we need that?
  // testing vector support (non-STL)
 /*
  vecA.push_back(b);
  std::cout << "\nvecA.push_back(b)\n";
  std::cout << "\n(A): \t" << vecA << "\n";

  vecA.pop_back();
  std::cout << "\nvecA.pop_back() \n";
  std::cout << "\n(A): \t" << vecA << "\n";

  vecA.clear();
  std::cout << "\nvecA.clear() \n";
  std::cout << "\n(A): \t" << vecA << "\n";
  
  mcdata<std::vector<double> > vecX;
  vecX.push_back(a);
  vecX.push_back(b);
  vecX.push_back(c);
  vecX.push_back(a);
  vecX.push_back(b);
  vecX.push_back(c);
  std::cout << "\nvecX: \n" << vecX << "\n";

  vecX.at(4);
  std::cout << "\nvecX.at(4): \t" << vecX.at(4) << "\n";

  vecX.insert(2,c);
  std::cout << "\nvecX.insert(2,c): \n:" << vecX << "\n";

  vecX.erase(0);
  std::cout << "\nvecX.erase(0): \n:" << vecX << "\n";


  // testing interface interchange
  std::vector<mcdata<double> > vec_of_vwe;
  mcdata<std::vector<double> > vec_with_error;

  vec_with_error = vecX;
  vec_of_vwe = obtain_vector_of_mcdata_from_vector_with_error<double>(vec_with_error);

  std::cout << "\nSuccessful converting from vec_with_error: \n" << vec_with_error << "to vec_of_vwe: \n";
  std::copy(vec_of_vwe.begin(),vec_of_vwe.end(),std::ostream_iterator<mcdata<double> >(std::cout,"\n"));

  std::cout << std::endl;

  vec_with_error.clear();
  vec_with_error = obtain_vector_with_error_from_vector_of_mcdata(vec_of_vwe);

  std::cout << "\nSuccessful converting from vec_of_vwe: \n";
  std::copy(vec_of_vwe.begin(),vec_of_vwe.end(),std::ostream_iterator<mcdata<double> >(std::cout,"\n"));
  std::cout << "to vec_with_error:\n" << vec_with_error;
*/  

  return 0;
}

