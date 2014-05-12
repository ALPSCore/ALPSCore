/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2010 by Ping Nang Ma <pingnang@itp.phys.ethz.ch>,
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

/* $Id: nobinning.h 3520 2010-03-21 10:00:00Z gamperl $ */

#include <alps/alea.h>
#include <alps/alea/binned_data.h>

#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <valarray>
#include <vector>
#include <alps/numeric/vector_functions.hpp>
#include <alps/numeric/valarray_functions.hpp>
#include <boost/random.hpp>
#include <algorithm>



int main(int argc, char** argv)
{
  typedef boost::minstd_rand0 random_base_type;
  typedef boost::uniform_01<random_base_type> random_type;
  random_base_type random_int;
  random_type random(random_int);

  std::ifstream inFile;

  using std::operator<<;
  using alps::numeric::operator<<;


//### CONSTRUCTORS

  // empty constructor
  alps::alea::binned_data<double> data1;
  std::cout << "\ndata1 :\n" << data1 << "\n";

  typedef alps::RealVectorObservable RealValarrayObservable;

  // constructor from std::vector
  std::vector<double> vec4;
  for (int i=0; i < 30; ++i)  {  vec4.push_back(i);  }
  std::cout << "\nvec4:\t" << vec4 << "\n";

  alps::alea::binned_data<double> data4(vec4);
  std::cout << "\nalps::alea::binned_data<double> data4(vec4)\ndata4 : \n" << data4 << std::endl;

  data4.set_bin_number(10);
  std::cout << "\ndata4.set_bin_number(10)\ndata4 : \n" << data4 << std::endl;

  alps::alea::binned_data<double> data4a(vec4,10);
  std::cout << "\nalps::alea::binned_data<double> data4a(vec4,10)\ndata4a : \n" << data4a << std::endl;

  std::cout << "\nJackknife mean and error of data4a : \t" << data4a.mean() << "\t" << data4a.error() << "\n";

  // constructor from AbstractSimpleObservable 
  alps::RealObservable obs2("obs2");
  obs2.reset(true);
  for (int i=0; i < 10; ++i)  {  obs2 << random();  }

  std::cout << "\nobs2: \n" << obs2 << "\n";
  
  alps::alea::binned_data<double> data2(obs2);
  std::cout << "data2 (valarray):\n" << data2 << "\n";

  // constructor from a slice of binned_data
  RealValarrayObservable OBS3("OBS3");
  OBS3.reset(true);
  for (int i=0; i < 10; ++i)  
  {
    std::valarray<double> OBS3_elem(5);
    std::fill(&OBS3_elem[0],&OBS3_elem[5],static_cast<double>(i));
    OBS3 << OBS3_elem;
  }
  std::cout << "\nOBS3: \n" << OBS3 << "\n";

  alps::alea::binned_data<std::valarray<double> > data3(OBS3);

  alps::alea::binned_data<double> data3_0(data3,0);
  std::cout << "\ndata3_0 : \n" << data3_0 << "\n";

  alps::alea::binned_data<double> data3_1(data3,1);
  std::cout << "\ndata3_1 : \n" << data3_1 << "\n";

  alps::alea::binned_data<double> data2x = data2;
  std::cout << "\ndata2x (valarray) (= data2):\n " << data2x << std::endl;

  alps::alea::binned_data<std::valarray<double> > data3x = data3;

  alps::alea::binned_data<double> data3x_0 = data3.slice(0);
  std::cout << "\ndata3x_0 (= data3_0):\n" << data3x_0 << std::endl;
  
  alps::alea::binned_data<double> data3x_1 = data3.slice(1);  
  std::cout << "\ndata3x_1 (= data3_1):\n" << data3x_1 << std::endl;

  
//### comparison operator 
  if (data4 == data4a)   {  std::cout << "\ndata4 == data4a\n";  }
  else                   {  std::cout << "\ndata4 != data4a\n";  }


//### numerical operators
  alps::alea::binned_data<double> data5;

  data5 = data4a;
  std::cout << "\ndata5 :\n" << data5 << "\n";
  data5 += data5;
  std::cout << "\ndata5 += data5 , data5 :\n" << data5 << "\n";

  data5 = data4a;
  std::cout << "\ndata5 :\n" << data5 << "\n";
  data5 -= data5;
  std::cout << "\ndata5 -= data5 , data5 :\n" << data5 << "\n";

  data5 = data4a;
  std::cout << "\ndata5 :\n" << data5 << "\n";
  data5 *= data5;
  std::cout << "\ndata5 *= data5 , data5 :\n" << data5 << "\n";

  data5 = data4a;
  std::cout << "\ndata5 :\n" << data5 << "\n";
  data5 /= data5;
  std::cout << "\ndata5 /= data5 , data5 :\n" << data5 << "\n";
  
  data5 = data4a;
  std::cout << "\ndata5 :\n" << data5 << "\n";
  data5 += 2.;
  std::cout << "\ndata5 += 2. , data5 :\n" << data5 << "\n";

  data5 = data4a;
  std::cout << "\ndata5 :\n" << data5 << "\n";
  data5 -= 3.;
  std::cout << "\ndata5 -= 3. , data5 :\n" << data5 << "\n";

  data5 = data4a;
  std::cout << "\ndata5 :\n" << data5 << "\n";
  data5 *= -2.;
  std::cout << "\ndata5 *= -2. , data5 :\n" << data5 << "\n";

  data5 = data4a;
  std::cout << "\ndata5 :\n" << data5 << "\n";
  data5 /= 4.;
  std::cout << "\ndata5 /= 4. , data5 :\n" << data5 << "\n";


  return 0;
}
