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
#include <algorithm>


int main(int argc, char** argv)
{
  std::ifstream inFile;

//### CONSTRUCTORS

  // empty constructor
  alps::alea::binned_data<double> data1;
  std::cout << "\ndata1 :\n" << data1 << "\n";

  // constructor from AbstractSimpleObservable
  alps::RealObservable obs2("obs2");
  obs2.reset(true);
  inFile.open("binned_data.test.input",std::ios::in);
  std::string obs2_elem_str;  while (std::getline(inFile,obs2_elem_str))  {  std::istringstream iss(obs2_elem_str);  double obs2_elem;  iss >> obs2_elem;  obs2 << obs2_elem; } 
  inFile.close();
  std::cout << "\nobs2: \n" << obs2 << "\n";
  
  alps::alea::binned_data<double> data2(obs2);
  std::cout << "data2 :\n" << data2 << "\n";

  // constructor from a slice of binned_data
  alps::alea::binned_data<double> data3(data2,2);
  std::cout << "\ndata3: \n" << data3 << "\n";

  return 0;
}
