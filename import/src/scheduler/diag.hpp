/*****************************************************************************
*
* ALPS Project Applications
*
* Copyright (C) 1994-2006 by Matthias Troyer <troyer@comp-phys.org>
*
* This software is part of the ALPS Applications, published under the ALPS
* Application License; you can use, redistribute it and/or modify it under
* the terms of the license, either version 1 or (at your option) any later
* version.
* 
* You should have received a copy of the ALPS Application License along with
* the ALPS Applications; see the file LICENSE.txt. If not, the license is also
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

/* $Id$ */

#ifndef ALPS_SCHEDULER_DIAG_HPP
#define ALPS_SCHEDULER_DIAG_HPP

#include <alps/type_traits/norm_type.hpp>
#include <alps/scheduler/measurement_operators.h>
#include <alps/model/model_helper.h>
#include <alps/lattice/graph_helper.h>
#include <alps/scheduler/task.h>

#include <alps/numeric/matrix/vector.hpp>
#include <alps/hdf5/numeric_vector.hpp>

namespace alps { namespace scheduler {

template <class T, class G = typename graph_helper<>::graph_type>
class DiagTask : public scheduler::Task, public graph_helper<G>,  public model_helper<>,  public MeasurementOperators
{
public:
  typedef T value_type;
  typedef typename norm_type<value_type>::type magnitude_type;
  typedef alps::numeric::vector<value_type> vector_type;
  typedef alps::numeric::vector<magnitude_type> mag_vector_type;
  typedef model_helper<>::half_integer_type half_integer_type;
  typedef boost::numeric::ublas::mapped_vector_of_mapped_vector<T, boost::numeric::ublas::row_major>  operator_matrix_type;
  
  DiagTask (const ProcessList& where , const boost::filesystem::path& p,bool delay_construct=false);

  void dostep() { boost::throw_exception(std::logic_error("Cannot call dostep on the base class DiagTask")); }

  void save(hdf5::archive &) const;
  void load(hdf5::archive &);
  

protected:
  void write_xml_body(oxstream&, const boost::filesystem::path&,bool) const;
  void handle_tag(std::istream& infile, const XMLTag& tag); 

  std::vector<mag_vector_type> eigenvalues_;
  std::vector<EigenvectorMeasurements<value_type> > measurements_;

  std::vector<std::vector<std::pair<std::string,std::string> > > quantumnumbervalues_;
  
  bool calc_vectors() const { return calc_averages() || print_vectors();}
  bool print_vectors() const { return print_vectors_;}
private:
  bool print_vectors_;
  bool read_hdf5_;
};

template <class T, class G>
DiagTask<T,G>::DiagTask(const ProcessList& where , const boost::filesystem::path& p, bool delay_construct) 
    : scheduler::Task(where,p)
    , graph_helper<G>(this->get_parameters())
    , model_helper<>(this->get_parameters())
    , MeasurementOperators(this->get_parameters())
    , print_vectors_(this->get_parameters().value_or_default("PRINT_EIGENVECTORS",false))
    , read_hdf5_(false)
{
  if (!delay_construct)
    this->construct();
}


template <class T, class G>
void DiagTask<T,G>::load(hdf5::archive & ar) {
  scheduler::Task::load(ar);
  if (ar.is_group("/spectrum/sectors")) {
      std::vector<std::string> list = ar.list_children("/spectrum/sectors");
      measurements_.resize(list.size(),EigenvectorMeasurements<value_type>(*this));
      for (unsigned i=0; i<list.size();++i) {
        std::string sectorpath = "/spectrum/sectors/"+list[i];
        
        // read quantum numbers
        std::vector<std::pair<std::string,std::string> > qnvals;
        if (ar.is_group(sectorpath+"/quantumnumbers")) {
          std::vector<std::string> qnlist = ar.list_children(sectorpath+"/quantumnumbers");
          for (std::vector<std::string>::const_iterator it = qnlist.begin(); it != qnlist.end(); ++it) {
              std::string v;
              ar >> make_pvp(sectorpath+"/quantumnumbers/"+*it, v);
              qnvals.push_back(std::make_pair(*it,v));
          }
        }
        this->quantumnumbervalues_.push_back(qnvals);

        // read energies
        if (ar.is_data(sectorpath+"/energies")) {
          mag_vector_type evals_vector;
          ar >> make_pvp(sectorpath+"/energies", evals_vector);
          eigenvalues_.push_back(evals_vector);
        }
        // read measurements
          ar >> make_pvp(sectorpath,measurements_[i]);
      }
  }
  std::cerr << eigenvalues_.size() << " sectors\n";
  this->read_hdf5_ = true; // skip XML, once all is being read
}

template <class T, class G>
void DiagTask<T,G>::save(hdf5::archive & ar) const {
  scheduler::Task::save(ar);
  for (unsigned  i=0;i<eigenvalues_.size();++i) {
    std::string sectorpath = "/spectrum/sectors/" + boost::lexical_cast<std::string>(i);
    for (unsigned j=0;j<this->quantumnumbervalues_[i].size();++j)
      ar << make_pvp(sectorpath + "/quantumnumbers/" + this->quantumnumbervalues_[i][j].first,
                     this->quantumnumbervalues_[i][j].second);
      ar << make_pvp(sectorpath + "/energies",eigenvalues_[i]);
    if (calc_averages() || this->parms.value_or_default("MEASURE_ENERGY",true))
      ar << make_pvp(sectorpath,measurements_[i]);
  }
}


template <class T, class G>
void DiagTask<T,G>::write_xml_body(oxstream& out, const boost::filesystem::path& name,bool writeallxml) const
{
  if (writeallxml) {
    for (unsigned i=0;i<eigenvalues_.size();++i) {
      unsigned num_eigenvalues = std::min(unsigned(this->parms.value_or_default("NUMBER_EIGENVALUES",
                  eigenvalues_[i].size())),unsigned(eigenvalues_[i].size()));
      out << start_tag("EIGENVALUES") << attribute("number",num_eigenvalues);
      for (unsigned j=0;j<this->quantumnumbervalues_[i].size();++j)
        out << start_tag("QUANTUMNUMBER") << attribute("name",this->quantumnumbervalues_[i][j].first)
            << attribute("value",this->quantumnumbervalues_[i][j].second) << end_tag("QUANTUMNUMBER");
      for (unsigned j=0;j<num_eigenvalues;++j)
        out << eigenvalues_[i][j] << "\n";
      out << end_tag("EIGENVALUES");
    }

    if (calc_averages() || this->parms.value_or_default("MEASURE_ENERGY",true)) {
      for (unsigned i=0;i<eigenvalues_.size();++i) {
         unsigned num_eigenvalues = std::min(unsigned(this->parms.value_or_default("NUMBER_EIGENVALUES",
                  eigenvalues_[i].size())),unsigned(eigenvalues_[i].size()));
        out << start_tag("EIGENSTATES") << attribute("number",num_eigenvalues);
        for (unsigned j=0;j<this->quantumnumbervalues_[i].size();++j)
          out << start_tag("QUANTUMNUMBER") << attribute("name",this->quantumnumbervalues_[i][j].first)
              << attribute("value",this->quantumnumbervalues_[i][j].second) << end_tag("QUANTUMNUMBER");
        for (unsigned j=0;j<num_eigenvalues;++j) {
          out << start_tag("EIGENSTATE") << attribute("number",j);
          measurements_[i].write_xml_one_vector(out,name,j);
          out << end_tag("EIGENSTATE");   
        }
        out << end_tag("EIGENSTATES");
      }
    }
  }
}
   
template <class T, class G>
void DiagTask<T,G>::handle_tag(std::istream& infile, const XMLTag& intag) 
{
  XMLTag tag(intag);
  
  // we don't need to read the XML file if the HDF-5 file has already been read
  if (this->read_hdf5_) {
     skip_element(infile,tag);
    return;
  }
  
  if (intag.type==XMLTag::SINGLE)
    return;
  if (intag.name=="EIGENVALUES") {
    // we don't need to read the XML file if the HDF-5 file has already been read

    std::vector<std::pair<std::string,std::string> > qnvals;
    std::vector<magnitude_type> evals;
    char c;
    infile >> c;
    while (c=='<' && infile) {
      infile.putback(c);
      tag=parse_tag(infile);
      if (tag.name=="QUANTUMNUMBER") {
        qnvals.push_back(std::make_pair(tag.attributes["name"],tag.attributes["value"]));
      }
      else if (tag.name=="/EIGENVALUES")
        return;
      skip_element(infile,tag);
      infile >> c;
    }
    do {
      infile.putback(c);
      magnitude_type ev;
      infile >> ev >> c;
      evals.push_back(ev);
    } while (c!='<' && infile);
    infile.putback(c);
    tag=parse_tag(infile);
    if (tag.name!="/EIGENVALUES")
      boost::throw_exception(std::runtime_error("Encountered unexpected tag " + tag.name 
                                  + " while pasrsing <EIGENVALUES>\n"));
    mag_vector_type evals_vector(evals.size());
    std::copy(evals.begin(),evals.end(),evals_vector.begin());
    eigenvalues_.push_back(evals_vector);
    this->quantumnumbervalues_.push_back(qnvals);
  }
  else if (intag.name=="EIGENSTATES") {
    measurements_.push_back(EigenvectorMeasurements<value_type>(*this));
    std::vector<std::pair<std::string,half_integer_type> > qnvals;
    XMLTag tag=parse_tag(infile);
    while (tag.name !="/EIGENSTATES") {
      if (tag.name=="QUANTUMNUMBER")
        skip_element(infile,tag);
      else if (tag.name=="EIGENSTATE") {
        if (tag.type != XMLTag::SINGLE) {
          tag = parse_tag(infile);
          tag = measurements_.rbegin()->handle_tag(infile,tag);
          if (tag.name!="/EIGENSTATE") 
            boost::throw_exception(std::runtime_error("unexpected element " + tag.name + " inside <EIGENSTATE>"));
        }
      }
      tag=parse_tag(infile);
    }
  }    
  else
    skip_element(infile,intag);
    
}

} } // name space scheduler

#endif // ALPS_SCHEDULER_DIAG_HPP

