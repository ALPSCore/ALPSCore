/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2003 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Synge Todo <wistaria@comp-phys.org>
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

/* $Id$ */

// TODO: bool, uint64_t, int64_t

#include <alps/osiris/comm.h>
#include <alps/osiris/mpdump.h>
#include <alps/osiris/process.h>
#include <boost/lexical_cast.hpp>
#include <boost/throw_exception.hpp>
#include <cstdio>
#include <stdexcept>

#ifdef ALPS_PVM
// some PVM implementations require <stdio.h>
# include <stdio.h>
# include <pvm3.h>
#endif

#ifdef ALPS_MPI
# include <mpi.h>
#endif

namespace alps {

//=======================================================================
// MPODump
//
// derived from ODump
//
// objects written int32_to this dump can be sent to another process, where
// they can be received as an IDump
//-----------------------------------------------------------------------

// create an empty buffer
OMPDump::OMPDump() : valid_(false)
{ init(); }

// delete the buffer
OMPDump::~OMPDump()
{
/*
#ifdef ALPS_PVM
  if (valid) {
    pvm_freebuf(bufid);
    pvm_setsbuf(oldbufid);
  }
#endif
*/
}

// reinitialize the buffer
void OMPDump::init()
{
#ifdef ALPS_PVM
  valid_=true;
  // oldbufid=pvm_getsbuf();
  // bufid=pvm_mkbuf(PvmDataDefault);
  bufid_=pvm_initsend(PvmDataDefault);
  // pvm_setsbuf(bufid);
#endif

#ifdef ALPS_MPI
  valid_=true;
  buf_.clear();
#endif
}


//-----------------------------------------------------------------------
// MESSAGE PASSING
//-----------------------------------------------------------------------

// send the dump to a given process with a given message id

#ifdef ALPS_PVM
void OMPDump::send(const Process& where,int32_t t)
{
#ifdef ALPS_TRACE
  std::cerr << "Sending message " << t << " to process " << where << ".\n";
#endif

#ifdef ALPS_DEBUG
  if(!valid_)
    boost::throw_exception ( std::logic_error("message not initialized in OMPDump::send"));
#endif

  int info = pvm_send(where,t);

  valid_=false;
  //pvm_freebuf(bufid);
  //pvm_setsbuf(oldbufid);
  // check return code
  
  if(info<0)
    boost::throw_exception( std::runtime_error( ("Error " + boost::lexical_cast<std::string,int>(info) + " wen sending message")));
}

#else
#ifdef ALPS_MPI

void OMPDump::send(const Process& where,int32_t t)
{
#ifdef ALPS_TRACE
  std::cerr << "Sending message " << t << " to process " << where << ".\n";
#endif

#ifdef ALPS_DEBUG
  if(!valid_)
    boost::throw_exception ( std::logic_error("message not initialized in OMPDump::send"));
#endif
  int info;
  if( (info = MPI_Send(buf_, buf_.size(), MPI_BYTE, where, t, MPI_COMM_WORLD))!=0)
    boost::throw_exception( std::runtime_error( ("Error " + boost::lexical_cast<std::string,int>(info) + " wen sending message")));
}

#else

void OMPDump::send(const Process&,int32_t) 
{
  boost::throw_exception( std::logic_error("message passing useless in single CPU programs" ));
}
#endif
#endif

// send message to several processes

void OMPDump::send(const ProcessList& where,int32_t t)
{
#ifdef ALPS_DEBUG
  if(!valid_)
    boost::throw_exception( std::runtime_error( "message not initialized in osiris::OMPDump::send"));
#endif

#ifdef ALPS_PVM
  // create an array to hold the id's
  
  if(where.size())
    {
      boost::scoped_array<int> tids ( new int[where.size()]);
      for (int i=0;i<where.size();i++)
        tids[i]=where[i];
    
      int info=pvm_mcast(tids.get(),where.size(),t);
      valid_=false;
      //pvm_freebuf(bufid);
      //pvm_setsbuf(oldbufid);
  
      // check return code
  
      if(info<0)
        boost::throw_exception( std::runtime_error( ("Error " + boost::lexical_cast<std::string,int>(info) + " wen sending message")));
    } 
#else

  // default action:
  // send to all processes and 
  // return the first (if any) nonzero return value
  
  for (int i=0;i<where.size();i++)
    if(!where[i].local())
       send(where[i],t);
#endif
}


//-----------------------------------------------------------------------
// WRITE AND READ TYPES
//-----------------------------------------------------------------------

#ifdef ALPS_PVM

namespace detail {
static int pvm_not_implemented(const void*, int, int)
{
  boost::throw_exception(std::logic_error("(un)packing of a type not implemented"));
  return 0;
}
}

#define ALPS_DUMP_DO_TYPE(T,PVMFUNC,PVMUFUNC,PVMTYPE) \
void OMPDump::write_simple(T x) { \
  int info = PVMFUNC(reinterpret_cast<PVMTYPE *>(&x), 1, 1); \
  if (info < 0) boost::throw_exception(std::runtime_error(("Error " + boost::lexical_cast<std::string, int>(info) + " when writing"))); \
} \
void OMPDump::write_array(std::size_t n, const T* p) { \
  int info = PVMFUNC(reinterpret_cast<PVMTYPE *>(const_cast<T *>(p)), n, 1); \
  if (info < 0) boost::throw_exception(std::runtime_error(("Error " + boost::lexical_cast<std::string, int>(info) + " when writing"))); \
} \
void IMPDump::read_simple(T& x) { \
  int info = PVMUFUNC(reinterpret_cast<PVMTYPE *>(&x), 1, 1); \
  if (info < 0) boost::throw_exception(std::runtime_error(("Error " + boost::lexical_cast<std::string, int>(info) + " when reading"))); \
} \
void IMPDump::read_array(std::size_t n, T *p) { \
  int info = PVMUFUNC(reinterpret_cast<PVMTYPE *>(p), n, 1); \
  if (info < 0) boost::throw_exception(std::runtime_error(("Error " + boost::lexical_cast<std::string, int>(info) + " when reading"))); \
}

#elif defined(ALPS_MPI)

#define ALPS_DUMP_DO_TYPE(T,A,B,C) \
void OMPDump::write_simple(T x) { buf_.write(x);} \
void OMPDump::write_array(std::size_t n,const T *p) {buf_.write(p,n);} \
void IMPDump::read_simple(T& x) {buf_.read(x);} \
void IMPDump::read_array(std::size_t n,T *p) {buf_.read(p,n);}

#else

#define ALPS_DUMP_DO_TYPE(T,A,B,C) \
void OMPDump::write_simple(T) { \
  boost::throw_exception(std::logic_error("message passing useless for single process programs")); \
} \
void IMPDump::read_array(std::size_t n, T *p) {\
  IDump::read_array(n,p);}\
void IMPDump::read_simple(T&) {\
  boost::throw_exception(std::logic_error("message passing useless for single process programs")); \
} \
void OMPDump::write_array(std::size_t n, const T *p) {\
  ODump::write_array(n,p); \
}

#endif

ALPS_DUMP_DO_TYPE(bool, pvm_pkbyte, pvm_upkbyte, char)
ALPS_DUMP_DO_TYPE(char, pvm_pkbyte, pvm_upkbyte, char)
ALPS_DUMP_DO_TYPE(signed char, pvm_pkbyte, pvm_upkbyte, char)
ALPS_DUMP_DO_TYPE(unsigned char, pvm_pkbyte, pvm_upkbyte, char)
ALPS_DUMP_DO_TYPE(short, pvm_pkshort, pvm_upkshort, short)
ALPS_DUMP_DO_TYPE(unsigned short, pvm_pkushort, pvm_upkushort, unsigned short)
ALPS_DUMP_DO_TYPE(int, pvm_pkint, pvm_upkint, int)
ALPS_DUMP_DO_TYPE(unsigned int, pvm_pkuint, pvm_upkuint, unsigned int)
ALPS_DUMP_DO_TYPE(long, pvm_pklong, pvm_upklong, long)
ALPS_DUMP_DO_TYPE(unsigned long, pvm_pkulong, pvm_upkulong, unsigned long)
#ifdef BOOST_HAS_LONG_LONG
ALPS_DUMP_DO_TYPE(long long, detail::pvm_not_implemented, detail::pvm_not_implemented, long long)
ALPS_DUMP_DO_TYPE(unsigned long long, detail::pvm_not_implemented, detail::pvm_not_implemented, unsigned long long)
#endif
ALPS_DUMP_DO_TYPE(float, pvm_pkfloat, pvm_upkfloat, float)
ALPS_DUMP_DO_TYPE(double, pvm_pkdouble,pvm_upkdouble,double)
ALPS_DUMP_DO_TYPE(long double, detail::pvm_not_implemented, detail::pvm_not_implemented, long double)
# undef ALPS_DUMP_DO_TYPE


//-----------------------------------------------------------------------
// WRITE A STRING
//-----------------------------------------------------------------------

#ifdef ALPS_PVM
void OMPDump::write_string(std::size_t, const char* x)
{
  int info=pvm_pkstr((char*) x);
  if(info<0) 
    boost::throw_exception( std::runtime_error( ("Error " + boost::lexical_cast<std::string,int>(info) + " when writing string")));
}
#else
#ifdef ALPS_MPI
void OMPDump::write_string(std::size_t n, const char* x)
{
  buf_.write(x,n);
}
#else
void OMPDump::write_string(std::size_t n, const char* x)
{
  ODump::write_string(n,x);
}
#endif
#endif

//=======================================================================
// MPIDump
//
// derived from IDump
//
// can be receive as a message
//-----------------------------------------------------------------------


// make an empty buffer

IMPDump::IMPDump()
  : valid_(false)
{
  init();
}


// re-initialize the buffer and reset everything

void IMPDump::init()
{
  theSender_=Process();
#ifdef ALPS_MPI
  buf_.clear();
#endif
}
IMPDump::IMPDump(int32_t t)
{
  init();
  receive(t);
}

IMPDump::IMPDump(const Process& p, int32_t t)
{
  init();
  receive(p,t);
}

//-----------------------------------------------------------------------
// MESSAGE PASSING FUNCTIONS
//-----------------------------------------------------------------------

// return the sender of the message

const Process& IMPDump::sender() const
{
  return theSender_;
}

// receive a message
#ifdef ALPS_PVM
void IMPDump::receive(const Process* where,int32_t t)
{
#ifdef ALPS_TRACE
  std::cerr << "Preparing receive\n";
#endif
  int tid = (where ? int(*where) : -1);  
  valid_=false;
  int info = bufid_ = pvm_recv(tid,t);
  if(info<0)
    boost::throw_exception( std::runtime_error( ("Error " + boost::lexical_cast<std::string,int>(info) + " when receiving message")));
  valid_=true;
  
  // get the sender process
  int bytes,msgtag;
  info = pvm_bufinfo(bufid_,&bytes,&msgtag,&tid);
  if(info<0) // fatal error
    boost::throw_exception( std::runtime_error( ("Error " + boost::lexical_cast<std::string,int>(info) + " when receiving message")));
  theSender_=process_from_id(tid);
  
#ifdef ALPS_TRACE
  std::cerr << "Received message " << msgtag << " from process " << tid << ".\n";
#endif
}
#else
#ifdef ALPS_MPI

void IMPDump::receive(const Process* where,int32_t t)
{
  // check for message and size
  int node = MPI_ANY_SOURCE;
  int tag = (t== -1 ? MPI_ANY_TAG : t);
  int info;
  
  MPI_Status status;
  if(where)
    node=(*where);
  if((info=MPI_Probe(node,tag,MPI_COMM_WORLD,&status))!=0)
    boost::throw_exception( std::runtime_error( ("Error " + boost::lexical_cast<std::string,int>(info) + " when receiving message")));

  // set the buffer to the apropriate length
  int cnt;
  MPI_Get_count(&status,MPI_BYTE,&cnt);
  buf_.clear();
  buf_.resize(cnt);
  // receive the message, aborts if an error occurs
  if((info=MPI_Recv(buf_,buf_.size(),MPI_BYTE,status.MPI_SOURCE, status.MPI_TAG,
           MPI_COMM_WORLD,&status))!=0)
    boost::throw_exception( std::runtime_error( ("Error " + boost::lexical_cast<std::string,int>(info) + " when receiving message")));

  // get the sender of the message
  theSender_=process_from_id(status.MPI_SOURCE);
  valid_=true;

#ifdef ALPS_TRACE
  std::cerr << "Received message " << status.MPI_TAG 
       << " from process " << status.MPI_SOURCE << ".\n";
#endif
}

#else

void IMPDump::receive(const Process*,int32_t)
{
  boost::throw_exception(std::logic_error("message passing useless for single process programs") );
}

#endif
#endif

void IMPDump::receive(const Process& where,int32_t t)
{
  receive(&where,t);
}

void IMPDump::receive(int32_t t)
{
  receive(0,t);
}

//-----------------------------------------------------------------------
// READ A STRING
//-----------------------------------------------------------------------

#ifdef ALPS_PVM
void IMPDump::read_string(std::size_t, char* x)
{
  int info=pvm_upkstr(x);
  if(info<0)
    boost::throw_exception( std::runtime_error( ("Error " + boost::lexical_cast<std::string,int>(info) + " when reading string")));
}

#else
#ifdef ALPS_MPI

void IMPDump::read_string(std::size_t n, char* x)
{
  buf_.read(x,n);
}

#else  

void IMPDump::read_string(std::size_t n, char* x)
{
  IDump::read_string(n,x);
}

#endif
#endif

//=======================================================================
// MESSAGE PROBING AND SIGNALING
//-----------------------------------------------------------------------

// check for a message
int32_t IMPDump::probe(const Process& w, int32_t t)
{
  return probe(&w,t);
}

int32_t IMPDump::probe(int32_t t)
{
  return probe(0,t);
}

#ifdef ALPS_PVM
int32_t IMPDump::probe(const Process* w, int32_t t)
{
#ifdef ALPS_TRACE
  std::cerr << "Checking for message " << t << " from " << 
    (w ? int(*w) : -1) << ".\n";
#endif


  int tid = (w ? int (*w) : -1);
  int info = pvm_probe(tid,t);

  if(info < 0)  {
    boost::throw_exception( std::runtime_error( ("Error " + boost::lexical_cast<std::string,int>(info) + " from pvm_probe")));
  }
  else if (info) // got a message
    {
      int tag=t;
      int bytes,id;
      if((info=pvm_bufinfo(info,&bytes,&tag,&id))<0)
      info=tag;
#ifdef ALPS_TRACE
      std::cerr << "Found message " << tag << " from " << id << ".\n";
#endif
    }
  return info;
}
#else
#ifdef ALPS_MPI

int32_t IMPDump::probe(const Process* w, int32_t t)
{
  int node = (w ? int(*w) : MPI_ANY_SOURCE);
  int tag = (t != -1 ? t : MPI_ANY_TAG);
  int flag;
  MPI_Status status;
  
  int info;
  if((info=MPI_Iprobe(node,tag,MPI_COMM_WORLD,&flag,&status))!=0) {
    boost::throw_exception( std::runtime_error( ("Error " + boost::lexical_cast<std::string,int>(info) + " from MPI_Iprobe")));
  }

  if(flag)
    {
#ifdef ALPS_TRACE
      std::cerr << "Found message " << status.MPI_TAG 
                   << " from " << status.MPI_SOURCE << ".\n";
#endif
      return status.MPI_TAG;
    }
  return 0;
}
#else

int32_t IMPDump::probe(const Process* , int32_t)
{
  return 0; // no messages in single CPU case
}
#endif
#endif

} // namespace alps
