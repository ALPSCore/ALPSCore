/***************************************************************************
* PALM++/osiris library
*
* osiris/mpdump.C   message passing via dumps
*
* $Id$
*
* Copyright (C) 1994-2002 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Synge Todo <wistaria@comp-phys.org>,
*
* This program is free software; you can redistribute it and/or
* modify it under the terms of the GNU General Public License
* as published by the Free Software Foundation; either version 2
* of the License, or (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
**************************************************************************/

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
  std::cerr << "Sending message " << t << " to process " << where.tid << ".\n";
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
  std::cerr << "Sending message " << t << " to process " << where.tid << ".\n";
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

#define ALPS_DUMP_DO_TYPE(T,XDRFUNC,XDRUFUNC,XDRTYPE) \
void OMPDump::write_simple(T x) { \
  int info = XDRFUNC(reinterpret_cast<XDRTYPE *>(&x), 1, 1); \
  if (info < 0) boost::throw_exception(std::runtime_error(("Error " + boost::lexical_cast<std::string, int>(info) + " when writing"))); \
} \
void OMPDump::write_array(std::size_t n, const T* p) { \
  int info = XDRFUNC(reinterpret_cast<XDRTYPE *>(const_cast<T *>(p)), n, 1); \
  if (info < 0) boost::throw_exception(std::runtime_error(("Error " + boost::lexical_cast<std::string, int>(info) + " when writing"))); \
} \
void IMPDump::read_simple(T& x) { \
  int info = XDRUFUNC(reinterpret_cast<XDRTYPE *>(&x), 1, 1); \
  if (info < 0) boost::throw_exception(std::runtime_error(("Error " + boost::lexical_cast<std::string, int>(info) + " when reading"))); \
} \
void IMPDump::read_array(std::size_t n, T *p) { \
  int info = XDRUFUNC(reinterpret_cast<XDRTYPE *>(p), n, 1); \
  if (info < 0) boost::throw_exception(std::runtime_error(("Error " + boost::lexical_cast<std::string, int>(info) + " when reading"))); \
}

#elif defined(ALPS_MPI)

#define ALPS_DUMP_DO_TYPE(T,XDRFUNC,XDRUFUNC,XDRTYPE) \
void OMPDump::write_simple(T x) { buf_.write(x);} \
void OMPDump::write_array(std::size_t n,const T *p) {buf_.write(p,n);} \
void IMPDump::read_simple(T& x) {buf_.read(x);} \
void IMPDump::read_array(std::size_t n,T *p) {buf_.read(p,n);}

#else

#define ALPS_DUMP_DO_TYPE(T,XDRFUNC,XDRUFUNC,XDRTYPE) \
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
ALPS_DUMP_DO_TYPE(int8_t, pvm_pkbyte, pvm_upkbyte, char)
ALPS_DUMP_DO_TYPE(uint8_t, pvm_pkbyte, pvm_upkbyte, char)
ALPS_DUMP_DO_TYPE(int16_t, pvm_pkshort, pvm_upkshort, int16_t)
ALPS_DUMP_DO_TYPE(uint16_t, pvm_pkushort, pvm_upkushort, uint16_t)
ALPS_DUMP_DO_TYPE(int32_t, pvm_pkint, pvm_upkint, int32_t)
ALPS_DUMP_DO_TYPE(uint32_t, pvm_pkuint, pvm_upkuint, uint32_t)
# ifndef BOOST_NO_INT64_T
ALPS_DUMP_DO_TYPE(int64_t, detail::pvm_not_implemented, detail::pvm_not_implemented, int64_t)
ALPS_DUMP_DO_TYPE(uint64_t, detail::pvm_not_implemented, detail::pvm_not_implemented, uint64_t)
# endif
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
