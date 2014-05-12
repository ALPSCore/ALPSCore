/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2010 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

#include <alps/config.h>

#ifdef ALPS_HAVE_MPI
# undef SEEK_SET
# undef SEEK_CUR
# undef SEEK_END
# include <mpi.h>
#endif
#include <alps/osiris/comm.h>
#include <alps/osiris/mpdump.h>
#include <alps/osiris/process.h>
#include <boost/lexical_cast.hpp>
#include <boost/throw_exception.hpp>
#include <cstdio>
#include <stdexcept>

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
}

// reinitialize the buffer
void OMPDump::init()
{
#ifdef ALPS_HAVE_MPI
  valid_=true;
  buf_.clear();
#endif
}


//-----------------------------------------------------------------------
// MESSAGE PASSING
//-----------------------------------------------------------------------

// send the dump to a given process with a given message id

#ifdef ALPS_HAVE_MPI

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
    boost::throw_exception( std::runtime_error( ("Error " + boost::lexical_cast<std::string,int>(info) + " when sending message")));
}

#else

void OMPDump::send(const Process&,int32_t)
{
  boost::throw_exception( std::logic_error("message passing useless in single CPU programs" ));
}
#endif

// send message to several processes

void OMPDump::send(const ProcessList& where,int32_t t)
{
#ifdef ALPS_DEBUG
  if(!valid_)
    boost::throw_exception( std::runtime_error( "message not initialized in osiris::OMPDump::send"));
#endif

  // default action:
  // send to all processes and
  // return the first (if any) nonzero return value

  for (std::size_t i=0; i < where.size(); ++i)
    if(!where[i].local())
       send(where[i],t);
}

#ifdef ALPS_HAVE_MPI

void OMPDump::broadcast(const alps::Process &thisprocess)
{
#ifdef ALPS_TRACE
  std::cerr << "broadcasting message " << " to process " << where << ".\n";
#endif

#ifdef ALPS_DEBUG
  if(!valid_)
    boost::throw_exception ( std::logic_error("message not initialized in OMPDump::send"));
#endif
  int info;
  //this sucks: MPI_Probe does not work for broadcast operations, so first we
  //communicate the size of the buffer and then the actual buffer. It will cost
  //us some latency... who has a better idea?
  int cnt=buf_.size();
  if( (info = MPI_Bcast(&cnt, 1, MPI_INT, thisprocess, MPI_COMM_WORLD))!=0)
    boost::throw_exception( std::runtime_error( ("Error " + boost::lexical_cast<std::string,int>(info) + " when broadcasting message")));
  if( (info = MPI_Bcast(buf_, cnt, MPI_BYTE, thisprocess, MPI_COMM_WORLD))!=0)
    boost::throw_exception( std::runtime_error( ("Error " + boost::lexical_cast<std::string,int>(info) + " when broadcasting message")));
}

#else

void OMPDump::broadcast(const alps::Process &thisprocess){
  //do nothing here. If someone tries to receive, we'll throw an exception.
}

#endif

//-----------------------------------------------------------------------
// WRITE AND READ TYPES
//-----------------------------------------------------------------------

#ifdef ALPS_HAVE_MPI

#define ALPS_DUMP_DO_TYPE(T) \
void OMPDump::write_simple(T x) { buf_.write(x);} \
void OMPDump::write_array(std::size_t n,const T *p) {buf_.write(p,n);} \
void IMPDump::read_simple(T& x) {buf_.read(x);} \
void IMPDump::read_array(std::size_t n,T *p) {buf_.read(p,n);}

#else

#define ALPS_DUMP_DO_TYPE(T) \
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

ALPS_DUMP_DO_TYPE(bool)
ALPS_DUMP_DO_TYPE(char)
ALPS_DUMP_DO_TYPE(signed char)
ALPS_DUMP_DO_TYPE(unsigned char)
ALPS_DUMP_DO_TYPE(short)
ALPS_DUMP_DO_TYPE(unsigned short)
ALPS_DUMP_DO_TYPE(int)
ALPS_DUMP_DO_TYPE(unsigned int)
ALPS_DUMP_DO_TYPE(long)
ALPS_DUMP_DO_TYPE(unsigned long)
#ifdef BOOST_HAS_LONG_LONG
ALPS_DUMP_DO_TYPE(long long)
ALPS_DUMP_DO_TYPE(unsigned long long)
#endif
ALPS_DUMP_DO_TYPE(float)
ALPS_DUMP_DO_TYPE(double)
ALPS_DUMP_DO_TYPE(long double)
# undef ALPS_DUMP_DO_TYPE


//-----------------------------------------------------------------------
// WRITE A STRING
//-----------------------------------------------------------------------

#ifdef ALPS_HAVE_MPI
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
#ifdef ALPS_HAVE_MPI
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
#ifdef ALPS_HAVE_MPI

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
  theSender_=Process(status.MPI_SOURCE);
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

void IMPDump::receive(const Process& where,int32_t t)
{
  receive(&where,t);
}

void IMPDump::receive(int32_t t)
{
  receive(0,t);
}

#ifdef ALPS_HAVE_MPI
void IMPDump::broadcast(const alps::Process &sender)
{
  // check for message and size
  int info;

  // set the buffer to the apropriate length
  int cnt;
  if((info=MPI_Bcast(&cnt,1,MPI_INT,sender,MPI_COMM_WORLD))!=0)
    boost::throw_exception( std::runtime_error( ("Error " + boost::lexical_cast<std::string,int>(info) + " when receiving message")));
  buf_.resize(cnt);
  // receive the message, aborts if an error occurs
  if((info=MPI_Bcast(buf_,buf_.size(),MPI_BYTE,sender,MPI_COMM_WORLD))!=0)
    boost::throw_exception( std::runtime_error( ("Error " + boost::lexical_cast<std::string,int>(info) + " when receiving message")));

  // get the sender of the message
  theSender_=sender;
  valid_=true;

#ifdef ALPS_TRACE
  std::cerr << "Received message " << "--bcast: no tag--"
       << " from process " << sender << ".\n";
#endif
}

#else

void IMPDump::broadcast(const alps::Process &sender)
{
  throw(std::runtime_error("message passing useless for single process programs"));
}
#endif

//-----------------------------------------------------------------------
// READ A STRING
//-----------------------------------------------------------------------

#ifdef ALPS_HAVE_MPI

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

#ifdef ALPS_HAVE_MPI

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

} // namespace alps
