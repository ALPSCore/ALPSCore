//
// Created by Sergei on 8/2/18.
//

#ifndef ALPSCORE_MPI_SHARED_ALLOCATOR_HPP
#define ALPSCORE_MPI_SHARED_ALLOCATOR_HPP

#include <alps/config.hpp>

#ifdef ALPS_HAVE_MPI
#include <mpi.h>


namespace alps {
  namespace numerics {
    namespace detail {
      template <typename T>
      class mpi_shared_allocator {
      public:

        mpi_shared_allocator(MPI_Comm& comm, MPI_Win& win) : comm_(&comm), win_(&win) {

        };

        T* allocate(size_t size) {
          void *mem;
          // compute local size for array
          size_t nloc = local_size(size);
          // allocate shared memory
          MPI_Win_allocate_shared(nloc * sizeof(T), sizeof(T), MPI_INFO_NULL, *comm_, &mem, win_);
          MPI_Aint alloc_length;
          int disp_unit;
          // get pointer to the 0-th element of the global array
          MPI_Win_shared_query(*win_, 0, &alloc_length, &disp_unit, &mem);
          return (T*) mem;
        }

        void deallocate(T* data) {
          // we do not release anything.
          // one should explicitly release MPI-window in the very original place.
        }
        void lock(){
          lock_ = true;
          MPI_Win_lock_all(MPI_MODE_NOCHECK , *win_);
        }
        void release(){
          MPI_Win_sync(*win_);
          MPI_Win_unlock_all(*win_);
          lock_ = false;
        }
        bool locked() {return lock_;}
      private:
        /**
         * compute local size for global array
         * @param n -- global size
         * @return local size
         */
        size_t local_size(size_t n) const {
          int s, r;
          MPI_Comm_size(*comm_, &s);
          MPI_Comm_rank(*comm_, &r);
          size_t nloc = n / s;
          if(n%s > r) nloc++;
          return nloc;
        }

        MPI_Comm *comm_;
        MPI_Win *win_;
        bool lock_;
      };
    }
  }
}
#endif //ALPS_HAVE_MPI
#endif //ALPSCORE_MPI_SHARED_ALLOCATOR_HPP
