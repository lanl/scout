/*
 * -----  Scout Programming Language -----
 *
 * This file is distributed under an open source license by Los Alamos
 * National Security, LCC.  See the file License.txt (located in the
 * top level of the source distribution) for details.
 * 
 *-----
 * 
 */

#ifndef SCOUT_GPU_H_
#define SCOUT_GPU_H_

#include <cstdlib>
#include <string>

namespace scout{

  class gpu_rt{
    gpu_rt();

    ~gpu_rt();

    size_t numDevices() const;
      
    size_t clockRate(size_t device) const;

    size_t memory(size_t device) const;

    size_t numMultiProcessors(size_t device) const;

    const std::string& vendor(size_t device) const;
    
  private:
    class gpu_rt_* x_;
  };

} // end namespace scout

#endif // SCOUT_GPU_H_
