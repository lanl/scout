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

#ifndef SCOUT_SYSINFO_H_
#define SCOUT_SYSINFO_H_

#include <cstdlib>
#include <string>

namespace scout{

  class sysinfo_summary_rt{
  public:
    sysinfo_summary_rt();

    ~sysinfo_summary_rt();

    size_t totalSockets() const;

    size_t totalNumaNodes() const;

    size_t totalCores() const;

    size_t totalProcessingUnits() const;

    size_t processingUnitsPerCore() const;

    size_t numaNodesPerSocket() const;

    size_t memoryPerSocket() const;
  
    size_t memoryPerNumaNode() const;

    std::string treeToString() const;

  private:
    class sysinfo_summary_rt_* x_;
  };

} // end namespace scout

#endif // SCOUT_SYSINFO_H_
