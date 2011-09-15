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

#ifndef SCOUT_SYSTEM_H_
#define SCOUT_SYSTEM_H_

#include <cstdlib>
#include <string>

namespace scout{

  class system_rt{
  public:
    system_rt();

    ~system_rt();

    size_t totalSockets() const;

    size_t totalNumaNodes() const;

    size_t totalCores() const;

    size_t totalProcessingUnits() const;

    size_t processingUnitsPerCore() const;

    size_t numaNodesPerSocket() const;

    size_t memoryPerSocket() const;
  
    size_t memoryPerNumaNode() const;

    std::string treeToString() const;

    void* allocArrayOnNumaNode(size_t size, size_t nodeId);

    void freeArrayFromNumaNode(void* m);

  private:
    class system_rt_* x_;
  };

} // end namespace scout

#endif // SCOUT_SYSTEM_H_
