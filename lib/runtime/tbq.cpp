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

#include "runtime/tbq.h"

using namespace std;
using namespace scout;

void performIteration(void (^block)(void*,int*,int*,int*,tbq_params_rt),
		      int xStart, int xEnd,
		      int yStart, int yEnd,
		      int zStart, int zEnd){

}
