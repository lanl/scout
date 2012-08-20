/*
 * -----  Scout Programming Language -----
 *
 * This file is distributed under an open source license by Los Alamos
 * National Security, LCC.  See the file License.txt (located in the
 * top level of the source distribution) for details.
 * 
 * -----
 * 
 */

#include "scout/Runtime/renderall_base.h"

using namespace std;
using namespace scout;

renderall_base_rt* __sc_renderall = 0;

renderall_base_rt::renderall_base_rt(size_t width,
				     size_t height,
				     size_t depth)
  : width_(width),
    height_(height),
    depth_(depth){
    
}

renderall_base_rt::~renderall_base_rt(){
    
}

void __sc_begin_renderall(){
  __sc_renderall->begin();
}

void __sc_end_renderall(){
  __sc_renderall->end();
}

void __sc_delete_renderall(){
  if (__sc_renderall != NULL) {
    delete __sc_renderall;
    __sc_renderall = NULL;
  }
}



