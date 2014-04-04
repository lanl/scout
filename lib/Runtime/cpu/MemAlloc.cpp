/*
 *  
 *###########################################################################
 * Copyright (c) 2010, Los Alamos National Security, LLC.
 * All rights reserved.
 * 
 *  Copyright 2010. Los Alamos National Security, LLC. This software was
 *  produced under U.S. Government contract DE-AC52-06NA25396 for Los
 *  Alamos National Laboratory (LANL), which is operated by Los Alamos
 *  National Security, LLC for the U.S. Department of Energy. The
 *  U.S. Government has rights to use, reproduce, and distribute this
 *  software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY,
 *  LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY
 *  FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
 *  derivative works, such modified software should be clearly marked,
 *  so as not to confuse it with the version available from LANL.
 *
 *  Additionally, redistribution and use in source and binary forms,
 *  with or without modification, are permitted provided that the
 *  following conditions are met:
 *
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 * 
 *    * Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following
 *      disclaimer in the documentation and/or other materials provided 
 *      with the distribution.
 *
 *    * Neither the name of Los Alamos National Security, LLC, Los
 *      Alamos National Laboratory, LANL, the U.S. Government, nor the
 *      names of its contributors may be used to endorse or promote
 *      products derived from this software without specific prior
 *      written permission.
 * 
 *  THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
 *  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 *  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 *  OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 *  SUCH DAMAGE.
 */ 

#include <stdio.h>
#include <cassert>
#include <stdlib.h>
#include <stdint.h>

#include "scout/Runtime/cpu/MemAlloc.h"

namespace scout 
{

  #define SC_PREFERRED_ALIGNMENT 64

  // make sure this size is a multiple of SC_PREFERRED_ALIGNMENT
  struct sc_array_header{
#if SC_PREFERRED_ALIGNMENT == 64
    uint64_t size;
#else
    uint32_t size;
#endif
  };  


  // malloc wrapper
  extern "C"
  void * __scrt_malloc(size_t size) {
    assert(size != 0);
    void *a_ptr;
    a_ptr = malloc(size);
    return a_ptr; 
  }

  // ----- __sc_aligned_malloc
  // 
  extern "C"
  void* __scrt_aligned_malloc(size_t size)
  {
    assert(size != 0);

    void* a_ptr;
    int failure = posix_memalign(&a_ptr, SC_PREFERRED_ALIGNMENT, size);
    if (! failure)
      return a_ptr;
    else
      return 0;
  }

  // ----- __sc_aligned_array_malloc
  // 
  extern "C"
  void* __scrt_aligned_array_malloc(size_t size){
    void* a_ptr;
    int failure = posix_memalign(&a_ptr, SC_PREFERRED_ALIGNMENT,
        sizeof(sc_array_header) + size);

    if(failure){
      return 0;
    }

    ((sc_array_header*)a_ptr)->size = size;
 
    return (char*)a_ptr + sizeof(sc_array_header);
  }

  // ----- __sc_aligned_array_size
  // 
  extern "C"
  size_t __scrt_aligned_array_size(void* a_ptr){
    sc_array_header* h  = 
      (sc_array_header*)((char*)a_ptr - sizeof(sc_array_header));
    return h->size;
  }   
 
  // ----- __sc_realloc
  //
  extern "C"
  void* __scrt_aligned_realloc(void* ptr, size_t size)
  {
    assert(ptr != 0);
    assert(size != 0);

    return realloc(ptr, size);
  }
  
    
  // ----- __sc_free
  //
  extern "C"
  void __scrt_free(void* a_ptr)
  {
    free(a_ptr);
  }

  extern "C"
  void __scrt_aligned_free(void* a_ptr)
  {
    free(a_ptr);
  }

}
  
