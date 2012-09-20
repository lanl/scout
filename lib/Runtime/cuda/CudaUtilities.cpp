/*
 * ###########################################################################
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
 * ########################################################################### 
 * 
 * Notes
 *
 * ##### 
 */

#include <stdio.h>
#include <cuda.h>

#include "scout/Runtime/Utilities.h"

namespace scout {
  
  /** ----- CudaDriverErrorStrings
   * A set of error strings associated with the CUDA driver API. 
   */
  static ScErrorEntry CudaDriverErrorStrings[] = {
    { "CUDA_SUCCESS",                                  ScErrorId(CUDA_SUCCESS) },
    { "CUDA_ERROR_INVALID_VALUE",                      ScErrorId(CUDA_ERROR_INVALID_VALUE) },
    { "CUDA_ERROR_OUT_OF_MEMORY",                      ScErrorId(CUDA_ERROR_OUT_OF_MEMORY) }, 
    { "CUDA_ERROR_NOT_INITIALIZED",                    ScErrorId(CUDA_ERROR_NOT_INITIALIZED) },
    { "CUDA_ERROR_DEINITIALIZED",                      ScErrorId(CUDA_ERROR_DEINITIALIZED) },
    { "CUDA_ERROR_PROFILER_DISABLED",                  ScErrorId(CUDA_ERROR_PROFILER_DISABLED) },
    { "CUDA_ERROR_PROFILER_NOT_INITIALIZED",           ScErrorId(CUDA_ERROR_PROFILER_NOT_INITIALIZED) },
    { "CUDA_ERROR_PROFILER_ALREADY_STARTED",           ScErrorId(CUDA_ERROR_PROFILER_ALREADY_STARTED) },
    { "CUDA_ERROR_PROFILER_ALREADY_STOPPED",           ScErrorId(CUDA_ERROR_PROFILER_ALREADY_STOPPED) },
    { "CUDA_ERROR_NO_DEVICE",                          ScErrorId(CUDA_ERROR_NO_DEVICE) },
    { "CUDA_ERROR_INVALID_DEVICE",                     ScErrorId(CUDA_ERROR_INVALID_DEVICE) },
    { "CUDA_ERROR_INVALID_IMAGE",                      ScErrorId(CUDA_ERROR_INVALID_IMAGE) },
    { "CUDA_ERROR_INVALID_CONTEXT",                    ScErrorId(CUDA_ERROR_INVALID_CONTEXT) },
    { "CUDA_ERROR_CONTEXT_ALREADY_CURRENT",            ScErrorId(CUDA_ERROR_CONTEXT_ALREADY_CURRENT) },
    { "CUDA_ERROR_MAP_FAILED",                         ScErrorId(CUDA_ERROR_MAP_FAILED) },
    { "CUDA_ERROR_UNMAP_FAILED",                       ScErrorId(CUDA_ERROR_UNMAP_FAILED) },
    { "CUDA_ERROR_ARRAY_IS_MAPPED",                    ScErrorId(CUDA_ERROR_ARRAY_IS_MAPPED) },
    { "CUDA_ERROR_ALREADY_MAPPED",                     ScErrorId(CUDA_ERROR_ALREADY_MAPPED) },
    { "CUDA_ERROR_NO_BINARY_FOR_GPU",                  ScErrorId(CUDA_ERROR_NO_BINARY_FOR_GPU) },
    { "CUDA_ERROR_ALREADY_ACQUIRED",                   ScErrorId(CUDA_ERROR_ALREADY_ACQUIRED) },
    { "CUDA_ERROR_NOT_MAPPED",                         ScErrorId(CUDA_ERROR_NOT_MAPPED) },
    { "CUDA_ERROR_NOT_MAPPED_AS_ARRAY",                ScErrorId(CUDA_ERROR_NOT_MAPPED_AS_ARRAY) },
    { "CUDA_ERROR_NOT_MAPPED_AS_POINTER",              ScErrorId(CUDA_ERROR_NOT_MAPPED_AS_POINTER) },
    { "CUDA_ERROR_ECC_UNCORRECTABLE",                  ScErrorId(CUDA_ERROR_ECC_UNCORRECTABLE) },
    { "CUDA_ERROR_UNSUPPORTED_LIMIT",                  ScErrorId(CUDA_ERROR_UNSUPPORTED_LIMIT) },
    { "CUDA_ERROR_CONTEXT_ALREADY_IN_USE",             ScErrorId(CUDA_ERROR_CONTEXT_ALREADY_IN_USE) },
    { "CUDA_ERROR_INVALID_SOURCE",                     ScErrorId(CUDA_ERROR_INVALID_SOURCE) },
    { "CUDA_ERROR_FILE_NOT_FOUND",                     ScErrorId(CUDA_ERROR_FILE_NOT_FOUND) },
    { "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND",     ScErrorId(CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND) },
    { "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED",          ScErrorId(CUDA_ERROR_SHARED_OBJECT_INIT_FAILED) },
    { "CUDA_ERROR_OPERATING_SYSTEM",                   ScErrorId(CUDA_ERROR_SHARED_OBJECT_INIT_FAILED) },
    { "CUDA_ERROR_INVALID_HANDLE",                     ScErrorId(CUDA_ERROR_INVALID_HANDLE) },
    { "CUDA_ERROR_NOT_FOUND",                          ScErrorId(CUDA_ERROR_NOT_FOUND) },
    { "CUDA_ERROR_NOT_READY",                          ScErrorId(CUDA_ERROR_NOT_READY) },
    { "CUDA_ERROR_LAUNCH_FAILED",                      ScErrorId(CUDA_ERROR_LAUNCH_FAILED) },
    { "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES",            ScErrorId(CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES) },
    { "CUDA_ERROR_LAUNCH_TIMEOUT",                     ScErrorId(CUDA_ERROR_LAUNCH_TIMEOUT) },
    { "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING",      ScErrorId(CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING) },
    { "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED",        ScErrorId(CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED) },    
    { "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED",            ScErrorId(CUDA_ERROR_PEER_ACCESS_NOT_ENABLED) },
    { "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE",             ScErrorId(CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE) },
    { "CUDA_ERROR_CONTEXT_IS_DESTROYED",               ScErrorId(CUDA_ERROR_CONTEXT_IS_DESTROYED) },
    { "CUDA_ERROR_ASSERT",                             ScErrorId(CUDA_ERROR_ASSERT) },
    { "CUDA_ERROR_UNKNOWN",                            999 },
    
    // The line below should always remain the last entry...    
    { "UNRECOGNIZED CUDA ERROR VALUE",                  -1 }
  };

  
  /** ----- cuReportError
   * Print the details about the given CUDA error value to standard
   * error.  This function is best called when you have already
   * determined an error condition within the code...
   */
  void cuReportError(CUresult status, const char *filename, const int line) {
    fprintf(stderr, "scout[runtime]: cuda runtime error (driver API) --\n");
    fprintf(stderr, "\terror (#%04d): \"%s\"\n", status,
            scLookupErrorString(status, CudaDriverErrorStrings));
    fprintf(stderr, "\tfile         : %s\n", filename);
    fprintf(stderr, "\tline         : %d\n", line);
    fprintf(stderr, "\tcall stack ------------\n");
    scPrintCallStack(stderr);
    fprintf(stderr, "\t-----------------------\n");    
  }

  
  // ----- cuCheckForError
  /// Check to see if the given status/return value from a CUDA driver API
  /// call represents and error condition.  If the value is an error it will be
  /// reported and true will be returned (stating an error condition does exist).
  /// Non-error conditions will return false and no information will be reported. 
  bool cuCheckForError(CUresult status, const char *filename, const int line) {
    if (status != CUDA_SUCCESS) {
      cuReportError(status, filename, line);
      return true;
    } else {
      return false;
    }
  }
  
}
