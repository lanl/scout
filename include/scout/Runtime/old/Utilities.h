/*
 * ###########################################################################
 * Copyrigh (c) 2010, Los Alamos National Security, LLC.
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
#ifndef __SC_UTILITIES_H__
#define __SC_UTILITIES_H__

#include <stdio.h>

namespace scout {

  /**
   * A convient shorthand for error id values.
   */
  typedef int ScErrorId;

  /**
   * We use the following structure as part of our error reporting
   * infrastructure.  It holds a string representation and a
   * corresponding error value (an integer).  The runtime uses
   * various such tables to report errors.
   */
  struct ScErrorEntry {
    char const *errorString;
    ScErrorId   errorId;    
  };

  
  /** ----- scLookupErrorString
   * Given an error/return value look up the assocaited error string.
   */
  const char *scLookupErrorString(ScErrorId err_value,
                                    ScErrorEntry entries[]); 

  
  /** ----- scErrorType
   * A collection of error flags assocaited with various error states
   * within the runtime.
   */
  enum ScErrorType {
    SCOUT_NO_ERROR      = 0,
    SCOUT_INIT_ERROR    = 1,
    SCOUT_NO_DEVICES    = 100,
    SCOUT_UNKNOWN_ERROR = -1,
  };

  
  /** ----- scReportError
   * Report a scout-centric runtime error.  Also note the scError() macro that
   * will add a filename and line number expansion based on call point. 
   */
  extern void scReportError(ScErrorId error, const char *filename, int line);
  #define scError(error)  scReportError(error, __FILE__, __LINE__)


  /** ----- scCheckForError
   * Check a given error value (scErrorType) for an error condition.  If the
   * value represents an error report it to the assocaited error stream.
   */
  extern void scCheckForError(ScErrorId error, const char *filename, int line);
  
  // Conditionally support full error checking based on if we are building a
  // debug or release version.  This might not be that helpful of an idea if
  // runtime errors tend to occur only in optimized cases.
  #ifdef SC_DEBUG 
  #define scErrorCheck(error_id) scCheckForError(error_id, __FILE__, __LINE__)
  #else 
  #define scErrorCheck(error_id) /* no-op */
  #endif 

  
  /** ----- scPrintCallStack
   * Dump the current call stack (up to the calling function) to the given output
   * stream.
   */
  extern void scPrintCallStack(FILE *fp);
  
}

#endif


