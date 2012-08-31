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

#include <stdio.h>
#include <stdlib.h>
#include <execinfo.h>
#include <dlfcn.h>
#include <cxxabi.h>

#include "scout/Runtime/Utilities.h"

namespace scout {


  /** ----- scLookupErrorString
   * Given an error/return value look up the assocaited error string.
   */
  const char *scLookupErrorString(ScErrorId err_value,
                                  ScErrorEntry errorTable[]) {
    unsigned int idx = 0;
    while(errorTable[idx].errorId != err_value &&
          errorTable[idx].errorId != -1) {
      ++idx;
    }

    return errorTable[idx].errorString;
  }

  
  /** Dump the current call stack to the given output stream.
   *
   * This can be a very helpful function for debugging the runtime
   * details when debugging or reporting errors.  We strongly
   * encourage its use when reporting errors in the runtime.  It
   * should provide support not only for seeing the stack but also
   * demangles C++ as part of the process.
   *
   * Note that the depth of the trace is limited to the number of
   * entries in SC_MAX_TRACE_DEPTH.
   */
  void scPrintCallStack(FILE *fp) {

    static const unsigned short SC_MAX_TRACE_DEPTH=32;

    using namespace abi;
    
    int         status;
    const char *symbol_name;
    char       *demangled_name;    
    Dl_info     dlinfo;

    // Start out grabbing the backtrace -- 
    void *trace[SC_MAX_TRACE_DEPTH];
    int trace_size = backtrace(trace, SC_MAX_TRACE_DEPTH);

    // Loop through the entires in the trace -- note that we skip the
    // first entry that is the call to this function...
    for(int cur_trace = 1; cur_trace < trace_size; ++cur_trace) {

      // If we can't find the image information for the given address
      // skip it and keep going...
      if (! dladdr(trace[cur_trace], &dlinfo)) {
        fprintf(fp, "\t--- loader unable to find image, skipping entry %02d...\n", cur_trace);
        continue;
      }

      // Grab the symbol name... 
      symbol_name = dlinfo.dli_sname;

      // Demangle the name if necessary... 
      demangled_name = __cxa_demangle(symbol_name, NULL, 0, &status);
      if (status == 0 && demangled_name != 0) {
        symbol_name = demangled_name; // Successfully demangled symbol.
      }
      
      fprintf(fp, "\t%02d: %-16s -- '%-64s'\n",
              cur_trace,
              symbol_name,
              dlinfo.dli_fname);

      if (demangled_name)  // Don't forget to clean up! 
        free(demangled_name);
    }
  }

  
  /** A series of error strings and assocaited flag values.
   *
   * We use this information to help drive the error and warning
   * message reporting infrastructure within the runtime.  This
   * structure is partially adopted from NVIDIA's CUDA examples code
   * base...
   */
  static ScErrorEntry SC_ErrorStrings[] = {
    { "SCOUT_NO_ERROR",              int(SCOUT_NO_ERROR)},
    { "SCOUT_NO_DEVICES",            int(SCOUT_NO_DEVICES)},
    
    // The line below should always remain the last entry...
    { "SCOUT_UNKNOWN_ERROR",         int(SCOUT_UNKNOWN_ERROR) }
  };


  /** ----- scReportError
   * Report an error based on the given error value.  Note that
   * this routine does not check for an error state -- it assumes
   * you have done so within your own code.
   */
  void scReportError(ScErrorId status, const char *filename, const int line) {
    ScErrorId err = ScErrorId(status);
    fprintf(stderr, "scout[runtime]: general runtime error --\n");
    fprintf(stderr, "\terror (#%0d): \"%s\"\n", err, scLookupErrorString(err, SC_ErrorStrings));
    fprintf(stderr, "\tfile        : %s\n", filename);
    fprintf(stderr, "\tline        : %d\n", line);
    fprintf(stderr, "\tcall stack ------------\n");
    scPrintCallStack(stderr);
    fprintf(stderr, "\t-----------------------\n");
  }

  
  /** ----- scCheckForError
   * Check to see if the given error ID represents an error.  If so, report
   * the associated error.
   */
  void scCheckForError(ScErrorId errorID, const char *filename, int line) {
    if (errorID != SCOUT_NO_ERROR) {
      scReportError(errorID, filename, line);
    }
  }
  
}
