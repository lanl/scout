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

#include "scout/Runtime/Utilities.h"
#include "scout/Runtime/opengl/opengl.h"

namespace scout 
{
  
  /** -----
   * An array of OpenGL error entries we're likely to encounter in
   * Scout's runtime.
   */
  static ScErrorEntry SC_CudaDriverErrorStrings[] = {
    { "GL_NO_ERROR",                      int(GL_NO_ERROR) },
    { "GL_INVALID_ENUM",                  int(GL_INVALID_ENUM) },
    { "GL_INVALID_VALUE",                 int(GL_INVALID_VALUE) },
    { "GL_INVALID_OPERATION",             int(GL_INVALID_OPERATION) },
    { "GL_OUT_OF_MEMORY",                 int(GL_OUT_OF_MEMORY) },
    { "GL_INVALID_FRAMEBUFFER_OPERATION", int(GL_INVALID_FRAMEBUFFER_OPERATION) },
      
    // The line below should always remain the last entry...    
    { "UNRECOGNIZED OPENGL ERROR VALUE",  -1 }
  };
  
  /** ----- glReportError
   *
   */
  void glReportError(GLenum error_id, const char *file, int line) {
    fprintf(stderr, "scout[runtime]: opengl error: '%s'\n",
            (const char*)gluErrorString(error_id));
    fprintf(stderr, "\terror (#%04d): (need more here).\n", int(error_id));
    fprintf(stderr, "\tfile         : %s\n", file);
    fprintf(stderr, "\tline         : %d\n", line);
    fprintf(stderr, "\tcall stack ------------\n");
    scPrintCallStack(stderr);
    fprintf(stderr, "\t-----------------------\n");    
  }

  
  /** ----- glCheckForError
   *
   */
  void glCheckForError(const char *file, int line) {
    GLenum error_id = glGetError();
    if (error_id != GL_NO_ERROR) {
      ScErrorId eid = ScErrorId(error_id);
      glReportError(eid, file, line);
    }
  }

  
  /** ----- glVersion
   *
   */
  void glVersion(int &majorVersion, int &minorVersion) {
    
    const char *versionString = (const char*)glGetString(GL_VERSION);
    
    if (versionString == NULL) {
      majorVersion = 0;
      minorVersion = 0;
    } else {
      
      int nitems = sscanf(versionString, "%d.%d", &majorVersion, &minorVersion);
      
      if (nitems != 2) {
        majorVersion = 0;
        minorVersion = 0;
      }
    }
  }

  
  /** ----- glslVersion 
   *
   */
  void glslVersion(int &majorVersion, int &minorVersion) {
    int gl_version_major, gl_version_minor;
    glVersion(gl_version_major, gl_version_minor);

    if (gl_version_major == 0) {
      majorVersion = minorVersion = 0;
    } else if (gl_version_major == 1) {
      // 1.x only provides GLSL v1.0. 
      majorVersion = 1;
      minorVersion = 0;
    } else {
      // We must parse the version string
      const char *versionString;
      versionString = (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION);
      
      if (versionString != 0) {
        int retval = sscanf(versionString, "%d.%d",
                            &majorVersion, &minorVersion);
        if (retval != 2) 
          majorVersion = minorVersion = 0;
      }
    }
  }
}
