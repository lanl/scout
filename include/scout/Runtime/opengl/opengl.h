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


#ifndef __SC_OPENGL_H__
#define __SC_OPENGL_H__

#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <OpenGL/glext.h>
#else // Linux/X11
#ifndef GL_GLEXT_PROTOTYPES
#define GL_GLEXT_PROTOTYPES
#endif
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glext.h>
#endif // Linux/X11 

#include "scout/Runtime/Device.h"
#include "scout/Runtime/DeviceList.h"

namespace scout
{
  class glDevice;
  
  struct oglColor {
    float red;
    float green;
    float blue;
    float alpha;
  };
  
  namespace opengl {
    
    /** ----- oglInitialize
     * Initialize the runtime's OpenGL layer.  This should be called
     * once at program startup -- and is only necessary if a program
     * contains rendering/visualization operations.
     *
     * Note: This function should be implemented/provided by each
     * supported platform (e.g. MacOS X, Linux+X11, etc.).
     */
    extern int scInitialize(DeviceList &devList);
  
    /** ----- oglFinalize
     * Shutdown the runtime's OpenGL layer.  This should be called once
     * before a program exits to clean up the allocated resources for
     * OpenG -- this is only necessary if a program contains
     * rendering/visualization operations.
     *
     * Note: This function should be implemented/provided by each
     * supported platform (e.g. MacOS X, Linux+X11, etc.).
     */
    extern void scFinalize();
  }
  

  /** ----- oglVersion
   * Return the major and minor versions of OpenGL in the associated
   * parameter.  Error conditions are represented by the major version
   * value being set to zero.
   */
  void oglVersion(int &majorVersion, int &minorVersion);

  
  /** ----- oglslVersion
   * Return the major and minor version numbers for the OpenGL Shading
   * Language.  Error conditions are represented by the major version
   * value being set to zero.
   */
  void oglslVersion(int &majorVersion, int &minorVersion);

  
  /** ----- oglReportError
   * Report an error condition within the OpenGL API. 
   */  
  extern void oglReportError(GLenum error_id, const char *file, int line_no);
  #define oglError(error_id) oglReportError(error_id, __FILE__, __LINE__)

  
  /** ----- glCheckForError
   * Check to see if the OpenGL enviornment has set an error flag.  If
   * an error condition is present it will be reported (via glReport
   * Error).  Note that if you use the macro version of this call
   * (oglErrorCheck) these calls will be removed if you are using a
   * release (optimized) build.
   */
  extern void oglCheckForError(const char *file, int line_no);
  #ifdef SC_DEBUG
  #define oglErrorCheck() glCheckForError(__FILE__, __LINE__)
  #else
  #define oglErrorCheck() /* no-op */
  #endif

  
}

#endif
