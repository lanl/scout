/*
 * ###########################################################################
 * Copyright (c) 2014, Los Alamos National Security, LLC.
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
 * This is a quick-n-dirty utility to pull information from the OpenGL
 * implementation on a system.  It can be helpful for debugging but we
 * primarily use it to help us gather information about OpenGL at
 * configuration time. 
 * 
 * Usage: gl-info
 *
 * ##### 
 */ 

#include <cassert>
#include <iostream>
#include <fstream>

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>

#ifdef __APPLE__

#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <OpenGL/glext.h>

#else /* Linux */ 

#ifndef GL_GLEXT_PROTOTYPES
#define GL_GLEXT_PROTOTYPES
#endif

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glext.h>

#endif

#define GLFW_INCLUDE_GLCOREARB
#include <GLFW/glfw3.h>

using namespace std;

// ----- opengl_error_check
//
static bool glErrorCheck(const char* file, int line_no)
{
  GLenum error = glGetError();
  if (error != GL_NO_ERROR) {
    cout << "-- opengl runtime error:\n";
    cout << "   file: " << file << endl;
    cout << "   line: " << line_no << endl;
    cout << "   mesg: '" << (const char*)gluErrorString(error) << "'\n";
    return true;
  } else {
    return false;
  }
}

// ----- initialize_opengl
//
bool initialize_opengl()
{
  if (glErrorCheck(__FILE__, __LINE__))  // return true on error... 
    return false;
  else
    return true;
}


// ----- main
//
int main(int argc, char *argv[])
{
  int   option_index;
  int   versionOnly;
  
  option long_options[] = {
    { "version", no_argument, &versionOnly, 'v'},   // --version or -v   
    { 0, 0, 0, 0 }
  };

  char c;
  while((c = getopt_long(argc, argv, "v", long_options, &option_index)) != -1) {

    switch(c) {
      
      case 'v':
        versionOnly = true;
        break;

      case 0:

        switch(versionOnly) {
          case 'v':
            versionOnly = true;
            break;
        }
        break;

      case '?':
      case 'h':
        cerr << "usage: gl-info [--version]\n";
        return 0;
        break;
        
      default:
        cerr << "usage: gl-info [--version]\n";
        return 1;
    }
  }

  GLFWwindow* window;
  if(!glfwInit()) {
    cerr << "failed to start glfw" << endl;
    return -1;
  }
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  window = glfwCreateWindow(2, 2, "", NULL, NULL);
  if(!window) {
    cerr << "failed to open window" << endl;
    return -1;
  }
  int major, minor, rev;
  glfwGetVersion(&major, &minor, &rev);
  cout << "OpenGL Version: " << major << "." << minor << "." << rev << endl;
  const char *versionStr  = glfwGetVersionString();
  cout << "OpenGL Version String: " << versionStr << endl;
  return 0;
}
