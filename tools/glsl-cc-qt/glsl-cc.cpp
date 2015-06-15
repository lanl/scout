/*
 * ###########################################################################
 *
 * Copyright (c) 2014, Los Alamos National Security, LLC.
 * All rights reserved.
 *
 *  Copyright 2013. Los Alamos National Security, LLC. This software was
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
 *
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
#include <QApplication>
#include <QOffscreenSurface>
#include <QOpenGLContext>

// ----- readShaderFile
// Read the shader source code from the given file and return the
// program as a text string.  If the file can't be opened and/or read
// successfully the call will return a null pointer.  The returned
// string is allocated on the heap and should be freed/deleted by the
// caller.
static char* readShaderFile(const char* filename) {
  using namespace std;  
  assert(filename != 0);
  ifstream is;
  is.open(filename, ios::binary);
  if (is.is_open()) {
    is.seekg(0, ios::end);
    int length = is.tellg();
    is.seekg(0, ios::beg);

    char* src = new char[length+1];
    is.read(src, length-1);
    src[length-1] = '\0';
    is.close();
    return src;
  } else {
    cerr << "glsl-cc: unable to open source file '" << filename << "'.\n";
    return 0;
  }
}


// ----- glErrorCheck                                                                                      
// Check to see if the OpenGL runtime has an error state.  If so, we
// will report the line number we were called from and as much detail
// as we can about the error. 
static bool glErrorCheck(const char* file, int line_no) {
  using namespace std;
  GLenum error = glGetError();
  if (error != GL_NO_ERROR) {
    cerr << "glsl-cc: encountered an opengl runtime error...\n";
    cerr << "   file: " << file << endl;
    cerr << "   line: " << line_no << endl;
    cerr << "   mesg: '" << (const char*)gluErrorString(error) << "'\n";
    return true;
  } else {
    return false;
  }
}


// ----- initializeOpenGL
// Intiailize the OpenGL enviornment and check to make sure we have all the
// features we need to deal with shaders.
bool initializeOpenGL(GLenum shader_kind) {
  using namespace std;

  // Check to see if this system has a shader compiler support.  All
  // desktop systems are suppose to have this support but...
  //
  // On the Mac this doesn't work -- it returns true but also actually
  // sets an OpenGL runtime error state ('invalid enumerant').  For
  // now we don't bail on a error state as things actually appear to
  // work correctly beyond this snafu... 
  GLboolean hasCompiler = GL_TRUE;
  glGetBooleanv(GL_SHADER_COMPILER, &hasCompiler);
  glErrorCheck(__FILE__, __LINE__);
  if (hasCompiler == GL_FALSE) {
    cerr << "glsl-cc: system's OpenGL version does not support shaders.\n";
    return false;
  }
  return true;
}


// ----- compileShader                                                                                          
//                                                                                                               
bool compileShader(const char* src_str, GLenum shader_kind)
{
  using namespace std;
  assert(src_str != 0);
  assert(shader_kind == GL_FRAGMENT_SHADER ||
         shader_kind == GL_VERTEX_SHADER   ||
         shader_kind == GL_GEOMETRY_SHADER);
  
  GLuint   prog_id = glCreateProgram();
  if (! glIsProgram(prog_id)) {
    cerr << "glsl-cc: unable to create a shader program!\n";
    glErrorCheck(__FILE__, __LINE__);
    return false;
  }

  glErrorCheck(__FILE__, __LINE__);  
  GLuint shader_id = glCreateShader(shader_kind);
  glErrorCheck(__FILE__, __LINE__);
  
  if (! glIsShader(shader_id)) {
    glDeleteProgram(prog_id);
    cerr << "glsl-cc: unabel to create shader instance!\n";
    glErrorCheck(__FILE__, __LINE__);
    return false;
  }

  
  // Set the shader's source. 
  glShaderSource(shader_id, 1, &src_str, NULL);
  glErrorCheck(__FILE__, __LINE__);

  glCompileShader(shader_id);
  GLint was_compiled;
  glGetShaderiv(shader_id, GL_COMPILE_STATUS, &was_compiled);
  if (! was_compiled) {
    GLint   log_length;
    GLchar* log;

    glGetShaderiv(shader_id, GL_INFO_LOG_LENGTH, &log_length);
    log = new GLchar[log_length+1];
    glGetShaderInfoLog(shader_id, log_length, &log_length, log);
    cerr << log << endl;
    delete []log;
  }

  glDeleteShader(shader_id);
  glDeleteProgram(prog_id);

  return(was_compiled == GL_TRUE);
}

void print_usage() {
  using namespace std;
  cout << "usage: glsl-cc [--fragment | --geometry | --vertex ] file\n";
  cout << "   must specify one shader type [fragment|geometry|vertex]\n";
  cout << "   file : input OpenGL shading language source file.\n"; 
}


int main(int argc, char *argv[]) {
  using namespace std;
  
  char   *input_file  = 0;
  int    shader_type = 0;
  GLenum shader_kind = GL_NONE;

  // TODO: Add tess and compute shaders. 
  static struct option long_options[] = {
    { "fragment", no_argument, 0, 'f' },     // --fragment or -f
    { "geometry", no_argument, 0, 'g' },     // --geometry or -g
    { "vertex",   no_argument, 0, 'v' },     // --vertex or -v
    { 0, 0, 0, 0 }
  };

  int c = 0;
  while((c = getopt_long(argc, argv, "fgvh?", long_options, 0)) != -1) {

    switch(c) {
      
      case 'f':
        shader_kind = GL_FRAGMENT_SHADER;
        break;

      case 'g':
        shader_kind = GL_GEOMETRY_SHADER;
        break;

      case 'v':
        shader_kind = GL_VERTEX_SHADER;
        break;

      case '?':
      case 'h':
        print_usage();
        return 0;
        break;

      default:
        cout << "unrecognized/missing options...\n";
        print_usage();
        return 1;
    }
  }
  
  if (optind == argc) {
    print_usage();
    return 1;
  }

  input_file = new char[strlen(argv[optind])+1];
  strcpy(input_file, argv[optind]);

  char *source_str = readShaderFile(input_file);
  if (source_str == 0) {
    cerr << "\texiting...'\n";
    delete []input_file;
    return 1;
  }

  QApplication app(argc, argv);
  QSurfaceFormat format;
  format.setVersion(4, 1);
  format.setRenderableType(QSurfaceFormat::OpenGL);
  
  QOffscreenSurface surface;
  surface.setFormat(format);
  surface.create();

  QOpenGLContext *context = new QOpenGLContext;
  context->create();
  context->makeCurrent(&surface);

  int retval = 0;

  if (surface.isValid() == false) {
    cerr << "glsl-cc: error creating a valid OpenGL rendering surface!\n";
    retval = 1;
  } else {
    if (initializeOpenGL(shader_type) == false) {
      cerr << "glsl-cc: error initializing a suitable OpenGL enviornment.\n";
      retval = 1;
    } else {
      if (compileShader(source_str, shader_kind) == false) {
        cout << "glsl-cc: compilation error, exiting...\n";
        retval = 1;
      } else {
        cerr << "glsl-cc: success.\n";
        retval = 0;
      }
    }
  }
  
  delete []input_file;
  delete []source_str;
  return 0; // app.exec();
}
