/*
 * -----  Scout Programming Language -----
 *
 * This file is distributed under an open source license by Los Alamos
 * National Security, LCC.  See the file License.txt (located in the
 * top level of the source distribution) for details.
 *
 *-----
 *
 * This is a quick-n-dirty utility to read in a GLSL shader and
 * compile it via the OpenGL runtime to primarily check for errors
 * before we bake a shader into the runtime library.  It is primarily
 * used as part of our build system...
 * 
 * Usage: glsl-cc --[fragment|geometry|vertex] file.glsl
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
#include <OpenGL/glext.h>

#else /* Linux */ 

#ifndef GL_GLEXT_PROTOTYPES
#define GL_GLEXT_PROTOTYPES
#endif

#include <GL/gl.h>
#include <GL/glext.h>

#endif

#include <GLFW/glfw3.h>

using namespace std;

const int FRAGMENT_FLAG = 1;
const int GEOMETRY_FLAG = 2;
const int VERTEX_FLAG   = 3;

// ----- read_glsl
//
static char* read_glsl(const char* filename)
{
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
    return 0;
  }
}


// ----- opengl_error_check
//
static bool glErrorCheck(const char* file, int line_no)
{
  GLenum error = glGetError();
  if (error != GL_NO_ERROR) {
    cout << "-- opengl runtime error:\n";
    cout << "   file: " << file << endl;
    cout << "   line: " << line_no << endl;
    cout << "   error: '" << error << "'\n";
    return true;
  } else {
    return false;
  }
}


// ----- initialize_opengl
//
bool initialize_opengl(int shader_type)
{
  glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
  
  if (glErrorCheck(__FILE__, __LINE__))  // return true on error... 
    return false;
  else
    return true;
}


// ----- compile_shader
//
bool compile_shader(const char* src_str, int shader_kind)
{

  GLenum gl_shader_type;
  
  switch(shader_kind) {

    case FRAGMENT_FLAG:
      gl_shader_type = GL_FRAGMENT_SHADER;
      break;

    case GEOMETRY_FLAG:
      gl_shader_type = GL_GEOMETRY_SHADER_EXT;
      break;

    case VERTEX_FLAG:
      gl_shader_type = GL_VERTEX_SHADER;
      break;

    default:
      cout << "-- unrecognized shader type in switch (this should not happen!)\n";
      return false;
  }

  GLuint   prog_id = glCreateProgram();
  if (! glIsProgram(prog_id)) {
    cout << "error 1\n";
    return false;
  }

  GLuint shader_id = glCreateShader(gl_shader_type);
  if (! glIsShader(shader_id)) {
    glDeleteProgram(prog_id);
    cout << "error 2\n";
    return false;
  }

  glShaderSource(shader_id, 1, &src_str, NULL);

  glCompileShader(shader_id);
  GLint was_compiled;
  glGetShaderiv(shader_id, GL_COMPILE_STATUS, &was_compiled);
  if (! was_compiled) {
    GLint   log_length;
    GLchar* log;

    glGetShaderiv(shader_id, GL_INFO_LOG_LENGTH, &log_length);
    log = new GLchar[log_length+1];
    glGetShaderInfoLog(shader_id, log_length, &log_length, log);
    cout << log << endl;
    delete []log;
  }

  glDeleteShader(shader_id);
  glDeleteProgram(prog_id);

  return(was_compiled == GL_TRUE);
}


// ----- main
//
int main(int argc, char *argv[])
{
  char *input_file  = 0;
  int   shader_type = 0;
  int   c = 0;
  
  static struct option long_options[] = {
    { "fragment", no_argument, 0, 'f'},   // --fragment or -f   
    { "geometry", no_argument, 0, 'g'},   // --geometry or -g
    { "vertex",   no_argument, 0, 'v'},   // --vertex or -v
    { 0, 0, 0, 0 }
  };

  while((c = getopt_long(argc, argv, "fgvh?", long_options, 0)) != -1) {
  
    switch(c) {
      case 'f':
        shader_type = FRAGMENT_FLAG;
        break;

      case 'g':
        shader_type = GEOMETRY_FLAG;
        break;
        
      case 'v':
        shader_type = VERTEX_FLAG;
        break;

      case '?':
      case 'h':
        cerr << "usage: glsl-cc [--fragment | --geometry | --vertex ] glsl-file\n";
        cerr << "   must specify one shader type [fragment|geometry|vertex]\n";        
        cerr << "   glsl-file : input GLSL source file\n";
        return 0;
        break;
        
      default:
        cerr << "usage: glsl-cc [--fragment | --geometry | --vertex ] glsl-file\n";
        cerr << "   must specify only one of fragment, geometry or vertex for shader type.\n";        
        cerr << "   glsl-file  : input GLSL source file\n";        
        return 1;
    }
  }

  if (shader_type == 0) {
    cout << "shader_type is zero...  this should not happen.\n";
    return 1;
  }

  if (optind == argc) {
    cerr << "usage: glsl-cc [--fragment | --geometry | --vertex ] glsl-file\n";
    cerr << "   must specify only one of fragment, geometry or vertex for shader type.\n";        
    cerr << "   glsl-file  : input GLSL source file\n";        
    return 1;
  }
  
  input_file = new char[strlen(argv[optind])+1];
  strcpy(input_file, argv[optind]);
  
  char *src_str = read_glsl(input_file);
  if (src_str == 0) {
    cerr << "-- error reading input file: " << input_file << endl;
    delete []input_file;
    return 1;
  }

  GLFWwindow* window;
  if(!glfwInit()) {
    cerr << "failed to start glfw" << endl;
    return -1;
  }
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  window = glfwCreateWindow(2, 2, "", NULL, NULL);
  if(!window) {
    cerr << "failed to open window" << endl;
    return -1;
  }
  glfwMakeContextCurrent(window);

  cout << "shading lang. version" << glGetString(GL_SHADING_LANGUAGE_VERSION) << "\n";

  if (initialize_opengl(shader_type) == false) {
    cerr << "-- error initializing opengl\n";
    delete []input_file;
    delete []src_str;
    return 1;
  }

  int retval = 0;

  if (compile_shader(src_str, shader_type) == false) {
    cout << "glsl-cc: failed to compile shader: " << input_file << "\n";
    retval = 1;
  }
  
  delete []input_file;
  delete []src_str;
  return retval;
}
