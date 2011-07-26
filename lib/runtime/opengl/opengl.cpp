/*
 * -----  Scout Programming Language -----
 *
 * This file is distributed under an open source license by Los Alamos
 * National Security, LCC.  See the file License.txt (located in the
 * top level of the source distribution) for details.
 * 
 *-----
 *
 */

#include <iostream>
#include <stdlib.h>

#include "runtime/opengl/opengl.h"

using namespace std;
using namespace scout;


// ----- glErrorCheck
//
void glErrorCheck(const std::string& file, int line_no)
{
  GLenum error = glGetError();
  if (error != GL_NO_ERROR) {
    cerr << "OpenGL runtime error:\n";
    cerr << "   file: " << file << endl;
    cerr << "   line: " << line_no << endl;
    cerr << "   mesg: '" << (const char*)gluErrorString(error) << "'\n";

    // fall flat on our face for now... 
    abort(); 
  }
}


    
    


