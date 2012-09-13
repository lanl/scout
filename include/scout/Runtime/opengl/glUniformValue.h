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

#ifndef SCOUT_GL_UNIFORM_VALUE_H_
#define SCOUT_GL_UNIFORM_VALUE_H_

#include <iostream>

#include "scout/Runtime/opengl/opengl.h"
#include "scout/Runtime/vec_types.h"


namespace scout
{
  
  // ..... glUniformValue
  //  
  class glUniformValue {
    
   public:
    glUniformValue()
    { m_id = 0; }
    
    glUniformValue(GLint id)
    { m_id = id; }
    
    virtual ~glUniformValue()
    { /* no-op */ }

    GLint id() const
    { return m_id; }

    virtual void bind() = 0;

   private:
    GLint    m_id;
  };


  
  inline void glUniform(GLint location, const int* iptr)
  {
    glUniform1iv(location, 1, iptr);
    oglErrorCheck();        
  }

  
  inline void glUniform(GLint location, const int4* iptr)
  {
    glUniform4iv(location, 4, &(iptr->components[0]));
    oglErrorCheck();    
  }
  
  inline void glUniform(GLint location, const float* fptr)
  {
    glUniform1fv(location, 1, fptr);
    oglErrorCheck();
  }

  inline void glUniform(GLint location, const float4* fptr)
  {
    glUniform4fv(location, 4, &(fptr->components[0]));
    oglErrorCheck();    
  }

  // ..... glTypedUniformValue
  //
  template <typename ElementType>
  class glTypedUniformValue : public glUniformValue {

   public:
    glTypedUniformValue(GLint id, const ElementType* value)
        : glUniformValue(id)
    { m_value = value; }
    
    glTypedUniformValue(GLint id, ElementType& value)
        : glUniformValue(id)
    { m_value = &value; }

    ~glTypedUniformValue()
    { /* no-op */  }

    void bind() 
    {
      glUniform(id(), m_value);
      oglErrorCheck();
    }
    
   private:
    const ElementType*      m_value;
  };

}

#endif


