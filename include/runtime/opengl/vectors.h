/*
 *           -----  The Scout Programming Language -----
 *
 * This file is distributed under an open source license by Los Alamos
 * National Security, LCC.  See the file License.txt (located in the
 * top level of the source distribution) for details.
 * 
 *-----
 *
 * $Revision$
 * $Date$
 * $Author$
 *
 *-----
 * 
 */

#ifndef SCOUT_GL_VECTORS_H_
#define SCOUT_GL_VECTORS_H_

#include <math.h>
#include <stdint.h>
#include <cassert>

#include "runtime/base_types.h"

namespace scout
{
  typedef unsigned int     uint;
  typedef unsigned int*    uintp;

  template <typename T> class glvec2;
  template <typename T> class glvec3;
  template <typename T> class glvec4;    
    
  // ---- glvec2<T>
  // 
  template <typename T>
  class glvec2 {

   public:    

    typedef T          value_type;
    typedef uint16_t   dim_type;

    glvec2()
    {
      for(dim_type i = 0; i < size(); ++i)
        _values[i] = T(0.0);
    }    

    glvec2(const T &v)
    {
      for(dim_type i = 0; i < size(); ++i)
        _values[i] = v;
    }
    
    glvec2(const T &u, const T &v)
    {
      x = u;
      y = v;
    }

    glvec2(const T *v)
    {
      for(ushort i = 0; i < size(); ++i) 
        _values[i] = v[i];
    }

    explicit glvec2(const glvec3<T> &u)
    {
      for(ushort i = 0; i < size(); ++i)
        _values[i] = u[i];
    }

    explicit glvec2(const glvec4<T> &u)
    {
      for(ushort i = 0; i < size(); ++i)
        _values[i] = u[i];
    }

    ushort size() const 
    { return 2; }

    const T* address() const 
    { return _values; }

    glvec2<T> setValue(const T* rhs) 
    {
      for(ushort i = 0; i < size(); ++i)
        _values[i] = rhs[i];
      return *this;
    }
    
    T& operator[](int i)
    {
      assert(i < size());
      return _values[i];
    }

    const T& operator[](int i) const
    {
      assert(i < size());
      return _values[i];
    }

    operator T* () 
    { return _values; }

    operator const T* () const 
    { return _values; }

    friend glvec2<T> & operator *= (glvec2<T> &lhs, T d)
    {
      for(int i = 0; i < lhs.size(); i++)
        lhs._values[i] *= d;
      return lhs;
    }

    friend glvec2<T> & operator *= (glvec2<T> &lhs, const glvec2<T> &rhs) {
      for(int i = 0; i < lhs.size(); i++)
        lhs._values[i] *= rhs._values[i];
      return lhs;
    }

    friend glvec2<T> & operator /= (glvec2<T> &lhs, T d) {
      if(d == 0)
        return lhs;
      for(int i = 0; i < lhs.size(); i++)
        lhs._values[i] /= d;
      return lhs;
    }

    friend glvec2<T> & operator /= (glvec2<T> &lhs, const glvec2<T> & rhs)
    {
      for(int i = 0; i < lhs.size(); i++)
        lhs._values[i] /= rhs._values[i];
      return lhs;
    }

    friend glvec2<T> & operator += (glvec2<T> &lhs, const glvec2<T> & rhs)
    {
      for(int i = 0; i < lhs.size(); i++)
        lhs._values[i] += rhs._values[i];
      return lhs;
    }

    friend glvec2<T> & operator -= (glvec2<T> &lhs, const glvec2<T> & rhs)
    {
      for(int i = 0; i < lhs.size(); i++)
        lhs._values[i] -= rhs._values[i];
      return lhs;
    }

    friend glvec2<T> operator - (const glvec2<T> &rhs)
    {
      glvec2<T> rv;
      for(int i = 0; i < rhs.size(); i++)
        rv._values[i] = -rhs._values[i];
      return rv;
    }

    friend glvec2<T> operator + (const glvec2<T> & lhs, const glvec2<T> & rhs)
    {
      glvec2<T> rt(lhs);
      return rt += rhs;
    }

    friend glvec2<T> operator - (const glvec2<T> & lhs, const glvec2<T> & rhs)
    {
      glvec2<T> rt(lhs);
      return rt -= rhs;
    }

    friend glvec2<T> operator * (const glvec2<T> & lhs, T rhs)
    {
      glvec2<T> rt(lhs);
      return rt *= rhs;
    }

    friend glvec2<T> operator * (T lhs, const glvec2<T> & rhs)
    {
      glvec2<T> rt(lhs);
      return rt *= rhs;
    }

    friend glvec2<T> operator * (const glvec2<T> & lhs, const glvec2<T> & rhs)
    {
      glvec2<T> rt(lhs);
      return rt *= rhs;
    }

    friend glvec2<T> operator / (const glvec2<T> & lhs, T rhs)
    {
      glvec2<T> rt(lhs);
      return rt /= rhs;
    }

    friend glvec2<T> operator / (const glvec2<T> & lhs, const glvec2<T> & rhs)
    {
      glvec2<T> rt(lhs);
      return rt /= rhs;
    }

    friend bool operator == (const glvec2<T> &lhs, const glvec2<T> &rhs )
    {
      bool r = true;
      for (int i = 0; i < lhs.size(); i++)
        r &= lhs._values[i] == rhs._values[i];
      return r;
    }

    friend bool operator != ( const glvec2<T> &lhs, const glvec2<T> &rhs )
    {
      bool r = true;
      for (int i = 0; i < lhs.size(); i++)
        r &= lhs._values[i] != rhs._values[i];
      return r;
    }
   
    union {
      
      struct {
        T   x, y;
      };
      
      T _values[2];
    };
    
  };

  typedef glvec2<char>   glchar2;
  typedef glvec2<uchar>  gluchar2;
  typedef glvec2<short>  glshort2;
  typedef glvec2<ushort> glushort2;  
  typedef glvec2<int>    glint2;
  typedef glvec2<uint>   gluint2;  
  typedef glvec2<long>   gllong2;
  typedef glvec2<ulong>  glulong2;  
  typedef glvec2<float>  glfloat2;
  typedef glvec2<double> gldouble2;

  // ---- glvec3<T>
  // 
  template <typename T>
  class glvec3 {

   public:    
    
    typedef T value_type;
    typedef unsigned short ushort;

    glvec3()
    {
      for(ushort i = 0; i < size(); ++i)
        _values[i] = T(0.0);
    }    

    glvec3(const T &v)
    {
      for(ushort i = 0; i < size(); ++i)
        _values[i] = v;
    }
    
    glvec3(const T &u, const T &v, const T &w)
    {
      x = u;
      y = v;
      z = w;
    }

    glvec3(const T *v)
    {
      for(ushort i = 0; i < size(); ++i) 
        _values[i] = v[i];
    }

    explicit glvec3(const glvec3<T> &u)
    {
      for(ushort i = 0; i < size(); ++i)
        _values[i] = u[i];
    }

    explicit glvec3(const glvec4<T> &u)
    {
      for(ushort i = 0; i < size(); ++i)
        _values[i] = u[i];
    }

    ushort size() const 
    { return 3; }

    const T* address() const 
    { return _values; }

    glvec3<T> setValue(const T* rhs) 
    {
      for(ushort i = 0; i < size(); ++i)
        _values[i] = rhs[i];
      return *this;
    }
    
    T& operator[](int i)
    {
      assert(i < size());
      return _values[i];
    }

    const T& operator[](int i) const
    {
      assert(i < size());
      return _values[i];
    }

    operator T* () 
    { return _values; }

    operator const T* () const 
    { return _values; }

    friend glvec3<T> & operator *= (glvec3<T> &lhs, T d)
    {
      for(int i = 0; i < lhs.size(); i++)
        lhs._values[i] *= d;
      return lhs;
    }

    friend glvec3<T> & operator *= (glvec3<T> &lhs, const glvec3<T> &rhs) {
      for(int i = 0; i < lhs.size(); i++)
        lhs._values[i] *= rhs._values[i];
      return lhs;
    }

    friend glvec3<T> & operator /= (glvec3<T> &lhs, T d) {
      if(d == 0)
        return lhs;
      for(int i = 0; i < lhs.size(); i++)
        lhs._values[i] /= d;
      return lhs;
    }

    friend glvec3<T> & operator /= (glvec3<T> &lhs, const glvec3<T> & rhs)
    {
      for(int i = 0; i < lhs.size(); i++)
        lhs._values[i] /= rhs._values[i];
      return lhs;
    }

    friend glvec3<T> & operator += (glvec3<T> &lhs, const glvec3<T> & rhs)
    {
      for(int i = 0; i < lhs.size(); i++)
        lhs._values[i] += rhs._values[i];
      return lhs;
    }

    friend glvec3<T> & operator -= (glvec3<T> &lhs, const glvec3<T> & rhs)
    {
      for(int i = 0; i < lhs.size(); i++)
        lhs._values[i] -= rhs._values[i];
      return lhs;
    }

    friend glvec3<T> operator - (const glvec3<T> &rhs)
    {
      glvec3<T> rv;
      for(int i = 0; i < rhs.size(); i++)
        rv._values[i] = -rhs._values[i];
      return rv;
    }

    friend glvec3<T> operator + (const glvec3<T> & lhs, const glvec3<T> & rhs)
    {
      glvec3<T> rt(lhs);
      return rt += rhs;
    }

    friend glvec3<T> operator - (const glvec3<T> & lhs, const glvec3<T> & rhs)
    {
      glvec3<T> rt;
      rt.x = lhs.x - rhs.x;
      rt.y = lhs.y - rhs.y;
      rt.z = lhs.z - rhs.z;            
      return rt;
    }

    friend glvec3<T> operator * (const glvec3<T> & lhs, T rhs)
    {
      glvec3<T> rt(lhs);
      return rt *= rhs;
    }

    friend glvec3<T> operator * (T lhs, const glvec3<T> & rhs)
    {
      glvec3<T> rt(lhs);
      return rt *= rhs;
    }

    friend glvec3<T> operator * (const glvec3<T> & lhs, const glvec3<T> & rhs)
    {
      glvec3<T> rt(lhs);
      return rt *= rhs;
    }

    friend glvec3<T> operator / (const glvec3<T> & lhs, T rhs)
    {
      glvec3<T> rt(lhs);
      return rt /= rhs;
    }

    friend glvec3<T> operator / (const glvec3<T> & lhs, const glvec3<T> & rhs)
    {
      glvec3<T> rt(lhs);
      return rt /= rhs;
    }

    friend bool operator == (const glvec3<T> &lhs, const glvec3<T> &rhs )
    {
      bool r = true;
      for (int i = 0; i < lhs.size(); i++)
        r &= lhs._values[i] == rhs._values[i];
      return r;
    }

    friend bool operator != ( const glvec3<T> &lhs, const glvec3<T> &rhs )
    {
      bool r = true;
      for (int i = 0; i < lhs.size(); i++)
        r &= lhs._values[i] != rhs._values[i];
      return r;
    }

    friend glvec3<T> cross(const glvec3<T> &lhs, const glvec3<T> &rhs)
    {
      glvec3<T> r;
        
      r.x = lhs.y * rhs.z - lhs.z * rhs.y;
      r.y = lhs.z * rhs.x - lhs.x * rhs.z;
      r.z = lhs.x * rhs.y - lhs.y * rhs.x;
      return r;
    }
   
    union {
      
      struct {
        T   x, y, z;
      };
      
      T _values[3];
    };
    
  };


  typedef glvec3<char>   glchar3;
  typedef glvec3<uchar>  gluchar3;
  typedef glvec3<short>  glshort3;
  typedef glvec3<ushort> glushort3;  
  typedef glvec3<int>    glint3;
  typedef glvec3<uint>   gluint3;  
  typedef glvec3<long>   gllong3;
  typedef glvec3<ulong>  glulong3;  
  typedef glvec3<float>  glfloat3;
  typedef glvec3<double> gldouble3;

  // ---- glvec4<T>
  // 
  template <typename T>
  class glvec4 {

   public:
    
    typedef T value_type;
    typedef unsigned short ushort;

    glvec4()
    {
      for(ushort i = 0; i < size(); ++i)
        _values[i] = T(0.0);
    }    

    glvec4(const T &v)
    {
      for(ushort i = 0; i < size(); ++i)
        _values[i] = v;
    }
    
    glvec4(const T &v0, const T &v1, const T &v2, const T &v3)
    {
      x = v0;
      y = v1;
      z = v2;
      w = v3;
    }

    glvec4(const T *v)
    {
      for(ushort i = 0; i < size(); ++i) 
        _values[i] = v[i];
    }

    explicit glvec4(const glvec4<T> &u)
    {
      for(ushort i = 0; i < size(); ++i)
        _values[i] = u[i];
    }

    ushort size() const 
    { return 4; }

    const T* address() const 
    { return _values; }

    glvec4<T> setValue(const T* rhs) 
    {
      for(ushort i = 0; i < size(); ++i)
        _values[i] = rhs[i];
      return *this;
    }
    
    T& operator[](int i)
    {
      assert(i < size());
      return _values[i];
    }

    const T& operator[](int i) const
    {
      assert(i < size());
      return _values[i];
    }

    operator T* () 
    { return _values; }

    operator const T* () const 
    { return _values; }

    friend glvec4<T> & operator *= (glvec4<T> &lhs, T d)
    {
      for(int i = 0; i < lhs.size(); i++)
        lhs._values[i] *= d;
      return lhs;
    }

    friend glvec4<T> & operator *= (glvec4<T> &lhs, const glvec4<T> &rhs) {
      for(int i = 0; i < lhs.size(); i++)
        lhs._values[i] *= rhs._values[i];
      return lhs;
    }

    friend glvec4<T> & operator /= (glvec4<T> &lhs, T d) {
      if(d == 0)
        return lhs;
      for(int i = 0; i < lhs.size(); i++)
        lhs._values[i] /= d;
      return lhs;
    }

    friend glvec4<T> & operator /= (glvec4<T> &lhs, const glvec4<T> & rhs)
    {
      for(int i = 0; i < lhs.size(); i++)
        lhs._values[i] /= rhs._values[i];
      return lhs;
    }

    friend glvec4<T> & operator += (glvec4<T> &lhs, const glvec4<T> & rhs)
    {
      for(int i = 0; i < lhs.size(); i++)
        lhs._values[i] += rhs._values[i];
      return lhs;
    }

    friend glvec4<T> & operator -= (glvec4<T> &lhs, const glvec4<T> & rhs)
    {
      for(int i = 0; i < lhs.size(); i++)
        lhs._values[i] -= rhs._values[i];
      return lhs;
    }

    friend glvec4<T> operator - (const glvec4<T> &rhs)
    {
      glvec4<T> rv;
      for(int i = 0; i < rhs.size(); i++)
        rv._values[i] = -rhs._values[i];
      return rv;
    }

    friend glvec4<T> operator + (const glvec4<T> & lhs, const glvec4<T> & rhs)
    {
      glvec4<T> rt(lhs);
      return rt += rhs;
    }

    friend glvec4<T> operator - (const glvec4<T> & lhs, const glvec4<T> & rhs)
    {
      glvec4<T> rt(lhs);
      return rt -= rhs;
    }

    friend glvec4<T> operator * (const glvec4<T> & lhs, T rhs)
    {
      glvec4<T> rt(lhs);
      return rt *= rhs;
    }

    friend glvec4<T> operator * (T lhs, const glvec4<T> & rhs)
    {
      glvec4<T> rt(lhs);
      return rt *= rhs;
    }

    friend glvec4<T> operator * (const glvec4<T> & lhs, const glvec4<T> & rhs)
    {
      glvec4<T> rt(lhs);
      return rt *= rhs;
    }

    friend glvec4<T> operator / (const glvec4<T> & lhs, T rhs)
    {
      glvec4<T> rt(lhs);
      return rt /= rhs;
    }

    friend glvec4<T> operator / (const glvec4<T> & lhs, const glvec4<T> & rhs)
    {
      glvec4<T> rt(lhs);
      return rt /= rhs;
    }

    friend bool operator == (const glvec4<T> &lhs, const glvec4<T> &rhs )
    {
      bool r = true;
      for (int i = 0; i < lhs.size(); i++)
        r &= lhs._values[i] == rhs._values[i];
      return r;
    }

    friend bool operator != ( const glvec4<T> &lhs, const glvec4<T> &rhs )
    {
      bool r = true;
      for (int i = 0; i < lhs.size(); i++)
        r &= lhs._values[i] != rhs._values[i];
      return r;
    }
   
    union {
      
      struct {
        T   x, y, z, w;
      };
      
      T _values[4];
    };
    
  };

  
  typedef glvec4<char>   glchar4;
  typedef glvec4<uchar>  gluchar4;
  typedef glvec4<short>  glshort4;
  typedef glvec4<ushort> glushort4;  
  typedef glvec4<int>    glint4;
  typedef glvec4<uint>   gluint4;  
  typedef glvec4<long>   gllong4;
  typedef glvec4<ulong>  glulong4;  
  typedef glvec4<float>  glfloat4;
  typedef glvec4<double> gldouble4;
  

  // dot product 
  template<class T>
  inline typename T::value_type dot(const T &lhs, const T& rhs ) { 
    typename T::value_type r = 0;
    for(int i = 0; i < lhs.size(); i++)
      r += lhs[i] * rhs[i];
    return r;
  }

  // length of glvector
  template< class T>
  inline typename T::value_type length(const T &glvec) {
    typename T::value_type r = 0;
    for(int i = 0; i < glvec.size(); i++)
      r += glvec[i] * glvec[i]; 
    return typename T::value_type(sqrt(r));
  }

  // return the squared norm
  template< class T>
  inline typename T::value_type square_norm( const T & glvec) {
    typename T::value_type r = 0;
    for(int i = 0; i < glvec.size(); i++)
      r += glvec[i] * glvec[i]; 
    return r;
  }

  // normalize glvector
  template< class T>
  inline T normalize( const T & glvec) { 
    typename T::value_type sum(0);
    T r;
    for(int i = 0; i < glvec.size(); ++i) 
      sum += glvec[i] * glvec[i];
    sum = typename T::value_type(sqrt(sum));
    if (sum > 0)
      for(int i = 0; i < glvec.size(); ++i) 
        r[i] = glvec[i] / sum;
    return r;
  }
}

#endif
